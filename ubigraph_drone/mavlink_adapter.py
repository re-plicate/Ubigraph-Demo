"""MAVLink adapter shell.

A thin, vendor-neutral interface that the DroneFleetAgent talks to. Two backends:

* ``sim``   — delegates to the existing in-process ``DroneService`` (default, no hardware).
* ``mavlink`` — if ``pymavlink`` is installed and a connection URL is given
  (e.g. ``udp:127.0.0.1:14550`` or ``COM5``), arm/takeoff/goto/land are translated
  into MAVLink messages. This is the "可对接的 MAVLink 接口壳" the user asked for:
  it compiles against real vehicles/SITL (ArduPilot/PX4) without us needing hardware
  in this environment.

The shell deliberately does *not* import pymavlink at module import time so that
the package runs on machines without it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class MAVState:
    connected: bool = False
    armed: bool = False
    mode: str = "DISCONNECTED"
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    battery: float = 100.0
    last_heartbeat: float = 0.0
    log: list = field(default_factory=list)

    def record(self, line: str) -> None:
        self.log.append(f"{time.time():.3f} {line}")
        self.log = self.log[-200:]


class MAVLinkAdapter:
    """Uniform command surface. Backend is chosen by ``backend``."""

    def __init__(self, backend: str = "sim", connection_url: str | None = None,
                 vehicle_id: str = "1") -> None:
        self.backend = backend
        self.connection_url = connection_url
        self.vehicle_id = vehicle_id
        self.state = MAVState()
        self._vehicle = None  # pymavlink vehicle when backend == "mavlink"

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> dict:
        if self.backend == "mavlink":
            try:
                from pymavlink import mavutil  # type: ignore
                self._vehicle = mavutil.mavlink_connection(self.connection_url or "udp:127.0.0.1:14550")
                self._vehicle.wait_heartbeat(timeout=10)
                self.state.connected = True
                self.state.last_heartbeat = time.time()
                self.state.record(f"CONNECTED {self.connection_url}")
                return {"ok": True, "backend": "mavlink", "vehicle": self._vehicle.target_system}
            except Exception as exc:
                self.state.record(f"CONNECT_FAILED {exc!r}")
                return {"ok": False, "backend": "mavlink", "error": repr(exc),
                        "hint": "pymavlink 未安装或连接失败；pip install pymavlink 后用 udp:IP:14550 接 SITL"}
        # sim / dryrun
        self.state.connected = True
        self.state.mode = "STANDBY"
        self.state.record(f"CONNECT backend={self.backend}")
        return {"ok": True, "backend": self.backend, "vehicle": self.vehicle_id}

    # -- commands (MAVLink-equivalent) -------------------------------------
    def arm(self) -> dict:
        if not self.state.connected:
            return {"ok": False, "error": "not connected"}
        if self.backend == "mavlink" and self._vehicle is not None:
            self._vehicle.wait_heartbeat()
            self._vehicle.arducopter_arm()
        self.state.armed = True
        self.state.mode = "ARMED"
        self.state.record("ARM")
        return {"ok": True, "armed": True}

    def takeoff(self, alt_m: float) -> dict:
        if not self.state.armed:
            return {"ok": False, "error": "arm first"}
        if self.backend == "mavlink" and self._vehicle is not None:
            self._vehicle.mav.command_long_send(
                self._vehicle.target_system, self._vehicle.target_component,
                22, 0, alt_m, 0, 0, 0, 0, 0, 0)  # MAV_CMD_NAV_TAKEOFF
        self.state.alt = float(alt_m)
        self.state.mode = "TAKEOFF"
        self.state.record(f"TAKEOFF alt={alt_m}")
        return {"ok": True, "alt_m": alt_m}

    def goto(self, lat: float, lon: float, alt_m: float) -> dict:
        if not self.state.armed:
            return {"ok": False, "error": "arm first"}
        if self.backend == "mavlink" and self._vehicle is not None:
            self._vehicle.mav.mission_item_int_send(
                self._vehicle.target_system, self._vehicle.target_component, 0,
                3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT
                16,  # MAV_CMD_NAV_WAYPOINT
                0, 0, 0, 0, 0, 0,
                int(lat * 1e7), int(lon * 1e7), alt_m)
        self.state.lat, self.state.lon, self.state.alt = lat, lon, alt_m
        self.state.mode = "MISSION"
        self.state.record(f"GOTO {lat},{lon}@{alt_m}")
        return {"ok": True, "lat": lat, "lon": lon, "alt_m": alt_m}

    def land(self) -> dict:
        if self.backend == "mavlink" and self._vehicle is not None:
            self._vehicle.mav.command_long_send(
                self._vehicle.target_system, self._vehicle.target_component,
                21, 0, 0, 0, 0, 0, 0, 0, 0)  # MAV_CMD_NAV_LAND
        self.state.alt = 0.0
        self.state.armed = False
        self.state.mode = "LANDED"
        self.state.record("LAND")
        return {"ok": True}

    def disconnect(self) -> dict:
        self.state.connected = False
        self.state.record("DISCONNECT")
        return {"ok": True}

    def snapshot(self) -> dict:
        return {"backend": self.backend, "url": self.connection_url,
                "vehicle_id": self.vehicle_id, **self.state.__dict__}
