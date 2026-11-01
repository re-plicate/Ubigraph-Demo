"""Persistent asynchronous simulated fleet. No cflib imports or hardware fallback."""
import bisect
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from .commands import manifest
from .planner import assign, build_routes, cell, energy_requirements
from .scene import Drone, HEIGHT, HOMES, NODES, RESERVE, SEPARATION, SLOT, node_id, number, scene_info

TERMINAL = {"completed", "cancelled", "failed", "interrupted"}
DT = .05


def identity(prefix):
    return prefix + "_" + uuid.uuid4().hex[:12]


class DroneService:
    def __init__(self, directory, autostart=True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / "fleet.sqlite3"
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.stopping = threading.Event()
        self.sim_time, self.speed, self.paused, self.revision = 0., 8., False, 0
        self.drones = {key: Drone(key, *p) for key, p in HOMES.items()}
        self.plans, self.missions, self.events = {}, {}, []
        self.active = None
        self.epoch = identity("scene")
        self.last_error, self._closed = None, False
        self._lease = sqlite3.connect(self.directory/"fleet-owner.sqlite3", timeout=0, check_same_thread=False)
        try:
            self._lease.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError:
            self._lease.close()
            raise ValueError('Simulation runtime directory is in use. Reuse the running service or select another runtime directory.') from None
        try:
            self._restore()
            self._save()
        except BaseException:
            self._lease.close()
            raise
        self.thread = None
        if autostart:
            self.thread = threading.Thread(target=self._loop, daemon=True, name="ubigraph-simulated-fleet")
            self.thread.start()

    def _restore(self):
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS snapshot(id INTEGER PRIMARY KEY,value TEXT)")
            row = db.execute("SELECT value FROM snapshot WHERE id=1").fetchone()
        if row:
            saved = json.loads(row[0])
            for key in ("sim_time", "speed", "revision", "plans", "missions", "events", "epoch"):
                if key in saved:
                    setattr(self, key, saved[key])
            self.drones = {key: Drone(**{k: v for k, v in d.items() if k != "home"}) for key, d in saved["drones"].items()}
            for mission in self.missions.values():
                if mission["status"] not in TERMINAL:
                    mission["status"] = "interrupted"
                    mission["reason"] = 'Service restarted: simulation interrupted without resuming flights; simulated drones placed on ground'
                    mission["finished_sim_time"] = self.sim_time
                    mission["actual_duration_seconds"] = max(0., self.sim_time - mission["started_sim_time"])
                    for route in mission["routes"].values():
                        for job in route["jobs"]:
                            if job["status"] != "completed":
                                job["status"] = "interrupted"
            for d in self.drones.values():
                d.z, d.mission_id, d.velocity = 0., None, [0., 0., 0.]
                d.status = "fault" if d.fault else "idle"
            self.revision += 1  # recovery changes poses, so unstarted plans must be regenerated
            self._event("recovered", 'Service restarted. Incomplete missions are marked interrupted; simulated drones are on the ground.', save=False)

    def _db(self):
        return sqlite3.connect(self.db_path)

    def _save(self):
        value = {key: getattr(self, key) for key in ("sim_time", "speed", "revision", "plans", "missions", "events", "epoch")}
        value["drones"] = {key: d.public() for key, d in self.drones.items()}
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO snapshot VALUES(1,?)", (json.dumps(value, ensure_ascii=False, allow_nan=False),))

    def _event(self, kind, message, mission_id=None, drone_id=None, node=None, save=True):
        event = {"id": self.events[-1]["id"] + 1 if self.events else 1, "type": kind,
                 "message": message, "sim_time": round(self.sim_time, 3), "wall_time": time.time(),
                 "mission_id": mission_id or self.active, "drone_id": drone_id, "node": node}
        self.events.append(event)
        self.events = self.events[-1200:]
        if save:
            self._save()
        with self.condition:
            self.condition.notify_all()
        return event

    def _snapshot(self):
        return {"kind": "drone_fleet", "backend": "simulation", "scene": scene_info(),
                "epoch": self.epoch, "sim_time": round(self.sim_time, 3), "speed": self.speed,
                "paused": self.paused, "revision": self.revision, "active_mission_id": self.active,
                "last_error": self.last_error, "worker_running": bool(self.thread and self.thread.is_alive()),
                "fleet": [d.public() for d in self.drones.values()],
                "missions": list(self.missions.values())[-20:], "events": self.events[-80:],
                "note": 'Simulated missions use externally supplied targets. No physical hardware or V3 observation writeback is connected.'}

    def snapshot(self, compact=False):
        with self.lock:
            value = self._snapshot()
            if compact:
                value["fleet"] = [d.public(False) for d in self.drones.values()]
                value["missions"] = [{k: m[k] for k in ("id", "request_id", "status", "mode", "reason", "completed_targets", "targets")} for m in value["missions"]]
                value["events"] = value["events"][-8:]
            return copy.deepcopy(value)

    def _free(self, allow_error=False):
        if self.stopping.is_set():
            raise ValueError('Simulation service closed')
        if self.last_error and not allow_error:
            raise ValueError('Simulator execution error. Reset the scene first.')
        if self.active:
            raise ValueError('A mission already occupies the simulated airspace. Check or cancel it first.')

    @staticmethod
    def _request(value):
        if not isinstance(value, str) or not 1 <= len(value) <= 80 or any(ord(c) < 32 for c in value):
            raise ValueError('request_id must contain 1 to 80 printable characters for deduplication')
        return value

    def _selected(self, values):
        if not isinstance(values, list) or not values or len(values) > 3:
            raise ValueError('Select 1 to 3 distinct drone IDs')
        if any(not isinstance(k, str) or k not in self.drones for k in values):
            raise ValueError('Unknown drone. Valid IDs are 1, 2, and 3')
        if len(set(values)) != len(values):
            raise ValueError('Drone IDs must be distinct')
        return [self.drones[k] for k in sorted(values)]

    def _cached_request(self, request_id, payload):
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        for plan in self.plans.values():
            if plan["request_id"] == request_id:
                if plan["payload_hash"] != digest:
                    raise ValueError('This request_id was already used with different content. Use a new request ID.')
                return digest, copy.deepcopy(plan)
        return digest, None

    def plan(self, request_id, targets, drone_ids=None, priority_targets=None, dwell_seconds=3., return_to_base=True):
        with self.lock:
            self._request(request_id)
            if type(return_to_base) is not bool:
                raise ValueError('return_to_base must be a boolean')
            dwell_seconds = number(dwell_seconds, 'Collection duration', 0, 30)
            if not isinstance(targets, list) or not 1 <= len(targets) <= 8:
                raise ValueError('Provide 1 to 8 target nodes')
            targets = [node_id(k) for k in targets]
            if len(set(targets)) != len(targets):
                raise ValueError('Duplicate targets are not allowed, including garage aliases and grid IDs')
            priority_targets = [] if priority_targets is None else priority_targets
            if not isinstance(priority_targets, list):
                raise ValueError('Priority targets must be an array')
            priority = [node_id(k) for k in priority_targets]
            if len(set(priority)) != len(priority) or not set(priority).issubset(targets):
                raise ValueError('Priority targets must be a distinct subset of selected targets')
            ids = [d.id for d in self._selected(list(self.drones) if drone_ids is None else drone_ids)]
            payload = {"targets": targets, "drone_ids": ids, "priority_targets": priority,
                       "dwell_seconds": dwell_seconds, "return_to_base": return_to_base}
            digest, cached = self._cached_request(request_id, payload)
            if cached:
                return cached
            self._free()
            selected = [self.drones[k] for k in ids if self.drones[k].connected and not self.drones[k].fault]
            ordered = priority + [k for k in targets if k not in priority]
            orders, rejected = assign(selected, ordered, dwell_seconds, return_to_base, list(self.drones.values()))
            routes, energy = {}, {}
            if not rejected:
                try:
                    routes = build_routes(selected, orders, dwell_seconds, return_to_base, list(self.drones.values()))
                    energy = energy_requirements(routes, list(self.drones.values()))
                    for key, need in energy.items():
                        if self.drones[key].battery - need < RESERVE:
                            raise ValueError(f"Drone {key} has insufficient return battery in the plan including waits")
                except ValueError as exc:
                    rejected = [{"target": "plan", "reason": str(exc)}]
            plan = {"kind": "drone_plan", "id": identity("plan"), "request_id": request_id,
                    "payload_hash": digest, "epoch": self.epoch, "revision": self.revision,
                    "mode": "collection", "targets": ordered, "routes": routes,
                    "feasible": not rejected and bool(routes), "rejected": rejected,
                    "return_to_base": return_to_base, "dwell_seconds": dwell_seconds,
                    "energy_required_percent": energy,
                    "duration_seconds": max((r["duration_seconds"] for r in routes.values()), default=0),
                    "distance_m": sum(r["distance_m"] for r in routes.values()),
                    "algorithm": 'External priorities, estimated-completion-time assignment, and space-time A* with vertex and reverse-edge reservations',
                    "note": 'No takeoff yet. Duration and battery are motion-simulation estimates. Priority affects assignment order, not guaranteed completion order.'}
            plan["command_manifest"] = manifest(plan)
            self.plans[plan["id"]] = plan
            self._event("planned", 'Mission plan generated' if plan["feasible"] else 'Planning infeasible. See rejection reasons.')
            return copy.deepcopy(plan)

    def command_manifest(self, plan_id):
        with self.lock:
            if plan_id not in self.plans:
                raise ValueError('Plan not found')
            plan = self.plans[plan_id]
            value = copy.deepcopy(plan.get("command_manifest") or manifest(plan))
            value["stale"] = plan["epoch"] != self.epoch or plan["revision"] != self.revision
            value["started"] = any(m["plan_id"] == plan_id for m in self.missions.values())
            return value

    def start(self, plan_id):
        with self.lock:
            if plan_id not in self.plans:
                raise ValueError('Plan not found. Generate a plan first.')
            plan = self.plans[plan_id]
            for m in self.missions.values():
                if m["plan_id"] == plan_id:
                    return self._mission(m["id"])
            self._free()
            if plan["epoch"] != self.epoch or plan["revision"] != self.revision:
                raise ValueError('The plan is stale because fleet state changed. Replan with a new request_id.')
            if not plan["feasible"]:
                raise ValueError('The plan is infeasible and cannot execute. Inspect the rejection reasons.')
            energy = energy_requirements(plan["routes"], list(self.drones.values()), plan["mode"] not in ("return", "land"))
            for key, need in energy.items():
                d = self.drones[key]
                if d.fault or not d.connected or (plan["mode"] not in ("return", "land") and d.battery - need < RESERVE):
                    raise ValueError('Preflight check failed: drone state or return battery changed. Generate a new plan.')
            mid = identity("mission")
            mission = {"kind": "drone_mission", "id": mid, "request_id": plan["request_id"],
                       "plan_id": plan_id, "epoch": self.epoch, "mode": plan["mode"],
                       "status": "running", "reason": 'Mission accepted; simulation running',
                       "targets": plan["targets"], "routes": copy.deepcopy(plan["routes"]),
                       "started_sim_time": self.sim_time, "segment_start": self.sim_time,
                       "duration_seconds": plan["duration_seconds"], "planned_distance_m": plan["distance_m"],
                       "completed_targets": [], "receipts": [], "distance_m": 0.,
                       "min_separation_m": None, "cancel_at": None, "cancel_reason": None,
                       "fault_drone": None, "fault_drone_ids": [], "final_status": None, "finished_sim_time": None}
            self.missions[mid], self.active = mission, mid
            for key in mission["routes"]:
                self.drones[key].mission_id = mid
            self.revision += 1
            self._event("started", 'Simulation mission started. Completion is determined by execution receipts.', mid)
            return self._mission(mid)

    def _mission(self, mid):
        if mid == "latest":
            mid = self.active or next(reversed(self.missions), None)
        if not mid or mid not in self.missions:
            raise ValueError('Task not found')
        value = copy.deepcopy(self.missions[mid])
        value["events"] = [e for e in self.events if e["mission_id"] == mid][-100:]
        return value

    def mission(self, mid):
        with self.lock:
            return self._mission(mid)

    def cancel(self, mid, reason='User cancelled remaining visits; preparing return'):
        with self.lock:
            m = self.missions[self._mission(mid)["id"]]
            if m["status"] in TERMINAL or m["cancel_at"] is not None or m["status"] == "returning":
                return self._mission(m["id"])
            self._request_return(m, reason)
            return self._mission(m["id"])

    def _request_return(self, m, reason, failed=False):
        elapsed = self.sim_time - m["segment_start"]
        if m["cancel_at"] is None:
            m["cancel_at"] = max(1, math.ceil((elapsed + .001) / SLOT)) * SLOT
        m["status"], m["cancel_reason"] = "cancelling", reason
        m["final_status"] = "failed" if failed or m["final_status"] == "failed" else "cancelled"
        self.paused = False
        m["reason"] = reason + '; replan the return route after completing the current segment'
        self._event("cancel_requested", m["reason"], m["id"])

    def control(self, request_id, drone_id, action, target="NONE", height=HEIGHT, duration_seconds=3.):
        with self.lock:
            self._request(request_id)
            drone = self._selected([drone_id])[0]
            height = number(height, 'Altitude', .2, 1.)
            duration_seconds = number(duration_seconds, 'Hover duration', 0, 30)
            if action not in ("takeoff", "navigate", "hover", "return", "land"):
                raise ValueError('Unsupported action. Use takeoff, navigate, hover, return, or land.')
            target_id = node_id(target) if action == "navigate" else None
            payload = {"mode": action, "drone_id": drone_id, "target": target_id,
                       "height": height, "duration_seconds": duration_seconds}
            digest, old = self._cached_request(request_id, payload)
            if old:
                return self.start(old["id"])
            self._free()
            if drone.fault or not drone.connected:
                raise ValueError('Drone faulted or disconnected. Restore simulation state first.')
            if action == "takeoff" and drone.z > .05:
                raise ValueError('Drone is already airborne')
            if action != "takeoff" and drone.z < .05:
                raise ValueError('Drone is on the ground. Take off first.')
            routes = build_routes([drone], {drone_id: [target_id] if target_id else []}, duration_seconds,
                                  action == "return", list(self.drones.values()), mode=action,
                                  height=height if action == "takeoff" else drone.z)
            route = routes[drone_id]
            energy = energy_requirements(routes, list(self.drones.values()), action not in ("return", "land"))
            if action not in ("return", "land") and any(self.drones[k].battery - need < RESERVE for k, need in energy.items()):
                raise ValueError('Insufficient simulated battery; return reserve required')
            plan = {"kind": "drone_plan", "id": identity("plan"), "request_id": request_id,
                    "payload_hash": digest, "epoch": self.epoch, "revision": self.revision,
                    "feasible": True, "rejected": [], "routes": routes, "mode": action,
                    "targets": [target_id] if target_id else [], "duration_seconds": route["duration_seconds"],
                    "distance_m": route["distance_m"], "energy_required_percent": energy}
            plan["command_manifest"] = manifest(plan)
            self.plans[plan["id"]] = plan
            return self.start(plan["id"])

    def settings(self, paused=None, speed=None):
        with self.lock:
            if paused is not None and type(paused) is not bool:
                raise ValueError('paused must be a boolean')
            if paused is False and self.last_error:
                raise ValueError('Simulator execution error. Reset the scene before continuing.')
            checked_speed = self.speed if speed is None else number(speed, 'Simulation speed', .5, 16)
            if paused is not None:
                self.paused = paused
            self.speed = checked_speed
            self._save()
            self.condition.notify_all()
            return self.snapshot()

    def configure_drone(self, drone_id, battery=None, fault=None):
        with self.lock:
            d = self._selected([drone_id])[0]
            checked_battery = d.battery if battery is None else number(battery, 'Simulated battery', 0, 100)
            if fault is not None:
                if type(fault) is not bool:
                    raise ValueError('fault must be a boolean')
                if not fault and self.active:
                    raise ValueError('Wait for the current mission to terminate before restoring a faulted drone')
            d.battery = checked_battery
            if fault is not None:
                d.fault = 'Simulated fault' if fault else None
            self.revision += 1
            if self.active and d.mission_id == self.active and (d.fault or d.battery < RESERVE):
                m = self.missions[self.active]
                if d.fault:
                    m["fault_drone"] = d.id
                    m["fault_drone_ids"] = sorted(set(m.get("fault_drone_ids", [])) | {d.id})
                self._request_return(m, 'Simulated fault or low battery: stop collection and handle return', failed=True)
            elif d.mission_id is None:
                d.status = "fault" if d.fault else ("hovering" if d.z > .05 else "idle")
            self._event("fleet_changed", f"Drone {d.id} simulation state updated", drone_id=d.id)
            return self.snapshot()

    def reset(self):
        with self.lock:
            self._free(allow_error=True)
            self.drones = {key: Drone(key, *p) for key, p in HOMES.items()}
            self.sim_time, self.paused = 0., False
            self.last_error = None
            self.revision += 1
            self.epoch = identity("scene")
            self._event("reset", 'Simulation scene reset. Mission records preserved.')
            return self.snapshot()

    @staticmethod
    def _reference(route, elapsed):
        path = route["path"]
        tick = max(0., elapsed / SLOT)
        i = max(0, min(len(path) - 1, bisect.bisect_right([p["tick"] for p in path], tick) - 1))
        a = path[i]
        if i == len(path) - 1:
            return (a["x"], a["y"], a["z"]), (0., 0., 0.), "hovering" if a["z"] else "idle"
        b = path[i + 1]
        fraction = min(1., max(0., tick - a["tick"]))
        pos = tuple(a[k] + (b[k] - a[k]) * fraction for k in ("x", "y", "z"))
        ff = tuple((b[k] - a[k]) / SLOT for k in ("x", "y", "z"))
        return pos, ff, b["phase"]

    def _begin_return(self, m):
        for route in m["routes"].values():
            for job in route["jobs"]:
                if job["status"] != "completed":
                    job["status"] = "cancelled"
        good = []
        for key in m["routes"]:
            d = self.drones[key]
            if d.fault:
                # Explicit simulation fault injection, not a successful landing command.
                d.z, d.velocity, d.status, d.mission_id = 0., [0., 0., 0.], "fault", None
                self._event("fault", 'Faulted simulated drone stopped and placed on ground; collection incomplete', m["id"], key)
            elif d.z < .05:
                # A shorter route may already have landed. Cancellation never relaunches it.
                d.z, d.velocity, d.status, d.mission_id = 0., [0., 0., 0.], "idle", None
            else:
                good.append(d)
        m.setdefault("original_routes", copy.deepcopy(m["routes"]))
        try:
            m["routes"] = build_routes(good, {}, 0, True, list(self.drones.values()), mode="return")
            m["segment_start"], m["cancel_at"], m["status"] = self.sim_time, None, "returning"
            m["reason"] = m["cancel_reason"] + '; returning to base; the mission has not finished'
            self._event("returning", m["reason"], m["id"])
        except ValueError as exc:
            for d in good:
                d.z, d.velocity = 0., [0., 0., 0.]
            m["reason"] = f"Return planning failed. Simulated drone stopped and placed on ground:{exc}"
            m["failure_reason"] = m["reason"]
            self._finish(m, "failed")

    def _finish(self, m, status, persist=True):
        m["status"], m["finished_sim_time"] = status, self.sim_time
        m["actual_duration_seconds"] = self.sim_time - m["started_sim_time"]
        if status == "completed":
            m["reason"] = 'Simulation complete. Collection receipts describe actions and contain no measured parking availability.' if m["mode"] == "collection" else 'Simulated control action completed'
        elif status == "cancelled":
            m["reason"] = m.get("cancel_reason") or m["reason"]
        else:
            m["reason"] = m.get("failure_reason") or m.get("cancel_reason") or m["reason"]
        for d in self.drones.values():
            if d.mission_id == m["id"]:
                d.mission_id, d.velocity = None, [0., 0., 0.]
                d.status = "fault" if d.fault else ("hovering" if d.z > .05 else "idle")
        self.active = None
        self.revision += 1
        self._event(status, m["reason"], m["id"], save=persist)

    def _fail_and_land(self, m, reason, persist=True):
        m["failure_reason"] = reason
        for routes in (m["routes"], m.get("original_routes", {})):
            for route in routes.values():
                for job in route["jobs"]:
                    if job["status"] != "completed":
                        job["status"] = "failed"
        for d in self.drones.values():
            if d.mission_id == m["id"]:
                d.z, d.velocity = 0., [0., 0., 0.]
        self._finish(m, "failed", persist=persist)

    @staticmethod
    def _at(drone, point):
        return math.dist((drone.x, drone.y, drone.z), (point["x"], point["y"], point["z"])) <= .05

    def _step(self, dt):
        # Caller holds lock. At t, track the t reference over [t,t+dt].
        if self.active:
            m = self.missions[self.active]
            elapsed = max(0., self.sim_time - m["segment_start"])
            for key, route in m["routes"].items():
                d = self.drones[key]
                ref, ff, phase = self._reference(route, elapsed + 1e-9)
                old = d.distance
                d.track(ref, ff, dt)
                m["distance_m"] += d.distance - old
                if d.status != phase:
                    d.status = phase
                    self._event("phase", f"Drone {key} entered the {phase} phase", m["id"], key, save=False)
            self.sim_time += dt
            elapsed += dt
            if any(d.battery <= 0 and d.z > .05 for d in self.drones.values()):
                self._fail_and_land(m, 'Simulated battery depleted. Stopped and placed on ground; normal return not claimed.')
                return
            if m["status"] == "running":
                for key, route in m["routes"].items():
                    for job in route["jobs"]:
                        if job["status"] == "pending" and elapsed + 1e-7 >= job["arrival_tick"] * SLOT:
                            if not self._at(self.drones[key], route["path"][job["arrival_tick"]]):
                                self._fail_and_land(m, 'The drone has not reached the target; no arrival or collection receipt can be issued.')
                                return
                            job["status"] = "collecting"
                            self._event("arrived", f"Drone {key} reached {job['node']}; starting simulated collection", m["id"], key, job["node"], save=False)
                        if job["status"] == "collecting" and elapsed + 1e-7 >= job["end_tick"] * SLOT:
                            if not self._at(self.drones[key], route["path"][job["end_tick"]]):
                                self._fail_and_land(m, 'Drone did not hold target position at collection end. Success receipt refused.')
                                return
                            job["status"] = "completed"
                            receipt = {"node": job["node"], "drone_id": key, "status": "completed",
                                       "source": "simulated_action_receipt", "measurement": None,
                                       "observed_at": None, "completed_sim_time": round(self.sim_time, 3),
                                       "note": 'Only simulated collection actions; no parking observations are generated or stored'}
                            job["receipt"] = receipt
                            m["receipts"].append(receipt)
                            m["completed_targets"].append(job["node"])
                            self._event("collected", f"Simulated collection completed at {job['node']}", m["id"], key, job["node"], save=False)
                low = [d.id for d in self.drones.values() if d.z > .05 and not d.fault and d.battery < RESERVE]
                if low and m["mode"] not in ("return", "land"):
                    self._request_return(m, 'Airborne drone reached return-battery reserve; further collection stopped', failed=True)
            active_drones = [d for d in self.drones.values() if d.z > .05 or d.mission_id]
            for i, a in enumerate(active_drones):
                for b in active_drones[i + 1:]:
                    gap = math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
                    m["min_separation_m"] = gap if m["min_separation_m"] is None else min(gap, m["min_separation_m"])
                    if gap < SEPARATION - 1e-6:
                        for d in active_drones:
                            d.z, d.velocity = 0., [0., 0., 0.]
                            d.status = "fault" if d.fault else "idle"
                        self._fail_and_land(m, 'Separation conflict during simulation. Stopped and placed on ground; completion not claimed.')
                        return
            if m["cancel_at"] is not None and elapsed + 1e-7 >= m["cancel_at"]:
                self._begin_return(m)
                return
            end = max((r["duration_seconds"] for r in m["routes"].values()), default=0)
            if elapsed + 1e-7 >= end:
                if any(not self._at(self.drones[key], r["path"][-1]) for key, r in m["routes"].items()):
                    self._fail_and_land(m, 'Controller has not reached the final position. Mission cannot be marked complete.')
                    return
                self._finish(m, m["final_status"] or "completed")
        else:
            self.sim_time += dt
        # Hovering outside missions still consumes the assumed hover energy.
        for d in self.drones.values():
            if d.z > .05 and d.mission_id is None:
                d.battery = max(0., d.battery - .035 * dt)
        if not self.active:
            low = next((d for d in self.drones.values() if d.z > .05 and d.battery < RESERVE and not d.fault), None)
            if low:
                try:
                    self.control(identity("lowbattery"), low.id, "return")
                except ValueError as exc:
                    low.z, low.velocity, low.status = 0., [0., 0., 0.], "fault"
                    low.fault = 'Automatic return failed'
                    self.revision += 1
                    self._event("fault", f"Drone {low.id} cannot return automatically; placed on simulated ground: {exc}", drone_id=low.id)

    def advance(self, seconds):
        """Deterministic simulation clock for acceptance tests; not a public HTTP command."""
        with self.lock:
            for _ in range(round(seconds / DT)):
                self._step(DT)
            self._save()
            return self.snapshot()

    def _loop(self):
        last, accumulator, saved = time.monotonic(), 0., time.monotonic()
        while not self.stopping.wait(.025):
            now = time.monotonic()
            with self.lock:
                try:
                    if not self.paused:
                        accumulator += min(now - last, .2) * self.speed
                        while accumulator + 1e-9 >= DT:
                            self._step(DT)
                            accumulator -= DT
                    if now - saved > 1:
                        self._save()
                        saved = now
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"[:300]
                    if self.last_error != reason:
                        self.last_error = reason
                        if self.active:
                            self._fail_and_land(self.missions[self.active], 'Simulator error; stopped and placed on ground:' + reason, persist=False)
                        self._event("engine_error", 'Simulator paused:' + reason, save=False)
                    self.paused, accumulator, saved = True, 0., now
                    try:
                        self._save()
                    except Exception:
                        pass  # last_error remains available in the live state even if storage failed
                last = now
                self.condition.notify_all()

    def close(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=3)
        with self.lock:
            if not self._closed:
                try:
                    self._save()
                finally:
                    self._lease.close()
                    self._closed = True
