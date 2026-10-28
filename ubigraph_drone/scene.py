"""A registered, explicitly synthetic metre-grid scene; not a map of Aarhus."""
from dataclasses import asdict, dataclass, field
import math

SLOT = 2.0
HEIGHT = .6
RESERVE = 15.0
SEPARATION = .65
WIDTH, HEIGHT_CELLS = 20, 16
OBSTACLES = {(10, y) for y in range(1, 14) if y not in (5, 10)}
NODES = {
    "G1": {"x": 5, "y": 2, "parking_alias": "BRUUNS"},
    "G2": {"x": 8, "y": 4, "parking_alias": "KALKVAERKSVEJ"},
    "G3": {"x": 15, "y": 2, "parking_alias": "MAGASIN"},
    "G4": {"x": 16, "y": 8, "parking_alias": "SALLING"},
    "G5": {"x": 12, "y": 13, "parking_alias": "SCANDCENTER"},
    "G6": {"x": 5, "y": 12, "parking_alias": "SKOLEBAKKEN"},
    "G7": {"x": 7, "y": 8, "parking_alias": None},
    "G8": {"x": 17, "y": 13, "parking_alias": None},
}
HOMES = {"1": (1, 2), "2": (1, 7), "3": (1, 13)}


def number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name}必须是{low}至{high}之间的有限数值")
    return float(value)


def node_id(value):
    if not isinstance(value, str):
        raise ValueError("目标必须是已登记的网格或停车场编号")
    name = value.strip().upper()
    for key, v in NODES.items():
        if name in (key, v["parking_alias"]):
            return key
    raise ValueError("未知目标；可选 G1—G8 或已登记停车场别名")


def scene_info():
    return {"id": "synthetic-grid-v1", "name": "补采调度模拟场景", "width": WIDTH,
            "height": HEIGHT_CELLS, "unit": "m", "nodes": NODES,
            "obstacles": sorted(OBSTACLES), "homes": HOMES, "slot_seconds": SLOT,
            "minimum_separation_m": SEPARATION, "backend": "simulation",
            "note": "位置为模拟布置；停车场名称只是业务别名，不表示真实地理位置。电量是仿真假设。"}


@dataclass
class Drone:
    id: str
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    battery: float = 100.0
    status: str = "idle"
    connected: bool = True
    mission_id: str | None = None
    fault: str | None = None
    distance: float = 0.0
    velocity: list = field(default_factory=lambda: [0., 0., 0.])
    trail: list = field(default_factory=list)

    def public(self, trails=True):
        value = asdict(self)
        value["home"] = HOMES[self.id]
        if not trails:
            value.pop("trail")
        return value

    def track(self, reference, feedforward, dt):
        """World feedforward + P correction -> body hover command -> kinematic plant.

        x_body is forward, y_body is left; yaw is degrees. Ideal, noise-free plant.
        This is a motion/control simulator, not a rigid-body aerodynamics model.
        """
        old = (self.x, self.y, self.z)
        vx = feedforward[0] + 2.0 * (reference[0] - self.x)
        vy = feedforward[1] + 2.0 * (reference[1] - self.y)
        scale = min(1., .8 / max(math.hypot(vx, vy), 1e-12))
        vx, vy = vx * scale, vy * scale
        c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        bx, by = c * vx + s * vy, -s * vx + c * vy
        self.x += (c * bx - s * by) * dt
        self.y += (s * bx + c * by) * dt
        vz = max(-.6, min(.6, feedforward[2] + 2.0 * (reference[2] - self.z)))
        self.z = max(0., self.z + vz * dt)
        for key in ("x", "y", "z"):
            if abs(getattr(self, key)) < 1e-9:
                setattr(self, key, 0.)
        self.velocity = [vx, vy, vz]
        distance = math.dist(old, (self.x, self.y, self.z))
        self.distance += distance
        self.battery = max(0., self.battery - .28 * distance - (.035 * dt if self.z > .02 else 0))
        if not self.trail or math.dist(self.trail[-1], (self.x, self.y)) >= .15:
            self.trail.append([self.x, self.y])
            self.trail = self.trail[-400:]
        return {"vx_body": bx, "vy_body": by, "height": reference[2]}
