"""Strict fixed-function Agent tools. The LLM never emits executable flight code."""
import copy
import math

from .scene import NODES

LABELS = {
    "get_fleet_status": ("Fleet tools", "Inspect simulated fleet"),
    "plan_collection_mission": ("Dispatch tools", "Plan collection for supplied targets"),
    "start_collection_mission": ("Fleet tools", "Submit simulation mission"),
    "get_mission_status": ("Fleet tools", "Read actual mission progress"),
    "cancel_mission": ("Fleet tools", "Cancel remaining visits & return"),
    "control_drone": ("Fleet tools", "Execute simulated control action"),
}


def schemas():
    nodes = list(NODES) + [n["parking_alias"] for n in NODES.values() if n["parking_alias"]]
    text = {"type": "string", "minLength": 1, "maxLength": 80}
    targets = {"type": "array", "items": {"type": "string", "enum": nodes}, "minItems": 1, "maxItems": 8}
    specs = [
        ("get_fleet_status", "Read the actual three-drone simulator, G1-G8 nodes, garage aliases and latest mission. No hardware or autonomous collection-need assessment.", {}),
        ("plan_collection_mission", "Plan supplied targets without execution. Default drone_ids=[1,2,3], dwell_seconds=3, return_to_base=true, priority_targets=[]. Use a unique request_id for new content; repeated identical requests return the same plan.", {
            "request_id": text, "targets": targets,
            "drone_ids": {"type": "array", "items": {"type": "string", "enum": ["1", "2", "3"]}, "minItems": 1, "maxItems": 3},
            "priority_targets": {**targets, "minItems": 0},
            "dwell_seconds": {"type": "number", "minimum": 0, "maximum": 30},
            "return_to_base": {"type": "boolean"}}),
        ("start_collection_mission", "Start an existing feasible plan only when execution is requested and allowed. Returns a background mission. Accepted is not completed.", {"plan_id": text}),
        ("get_mission_status", "Read actual mission progress, completed targets and receipts. mission_id may be latest. Never infer completion from estimates.", {"mission_id": text}),
        ("cancel_mission", "Cancel remaining visits. Drones return after the current segment; cancellation does not imply landing. mission_id may be latest.", {"mission_id": text}),
        ("control_drone", "Control a selected simulated drone: takeoff, navigate, hover, return or land. All except takeoff require airborne state. Use target=NONE for non-navigation actions. Defaults height=0.6 and duration_seconds=3. Returns an asynchronous mission.", {
            "request_id": text, "drone_id": {"type": "string", "enum": ["1", "2", "3"]},
            "action": {"type": "string", "enum": ["takeoff", "navigate", "hover", "return", "land"]},
            "target": {"type": "string", "enum": ["NONE"] + nodes},
            "height": {"type": "number", "minimum": .2, "maximum": 1.},
            "duration_seconds": {"type": "number", "minimum": 0, "maximum": 30}}),
    ]
    return [{"type": "function", "name": name, "description": desc, "strict": True,
             "parameters": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}}
            for name, desc, props in specs]


def check(value, rule, path="参数"):
    kind = rule["type"]
    valid = {"object": lambda: isinstance(value, dict), "array": lambda: isinstance(value, list),
             "string": lambda: isinstance(value, str), "boolean": lambda: type(value) is bool,
             "number": lambda: type(value) in (int, float) and math.isfinite(value)}[kind]()
    if not valid:
        raise ValueError(path + "类型无效")
    if "enum" in rule and value not in rule["enum"]:
        raise ValueError(path + "超出登记范围")
    if kind == "object":
        if set(value) != set(rule["properties"]):
            raise ValueError(path + "字段不匹配")
        for key, child in rule["properties"].items():
            check(value[key], child, path + "." + key)
    elif kind == "array":
        if not rule["minItems"] <= len(value) <= rule["maxItems"]:
            raise ValueError(path + "数量超出范围")
        for item in value:
            check(item, rule["items"], path + "[]")
        if rule.get("uniqueItems") and len(set(value)) != len(value):
            raise ValueError(path + "不能重复")
    elif kind == "number" and not rule["minimum"] <= value <= rule["maximum"]:
        raise ValueError(path + "数值超出范围")
    elif kind == "string" and not rule.get("minLength", 0) <= len(value) <= rule.get("maxLength", 1000):
        raise ValueError(path + "长度超出范围")


def compact(value):
    value = copy.deepcopy(value)
    if "routes" in value:
        value["routes"] = {key: {k: v for k, v in r.items() if k != "path"} | {"waypoint_count": len(r["path"])} for key, r in value["routes"].items()}
    value.pop("original_routes", None)
    # Keep the plan command manifest available to chat and the review workspace.
    if "events" in value:
        value["events"] = value["events"][-10:]
    value["backend"] = "simulation"
    return value


def execute(service, name, arguments):
    spec = next((s for s in schemas() if s["name"] == name), None)
    if not spec:
        raise ValueError("Drone tool is not registered")
    check(arguments, spec["parameters"])
    if name == "get_fleet_status":
        return service.snapshot(True)
    handlers = {"plan_collection_mission": service.plan, "start_collection_mission": service.start,
                "get_mission_status": service.mission, "cancel_mission": service.cancel, "control_drone": service.control}
    if "mission_id" in arguments:
        return compact(handlers[name](arguments["mission_id"]))
    return compact(handlers[name](**arguments))
