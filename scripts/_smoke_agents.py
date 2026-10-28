from ubigraph_agents.system import build_system
from ubigraph_agents.base import Envelope

b = build_system(run="models/parking")
print("agents:", list(b.agents))

# HIC trigger on high-spread references
out = b.dispatch("t", "hic", Envelope("evaluate", {
    "prediction": {
        "temporal_references": {"g": [{"delta_rate": 0.5}, {"delta_rate": -0.5}]},
        "current_input_status": {"a": "available"},
    }
}))
print("hic high-spread:", out)

# Detection stub
print("detection:", b.dispatch("t", "detection", Envelope("detect", {"image": "x.jpg"})))

# MAVLink sim shell
print("mavlink:", b.dispatch("t", "drone", Envelope("mavlink_connect", {"backend": "sim"})))
print("mav arm:", b.dispatch("t", "drone", Envelope("mavlink_command", {"action": "arm"})))
print("mav land:", b.dispatch("t", "drone", Envelope("mavlink_command", {"action": "land"})))
print("evidence entries:", len(b.evidence()))
