import os
os.environ["UBIGRAPH_RUN"] = "models/parking"
from ubigraph_agents.api import app
print("FastAPI app:", app.title, app.version)
for r in app.routes:
    print(" ", sorted(r.methods), r.path)
