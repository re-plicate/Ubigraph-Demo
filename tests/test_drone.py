"""Execution contracts: reservations, receipts, cancellation, restart and Agent/HTTP wiring."""
import copy
import json
import math
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from ubigraph_agent.config import AgentConfig
from ubigraph_agent.runtime import AgentRuntime
from ubigraph_agent.server import make_handler
from ubigraph_drone import DroneService
from ubigraph_drone.agent_tools import execute, schemas
from ubigraph_drone.planner import astar
from ubigraph_drone.scene import HOMES, NODES, OBSTACLES, SEPARATION, SLOT


@pytest.fixture
def fleet(tmp_path):
    service = DroneService(tmp_path / "fleet", autostart=False)
    yield service
    service.close()


def assert_grounded_at_home(service, exclude=()):
    for key, d in service.drones.items():
        if key not in exclude:
            assert (d.x, d.y) == pytest.approx(HOMES[key], abs=1e-6)
            assert d.z == pytest.approx(0, abs=1e-6)


def test_three_drone_eight_target_mission_has_real_receipts_and_no_conflicts(fleet):
    plan = fleet.plan("eight", list(NODES))
    assert plan["feasible"] and len(plan["routes"]) == 3
    mission = fleet.start(plan["id"])
    assert mission["status"] == "running" and not mission["receipts"]
    # Check ALL discrete reservations and every motion segment before executing.
    paths = {k: r["path"] for k, r in plan["routes"].items()}
    for tick in range(max(p[-1]["tick"] for p in paths.values()) + 1):
        occupancy = [p[min(tick, len(p) - 1)] for p in paths.values()]
        assert len({(p["x"], p["y"]) for p in occupancy}) == len(occupancy)
        assert all((p["x"], p["y"]) not in OBSTACLES for p in occupancy)
        if tick:
            for a, pa in paths.items():
                for b, pb in paths.items():
                    if a >= b:
                        continue
                    a0, a1 = pa[min(tick-1,len(pa)-1)], pa[min(tick,len(pa)-1)]
                    b0, b1 = pb[min(tick-1,len(pb)-1)], pb[min(tick,len(pb)-1)]
                    assert not ((a0["x"],a0["y"])==(b1["x"],b1["y"]) and (b0["x"],b0["y"])==(a1["x"],a1["y"]))
    fleet.advance(plan["duration_seconds"] + 1)
    result = fleet.mission(mission["id"])
    assert result["status"] == "completed"
    assert set(result["completed_targets"]) == set(NODES)
    assert len(result["receipts"]) == 8
    assert all(r["measurement"] is None and r["observed_at"] is None for r in result["receipts"])
    assert result["min_separation_m"] >= SEPARATION
    assert result["distance_m"] > 0
    assert_grounded_at_home(fleet)


def test_spacetime_astar_avoids_reverse_edge_and_waits_until_dwell_is_safe():
    vertices = {(0,0,0),(1,0,1),(2,0,2),(1,0,3)}
    edges = {((0,0),(1,0),1),((1,0),(2,0),2)}
    path = astar((2,0),(0,0),set(),vertices=vertices,edges=edges)
    assert all(p not in vertices for p in path)
    for a,b in zip(path,path[1:]):
        assert ((b[0],b[1]),(a[0],a[1]),b[2]) not in edges
    held = astar((0,1),(1,1),set(),vertices={(1,1,2)},goal_hold=2)
    assert held[-1][2] >= 3


@pytest.mark.parametrize("yaw", [0, 90, 180, -135])
def test_position_feedback_world_body_conversion_and_manual_actions(fleet, yaw):
    fleet.drones["1"].yaw = yaw
    first = fleet.control("up", "1", "takeoff")
    fleet.advance(3)
    assert fleet.mission(first["id"])["status"] == "completed"
    assert fleet.drones["1"].z == pytest.approx(.6,abs=1e-6)
    navigation = fleet.control("go", "1", "navigate", "G1")
    fleet.advance(navigation["duration_seconds"] + 1)
    assert fleet.drones["1"].x == pytest.approx(5,abs=1e-6)
    assert fleet.drones["1"].y == pytest.approx(2,abs=1e-6)
    # Untargeted drones receive no movement and no flight energy cost.
    assert fleet.drones["2"].distance == fleet.drones["3"].distance == 0
    back = fleet.control("back", "1", "return")
    fleet.advance(back["duration_seconds"] + 1)
    assert_grounded_at_home(fleet)


def test_idempotency_stale_plans_busy_airspace_and_priority_boundary(fleet):
    p = fleet.plan("same", ["G1","G3"], priority_targets=["G3"])
    assert p["targets"] == ["G3","G1"]
    assert fleet.plan("same", ["G1","G3"], priority_targets=["G3"])["id"] == p["id"]
    with pytest.raises(ValueError,match="不同内容"):
        fleet.plan("same", ["G2"])
    m = fleet.start(p["id"])
    assert fleet.start(p["id"])["id"] == m["id"]
    with pytest.raises(ValueError,match="空域"):
        fleet.plan("busy", ["G2"])
    with pytest.raises(ValueError):
        fleet.reset()
    fleet.advance(p["duration_seconds"]+1)
    assert fleet.start(p["id"])["status"] == "completed"  # never dispatch twice
    p = fleet.plan("stale", ["G4"])
    fleet.configure_drone("1",battery=50)
    with pytest.raises(ValueError,match="过期"):
        fleet.start(p["id"])


def test_low_battery_exclusion_infeasible_plan_and_late_submission_check(fleet):
    fleet.configure_drone("1",battery=18)
    p = fleet.plan("battery",["G1","G3"])
    assert p["feasible"] and "1" not in p["routes"]
    for key in ("2","3"):
        fleet.configure_drone(key,battery=18)
    p = fleet.plan("none",["G4"])
    assert not p["feasible"] and p["rejected"]
    with pytest.raises(ValueError,match="不可行"):
        fleet.start(p["id"])
    assert not fleet.active


def test_cancel_during_collection_retains_completed_receipts_and_returns(fleet):
    p=fleet.plan("cancel",["G1","G3"],drone_ids=["1"])
    m=fleet.start(p["id"])
    first=p["routes"]["1"]["jobs"][0]
    fleet.advance(first["end_tick"]*SLOT+.5)
    assert fleet.mission(m["id"])["completed_targets"] == ["G1"]
    c=fleet.cancel(m["id"])
    assert c["status"] == "cancelling" and c["finished_sim_time"] is None
    fleet.advance(200)
    c=fleet.mission(m["id"])
    assert c["status"] == "cancelled"
    assert c["completed_targets"] == ["G1"] and len(c["receipts"])==1
    assert not any(e["type"]=="collected" and e["node"]=="G3" for e in c["events"])
    assert_grounded_at_home(fleet)


def test_cancel_at_last_landing_never_relaunches_already_grounded_drones(fleet):
    p=fleet.plan('late-cancel',['G1','G3','G5'],priority_targets=['G3'])
    m=fleet.start(p['id'])
    fleet.advance(p['duration_seconds']-.5)
    grounded={d.id:d.distance for d in fleet.drones.values() if d.z<.05}
    assert grounded
    result=fleet.cancel(m['id']);fleet.advance(20)
    result=fleet.mission(m['id'])
    assert result['status']=='cancelled'
    cutoff=next(e['id'] for e in result['events'] if e['type']=='cancel_requested')
    assert not any(e['type']=='phase' and 'taking_off' in e['message'] for e in result['events'] if e['id']>cutoff)
    assert all(fleet.drones[k].distance==pytest.approx(v) for k,v in grounded.items())
    assert_grounded_at_home(fleet)


def test_fault_in_active_drone_aborts_future_collection_and_recovers_other_drones(fleet):
    p=fleet.plan("fault",list(NODES));m=fleet.start(p["id"]);fleet.advance(6.5)
    fleet.configure_drone("1",fault=True)
    fleet.advance(200)
    result=fleet.mission(m["id"])
    assert result["status"]=="failed" and fleet.drones["1"].status=="fault"
    assert_grounded_at_home(fleet,exclude=["1"])
    assert not result["completed_targets"]
    assert any(e["type"]=="fault" for e in result["events"])


def test_restart_retains_receipts_marks_active_interrupted_and_invalidates_old_plan(tmp_path):
    d=DroneService(tmp_path,autostart=False)
    p=d.plan("restart",["G1","G3"],drone_ids=["1"]);m=d.start(p["id"])
    d.advance(15);d.close()
    recovered=DroneService(tmp_path,autostart=False)
    try:
        result=recovered.mission(m["id"])
        assert result["status"]=="interrupted"
        assert len(result["receipts"])==1
        assert not recovered.active
        assert recovered.start(p["id"])["status"]=="interrupted"
    finally:
        recovered.close()


@pytest.mark.parametrize("kwargs", [
    {"targets":[]},{"targets":["UNKNOWN"]},{"targets":["G1","BRUUNS"]},
    {"dwell_seconds":float('nan')},{"dwell_seconds":True},{"return_to_base":"true"},
    {"drone_ids":[]},{"drone_ids":["9"]},{"drone_ids":["1","1"]},
    {"priority_targets":False},{"priority_targets":["G8"]}])
def test_bad_external_requests_are_rejected_without_dispatch(fleet,kwargs):
    with pytest.raises(ValueError):
        fleet.plan(**({"request_id":"bad","targets":["G1"]}|kwargs))
    assert not fleet.active


def test_rejected_settings_are_atomic_and_cancel_resumes_clock(fleet):
    with pytest.raises(ValueError):
        fleet.settings(paused=True,speed=100)
    assert not fleet.paused
    plan=fleet.plan('atomic',['G1']);mission=fleet.start(plan['id'])
    before=fleet.drones['1'].battery
    with pytest.raises(ValueError):
        fleet.configure_drone('1',battery=50,fault=False)
    assert fleet.drones['1'].battery==before
    fleet.settings(paused=True)
    fleet.cancel(mission['id'])
    assert not fleet.paused


def test_dashboard_mutations_preserve_full_mission_fields(fleet):
    plan=fleet.plan('display',['G1']);fleet.start(plan['id'])
    for state in [fleet.settings(paused=True),fleet.configure_drone('3',battery=99)]:
        assert 'distance_m' in state['missions'][-1]
        assert 'path' in state['missions'][-1]['routes']['1']


class EngineStub:
    names=["BRUUNS","SALLING"]
    def context(self,value=None):
        return value or {"as_of":"2014-09-30 23:30:00","protocol":"fewshot","seed":7}


def test_agent_uses_actual_drone_tools_and_mission_outlives_chat(tmp_path,fleet):
    class Gateway:
        turn=0
        def complete(self,inputs,tools,instructions,emit,cancelled,model):
            self.turn+=1
            names={t['name'] for t in tools}
            assert 'plan_collection_mission' in names and 'predict_parking' in names
            if self.turn==1:
                name,args='plan_collection_mission',dict(request_id='agent',targets=['G1'],drone_ids=['1'],priority_targets=[],dwell_seconds=3,return_to_base=True)
            elif self.turn==2:
                result=json.loads(inputs[-1]['output'])['result']
                name,args='start_collection_mission',{'plan_id':result['id']}
            else:
                return {'output':[{'type':'message','content':[{'type':'output_text','text':'模拟任务已提交，尚未完成。'}]}]}
            return {'output':[{'type':'function_call','call_id':f'call_{self.turn}','name':name,'arguments':json.dumps(args)}]}
    rt=AgentRuntime(AgentConfig(runtime_dir=tmp_path/'agent'),EngineStub(),gateway=Gateway(),drone_service=fleet)
    session=rt.memory.create_session(rt.engine.context())
    task=rt.submit(session['id'],'执行G1模拟补采',session['context'])
    deadline=time.monotonic()+5
    while rt.memory.task(task)['status'] in ('queued','running') and time.monotonic()<deadline:
        time.sleep(.01)
    assert rt.memory.task(task)['status']=='completed'
    assert fleet.active and fleet.mission('latest')['status']=='running'
    assert rt.memory.history(session['id'])[-1]['content'].startswith('模拟任务已提交')
    fleet.advance(60)
    assert fleet.mission('latest')['status']=='completed'
    assert any(e['type']=='collected' for e in fleet.mission('latest')['events'])


def test_http_contract_rejects_cross_origin_and_runs_without_model_api(tmp_path,fleet):
    rt=AgentRuntime(AgentConfig(runtime_dir=tmp_path/'agent'),EngineStub(),drone_service=fleet)
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(rt))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def request(path,data=None,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        req=urllib.request.Request(base+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)
        with urllib.request.urlopen(req,timeout=3) as r:return json.load(r)
    try:
        assert request('/api/health')['drone_simulation_enabled']
        p=request('/api/drone/plan',{'request_id':'http','targets':['G1']})
        m=request('/api/drone/start',{'plan_id':p['id']})
        assert request('/api/drone/missions/'+m['id'])['status']=='running'
        with pytest.raises(urllib.error.HTTPError):
            request('/api/drone/reset',{},'https://unrelated.example')
        with urllib.request.urlopen(base+'/api/drone/events',timeout=3) as stream:
            packet=json.loads(stream.readline().decode().removeprefix('data: '))
            assert packet['active_mission_id']==m['id']
        for path in ('/','/drone.js','/drone.css'):
            with urllib.request.urlopen(base+path,timeout=3) as r: assert r.status==200
        request('/api/drone/cancel',{'mission_id':m['id']});fleet.advance(60)
        assert request('/api/drone/missions/'+m['id'])['status']=='cancelled'
    finally:
        server.shutdown();server.server_close();thread.join(2)


def test_strict_drone_tool_schema_rejects_extra_and_boolean_number(fleet):
    with pytest.raises(ValueError):execute(fleet,'get_fleet_status',{'extra':1})
    spec=next(s for s in schemas() if s['name']=='plan_collection_mission')
    assert spec['strict'] and not spec['parameters']['additionalProperties']
    args=dict(request_id='tool',targets=['G1'],drone_ids=['1'],priority_targets=[],dwell_seconds=True,return_to_base=True)
    with pytest.raises(ValueError):execute(fleet,'plan_collection_mission',args)
