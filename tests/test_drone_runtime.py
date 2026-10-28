"""Regression cases reproduced during the independent drone code review."""
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ubigraph_agent.config import AgentConfig
from ubigraph_agent.runtime import AgentRuntime
from ubigraph_drone import DroneService
from ubigraph_drone.scene import HOMES, NODES, RESERVE, SEPARATION


@pytest.fixture
def fleet(tmp_path):
    d=DroneService(tmp_path/'fleet',autostart=False)
    yield d
    d.close()


def test_hover_drain_rechecks_obstacle_aware_return_before_dispatch(fleet):
    fleet.control('up','1','takeoff');fleet.advance(3)
    plan=fleet.plan('waited',['G3'],['1'],return_to_base=False)
    assert plan['feasible']
    fleet.advance((fleet.drones['1'].battery-27.7)/.035)
    assert fleet.drones['1'].battery==pytest.approx(27.7,abs=.002)
    with pytest.raises(ValueError,match='返航电量'):
        fleet.start(plan['id'])
    assert fleet.active is None


def test_unassigned_hovering_drone_has_wait_and_return_budget(fleet):
    fleet.control('up','2','takeoff');fleet.advance(3)
    fleet.configure_drone('2',battery=17)
    plan=fleet.plan('other-hover',['G3'],['1'])
    assert not plan['feasible']
    assert any('2号' in r['reason'] for r in plan['rejected'])


def test_restart_invalidates_unstarted_hover_plan_and_freezes_interrupted_time(tmp_path):
    fleet=DroneService(tmp_path,autostart=False)
    fleet.control('up','1','takeoff');fleet.advance(3)
    old=fleet.plan('not-started',['G1'],['1'])
    running=fleet.plan('running',['G3'],['1']);m=fleet.start(running['id']);fleet.advance(7)
    fleet.close()
    recovered=DroneService(tmp_path,autostart=False)
    try:
        with pytest.raises(ValueError,match='过期'):
            recovered.start(old['id'])
        record=recovered.mission(m['id'])
        assert record['status']=='interrupted' and record['actual_duration_seconds']==pytest.approx(7)
        finished=record['finished_sim_time']
        recovered.advance(50)
        assert recovered.mission(m['id'])['finished_sim_time']==finished
        assert recovered.mission(m['id'])['actual_duration_seconds']==pytest.approx(7)
    finally:
        recovered.close()


def test_multiple_faults_do_not_move_earlier_faulty_drone_back_home(fleet):
    plan=fleet.plan('two-faults',list(NODES));m=fleet.start(plan['id']);fleet.advance(7.5)
    before={key:d.distance for key,d in fleet.drones.items()}
    fleet.configure_drone('1',fault=True);fleet.configure_drone('3',fault=True)
    fleet.advance(200)
    result=fleet.mission(m['id'])
    assert result['status']=='failed' and result['fault_drone_ids']==['1','3']
    for key in ('1','3'):
        d=fleet.drones[key]
        assert d.status=='fault' and d.z==0
        assert d.distance-before[key]<=1.01  # only the unfinished current grid edge
    assert (fleet.drones['2'].x,fleet.drones['2'].y)==pytest.approx(HOMES['2'])
    assert not result['receipts']


def test_fault_during_cancel_return_stops_the_drone_and_keeps_original_jobs(fleet):
    p=fleet.plan('return-fault',['G1','G3'],['1']);m=fleet.start(p['id'])
    fleet.advance(16.5);fleet.cancel(m['id']);fleet.advance(2)
    assert fleet.mission(m['id'])['status']=='returning'
    before=fleet.drones['1'].distance
    fleet.configure_drone('1',fault=True);fleet.advance(50)
    result=fleet.mission(m['id'])
    assert result['status']=='failed' and result['completed_targets']==['G1']
    assert result['original_routes']['1']['targets']==['G1','G3']
    assert fleet.drones['1'].distance-before<=1.01
    assert fleet.drones['1'].status=='fault'


def test_only_one_service_can_own_a_runtime_directory(fleet):
    before=fleet.snapshot()
    with pytest.raises(ValueError,match='另一实例'):
        DroneService(fleet.directory,autostart=False)
    assert fleet.snapshot()['revision']==before['revision']
    assert fleet.snapshot()['fleet']==before['fleet']


def test_concurrent_submission_starts_exactly_one_mission(fleet):
    p=fleet.plan('concurrent',['G1','G3'])
    barrier=threading.Barrier(6)
    def submit(_):
        barrier.wait(timeout=3)
        return fleet.start(p['id'])['id']
    with ThreadPoolExecutor(max_workers=6) as pool:
        ids=list(pool.map(submit,range(6)))
    assert len(set(ids))==1 and len(fleet.missions)==1
    assert sum(e['type']=='started' for e in fleet.events)==1
    fleet.advance(p['duration_seconds']+.1)
    assert fleet.mission(ids[0])['status']=='completed'


def test_second_agent_runtime_cannot_interrupt_first_runtime_chat(tmp_path):
    class Engine:
        names=['BRUUNS']
        def context(self,value=None): return value or {'as_of':'2014-09-30 23:30:00','protocol':'fewshot','seed':7}
    cfg=AgentConfig(runtime_dir=tmp_path)
    runtime=AgentRuntime(cfg,Engine())
    try:
        session=runtime.memory.create_session(runtime.engine.context())
        job=runtime.memory.create_task(session['id'],session['context'])
        runtime.memory.task_status(job,'running')
        with pytest.raises(ValueError,match='另一实例'):
            AgentRuntime(cfg,Engine())
        assert runtime.memory.task(job)['status']=='running'
    finally:
        runtime.drones.close()


@pytest.mark.parametrize('phase',['collection','navigate'])
def test_failed_motion_never_produces_fake_arrival_or_completion(fleet,monkeypatch,phase):
    if phase=='navigate':
        fleet.control('up','1','takeoff');fleet.advance(3)
        m=fleet.control('nav','1','navigate','G1')
    else:
        p=fleet.plan('no-motion',['G1'],['1']);m=fleet.start(p['id'])
    monkeypatch.setattr(fleet.drones['1'],'track',lambda *a: None)
    fleet.advance(100)
    result=fleet.mission(m['id'])
    assert result['status']=='failed' and not result['receipts']
    assert not any(e['type'] in ('arrived','collected','completed') for e in result['events'])


def test_background_failure_is_visible_and_worker_survives(tmp_path,monkeypatch):
    d=DroneService(tmp_path,autostart=True)
    entered=threading.Event()
    try:
        d.settings(paused=True)
        p=d.plan('worker-failure',['G1']);m=d.start(p['id'])
        original=d._step
        def broken(dt):
            entered.set()
            raise RuntimeError('injected controller failure')
        monkeypatch.setattr(d,'_step',broken)
        d.settings(paused=False)
        assert entered.wait(3)
        with d.lock:
            assert d.last_error and d.paused and d.thread.is_alive()
            assert d.mission(m['id'])['status']=='failed'
        with pytest.raises(ValueError,match='重置'):
            d.plan('must-reset',['G2'])
        monkeypatch.setattr(d,'_step',original)
        d.reset()
        assert d.last_error is None
    finally:
        d.close()


def test_runtime_separation_failure_marks_jobs_and_stops_unassigned_hoverer(fleet):
    p=fleet.plan('runtime-conflict',['G1'],['1']);m=fleet.start(p['id'])
    # Simulate an unexpected parked vehicle drifting into the takeoff area.
    other=fleet.drones['3']
    other.x,other.y,other.z,other.status=1.,2.,.6,'hovering'
    fleet.advance(.05)
    result=fleet.mission(m['id'])
    assert result['status']=='failed' and not result['receipts']
    assert result['routes']['1']['jobs'][0]['status']=='failed'
    assert all(d.z==0 and d.status=='idle' for d in fleet.drones.values())


@pytest.mark.parametrize('bad',[['1',[]],[{}]])
def test_nonstring_drone_ids_are_controlled_validation_errors(fleet,bad):
    with pytest.raises(ValueError):
        fleet.plan('bad',['G1'],bad)
    assert fleet.active is None


@pytest.mark.parametrize('seed',range(12))
def test_varied_multi_target_paths_keep_separation_and_complete(fleet,seed):
    rng=random.Random(seed)
    targets=rng.sample(list(NODES),rng.randint(3,8))
    ids=[['1','2','3'],['1','3'],['2']][seed%3]
    home=seed%2==0
    p=fleet.plan('scenario-'+str(seed),targets,ids,[targets[0]],(seed%3)*3,home)
    assert p['feasible'],p['rejected']
    m=fleet.start(p['id']);fleet.advance(p['duration_seconds']+.1)
    result=fleet.mission(m['id'])
    assert result['status']=='completed',result['reason']
    assert set(result['completed_targets'])==set(targets)
    assert all(d.battery>=RESERVE for d in fleet.drones.values())
    assert result['min_separation_m'] is None or result['min_separation_m']>=SEPARATION-1e-6
    if home:
        for key in p['routes']:
            d=fleet.drones[key]
            assert math.dist((d.x,d.y),HOMES[key])<1e-6 and d.z<1e-6
