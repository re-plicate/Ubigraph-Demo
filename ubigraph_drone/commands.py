"""Inspectable command manifests derived only from a registered simulation plan."""
from .scene import NODES, HEIGHT, HOMES, SLOT


def manifest(plan):
    commands=[]
    for drone_id, route in plan['routes'].items():
        sequence=0
        def append(action, **fields):
            nonlocal sequence
            sequence+=1
            commands.append({'drone_id': drone_id, 'sequence': sequence, 'action': action, **fields})
        mode=plan.get('mode', 'collection')
        if mode != 'collection':
            height=route['path'][-1]['z']
            if mode == 'takeoff': append('TAKEOFF', height_m=height)
            elif mode == 'navigate':
                target=route['targets'][0];node=NODES[target]
                append('NAVIGATE', target=target, position_m=[node['x'],node['y'],height])
            elif mode == 'hover': append('HOLD_POSITION', height_m=height, duration_seconds=route['duration_seconds'])
            elif mode == 'return':
                append('RETURN_TO_BASE', position_m=[*HOMES[drone_id],route['path'][0]['z']]);append('LAND', height_m=0)
            elif mode == 'land': append('LAND', height_m=0)
            continue
        append('TAKEOFF', height_m=HEIGHT)
        for target in route['targets']:
            node=NODES[target]
            append('NAVIGATE', target=target, position_m=[node['x'],node['y'],HEIGHT])
            job=next(j for j in route['jobs'] if j['node']==target)
            append('COLLECT', target=target, duration_seconds=(job['end_tick']-job['arrival_tick'])*SLOT, requested_duration_seconds=plan['dwell_seconds'], measurement=None)
        if plan['return_to_base']:
            append('RETURN_TO_BASE', position_m=[*HOMES[drone_id],HEIGHT])
            append('LAND', height_m=0)
        else:
            append('HOLD_POSITION', height_m=HEIGHT)
    return {'format': 'ubigraph.flight.v1', 'backend': 'simulation', 'coordinate_frame': 'synthetic-grid-metres', 'plan_id':plan['id'], 'feasible':plan['feasible'], 'commands':commands, 'note':'声明式动作清单；执行器按已验证的时空航线运行。不是MAVLink数据包，不连接真实飞行器。'}
