"""One in-process agent registry shared by chat, operations and the fleet UI."""
import json
import threading
from pathlib import Path

import pandas as pd

from ubigraph_agents.base import Agent, Envelope
from ubigraph_agents.bus import AgentBus
from ubigraph_agents.detection_agent import DetectionAgent
from ubigraph_agents.hic_agent import HumanInLoopAgent
from ubigraph_agents.transfer_agent import TransferAgent
from ubigraph_agents.drone_agent import DroneFleetAgent


class EngineAgent(Agent):
    def __init__(self, bus, name, handlers):
        self.name, self._handlers = name, handlers
        super().__init__(bus)

    def _register(self):
        self.handlers = self._handlers


class AgentHub:
    def __init__(self, engine, drones, detection_weights=None):
        self.engine, self.drones = engine, drones
        self.bus, self.lock = AgentBus(), threading.RLock()
        EngineAgent(self.bus, 'data', {'inspect': lambda e: engine.inspect(e.payload['targets'], e.payload['context'])})
        EngineAgent(self.bus, 'temporal', {'predict': lambda e: engine.predict(e.payload['targets'], e.payload['context'], e.payload.get('horizon', 30), e.payload.get('include_other', True))})
        EngineAgent(self.bus, 'spatial', {'similarity': lambda e: engine.spatial(e.payload['target'], e.payload['context'], e.payload.get('top_k', 5))})
        HumanInLoopAgent(self.bus)
        DetectionAgent(self.bus, weights=detection_weights or None)
        TransferAgent(self.bus)
        DroneFleetAgent(self.bus, drone_service=drones)

    def call(self, name, action, payload):
        with self.lock:
            value = self.bus.dispatch('conversation', name, Envelope(action, payload))
        if 'error' in value:
            raise ValueError(value['error'])
        return value

    def directory(self):
        detection = self.bus.agents['detection'].handle(Envelope('status', {}))
        rows = [
            ('orchestrator', 'Orchestrator', 'Route natural-language requests to the right Agent', 'OR', 'coordination', 'ready'),
            ('data', 'Data inspection', 'Inspect historical observations, gaps and input quality', 'DA', 'inspection', 'ready'),
            ('temporal', 'Temporal forecasting', 'Compute next-step availability with the trained V3 model', 'TP', 'prediction', 'ready'),
            ('spatial', 'Spatial analysis', 'Inspect spatial relationships used by the checkpoint', 'SP', 'analysis', 'ready'),
            ('hic', 'Decision review', 'Recommend collection from observation quality', 'HI', 'review', 'ready'),
            ('drone', 'Fleet dispatch', 'Generate structured commands and run the three-drone simulation', 'FL', 'operations', 'ready' if self.drones else 'unavailable'),
            ('detection', 'Vision detection', 'Use local COCO weights to identify vehicles in images', 'CV', 'perception', 'ready' if detection.get('available') else 'unconfigured'),
            ('transfer', 'Transfer research', 'Traceable nearest-profile borrowing research baseline', 'TR', 'research', 'experimental'),
        ]
        return {'kind': 'agent_directory', 'agents': [dict(id=i, name=n, description=d, initials=a, category=c, status=s) for i,n,d,a,c,s in rows], 'backend': 'shared-in-process', 'hardware_connected': False}

    def assess(self, targets, context):
        history = self.call('data', 'inspect', {'targets': targets, 'context': context})
        decisions = []
        for row in history['items']:
            last = row['history'][-1]
            reasons = []
            if not row['valid']: reasons.append('Historical window does not meet model input requirements')
            if not row['latest_observed']: reasons.append('No direct observation at the cutoff time')
            if last.get('age') is None or last['age'] > 1: reasons.append('Latest observation is stale')
            risk = 1 - row['observed_points'] / max(1, row['history_points'])
            gate = self.call('hic', 'evaluate', {'prediction': {'current_input_status': {'target': 'missing' if reasons else 'available'}}, 'cost': .2})
            decisions.append({'garage': row['garage'], 'needs_collection': bool(reasons), 'reasons': reasons or ['Current history meets input requirements. No automatic collection is needed.'], 'observed_points': row['observed_points'], 'history_points': row['history_points'], 'missing_fraction': round(risk, 4), 'review': gate})
        return {'kind': 'collection_assessment', 'as_of': context['as_of'], 'items': decisions, 'recommended_targets': [d['garage'] for d in decisions if d['needs_collection']], 'backend': 'simulation', 'note': 'Rule-based review uses replay observation quality, not calibrated confidence. Recommendations are simulation only and do not change 2014 observations.'}

    def transfer(self, sources, target, context):
        if target in sources or not sources:
            raise ValueError('Source garages must be nonempty and exclude the target')
        def table(names):
            chunks=[]
            for name in names:
                frame=self.engine.history(name, context).copy()
                frame['garagecode']=name
                frame=frame.dropna(subset=['available','capacity'])
                chunks.append(frame)
            return pd.concat(chunks, ignore_index=True)
        result = self.call('transfer', 'transfer_predict', {'source': table(sources), 'target': table([target])})
        return {'kind': 'transfer_estimate', **result, 'as_of': context['as_of'], 'experimental': True, 'note': 'Only inputs at or before the cutoff are used outside training. Nearest-profile research baseline, not the V3 primary model or validated cross-city transfer.'}
