"""Bounded Responses orchestration, live execution events, cancellation and memory."""
import json
import threading
import time

from .hub import AgentHub
from .memory import MemoryStore
from .responses import ResponsesGateway, GatewayError, Cancelled
from .tools import Skills, TOOL_LABELS, instructions
from ubigraph_drone import DroneService


class Job:
    def __init__(self, identity, memory):
        self.id, self.memory = identity, memory
        self.cancel = threading.Event()
        self.condition = threading.Condition()
        self.state_lock = threading.RLock()
        self.finished = False

    def emit(self, kind, data):
        with self.state_lock:
            if self.finished:
                return None
            value = self.memory.event(self.id, kind, data)
        with self.condition:
            self.condition.notify_all()
        return value

    def finish(self, status, kind, data, error="", assistant=None):
        with self.state_lock:
            if self.finished:
                return
            self.memory.finish(self.id, status, kind, data, error, assistant)
            self.finished = True
        with self.condition:
            self.condition.notify_all()


class AgentRuntime:
    def __init__(self, config, engine, gateway=None, memory=None, drone_service=None):
        self.cfg, self.engine = config, engine
        self.gateway = gateway or ResponsesGateway(config)
        self.drones = drone_service or DroneService(config.runtime_dir/"drone", autostart=False)
        try:
            self.memory = memory or MemoryStore(config.runtime_dir/"memory.sqlite3")
        except BaseException:
            self.drones.close()
            raise
        self.hub = AgentHub(engine, self.drones, config.detection_weights)
        self.jobs, self.lock = {}, threading.RLock()
        self.models = [config.model]
        self.provider_status = "configured" if config.api_key else "unconfigured"

    def discover(self):
        if not self.cfg.api_key:
            self.provider_status = "unconfigured"
            return
        try:
            self.models = self.gateway.models() or [self.cfg.model]
            self.provider_status = "catalog_connected"
        except GatewayError:
            self.provider_status = "unreachable"

    def submit(self, session_id, message, context, model=None, agent="orchestrator", execution_policy="execute"):
        if not isinstance(message, str) or not message.strip() or len(message) > 6000:
            raise ValueError('Enter a question containing 1 to 6000 characters')
        context = self.engine.context(context)
        session = self.memory.session(session_id)
        if session["context"] != context:
            raise ValueError('Replay time, protocol or model revision changed. Start a new conversation to keep evidence consistent.')
        if model is not None and model not in self.models:
            raise ValueError('Select a text model available from the current provider')
        if agent not in {"orchestrator", *self.hub.bus.agents}:
            raise ValueError('Unknown agent')
        if execution_policy not in ("review", "execute"):
            raise ValueError('Invalid simulation execution policy')
        identity = self.memory.create_task(session_id, context)
        self.memory.message(session_id, "user", message.strip(), identity)
        job = Job(identity, self.memory)
        with self.lock:
            self.jobs[identity] = job
        job.emit("queued", {"message": 'Task received', "context": context})
        threading.Thread(target=self._run, args=(job, session_id, context, model or self.cfg.model, agent, execution_policy), daemon=True).start()
        return identity

    def _run(self, job, session_id, context, model, agent="orchestrator", execution_policy="execute"):
        with job.state_lock:
            if job.finished:
                return
            self.memory.task_status(job.id, "running")
        started, texts, calls, usage = time.monotonic(), [], 0, 0
        try:
            previous = self.memory.artifacts(session_id)
            skills = Skills(self.engine, context, previous, self.drones, self.hub, execution_policy)
            history = self.memory.history(session_id)[-12:]
            inputs = [{"role": r["role"], "content": r["content"]} for r in history]
            prompt = instructions(context, self.engine.names, previous, drones_enabled=True, input_spec=skills.input_spec)
            prompt += f"\nSelected entry agent: {agent}. Prioritize this role and call other specialist tools when needed. Agent selection does not change the data cutoff."
            prompt += '\nSimulation execution policy: ' + ("Review first. Plan only; takeoff, navigation and execution require the user's command workspace action or an explicit request with simulation execution enabled." if execution_policy == "review" else "Simulation execution allowed. Execute only the user's current request after checking fleet state and plan feasibility.")
            job.emit("agent_selected", {"agent": agent, "execution_policy": execution_policy})
            for round_no in range(1, self.cfg.max_rounds+1):
                if job.cancel.is_set():
                    raise Cancelled('Task stopped')
                job.emit("phase", {"role": 'Requirements and orchestration agent', "message": 'Understanding request and selecting tools' if round_no == 1 else 'Analyzing tool results', "round": round_no, "model": model})
                response = self.gateway.complete(inputs, skills.definitions, prompt, job.emit, job.cancel.is_set, model)
                self.provider_status = "connected"
                usage += response.get("usage", {}).get("total_tokens", 0)
                output = response.get("output", [])
                inputs.extend(output)  # includes reasoning items required for stateless continuation
                functions = [o for o in output if o.get("type") == "function_call"]
                current_text = "".join(c.get("text", "") for o in output if o.get("type") == "message"
                                       for c in o.get("content", []) if c.get("type") in ("output_text", "refusal"))
                if current_text:
                    if response.get("status") == "incomplete":
                        current_text += '\n\n(The response reached its length limit. The generated portion was saved; ask a follow-up about any step.)'
                    texts.append(current_text)
                if not functions:
                    if job.cancel.is_set():
                        raise Cancelled('Task stopped')
                    answer = "\n\n".join(texts).strip()
                    if not answer:
                        raise GatewayError('Model returned no displayable response. Try again.')
                    job.finish("completed", "done", {"answer": answer, "tool_calls": calls, "model": model,
                                      "seconds": round(time.monotonic()-started, 2), "total_tokens": usage,
                                      "answer_truncated": response.get("status") == "incomplete"}, assistant=answer)
                    return
                for call in functions:
                    if job.cancel.is_set():
                        raise Cancelled('Task stopped')
                    calls += 1
                    if calls > self.cfg.max_tool_calls:
                        raise GatewayError('Tool-call limit reached. Results are preserved; narrow the request and try again.')
                    name, call_id = call.get("name", ""), call.get("call_id", "")
                    role, title = TOOL_LABELS.get(name, ('Executor', name))
                    try:
                        args = json.loads(call.get("arguments", "{}"))
                    except (ValueError, TypeError):
                        args = None
                    job.emit("tool_started", {"name": name, "call_id": call_id, "role": role, "title": title, "arguments": args})
                    t0 = time.monotonic()
                    try:
                        result = skills.execute(name, args)
                        job.emit("artifact", {**result, "tool": name, "task_id": job.id})
                        job.emit("tool_completed", {"name": name, "call_id": call_id, "role": role,
                                                   "title": title, "seconds": round(time.monotonic()-t0, 3)})
                        tool_output = {"ok": True, "result": result}
                    except (ValueError, FileNotFoundError) as exc:
                        error = self.cfg.redact(exc)
                        job.emit("tool_error", {"name": name, "call_id": call_id, "error": error})
                        tool_output = {"ok": False, "error": error}
                    inputs.append({"type": "function_call_output", "call_id": call_id,
                                   "output": json.dumps(tool_output, ensure_ascii=False, allow_nan=False)})
            raise GatewayError('Maximum model rounds reached. Tool results are preserved; ask a more specific follow-up question.')
        except Cancelled:
            job.finish("cancelled", "cancelled", {"message": 'Task stopped. Computed results are preserved.'}, 'User stopped task')
        except Exception as exc:
            error = self.cfg.redact(exc)[:700]
            job.finish("failed", "error", {"message": error}, error)
        finally:
            with self.lock:
                self.jobs.pop(job.id, None)

    def cancel(self, identity):
        self.memory.task(identity)
        with self.lock:
            job = self.jobs.get(identity)
        if job is not None:
            job.cancel.set()
            # Stop the UI immediately. A blocking provider read may finish later;
            # no subsequent events, answers or tool calls are accepted for this job.
            job.finish("cancelled", "cancelled", {"message": 'Task stopped. Computed results are preserved.'}, 'User stopped task')

    def session_view(self, identity):
        return {"session": self.memory.session(identity), "messages": self.memory.history(identity),
                "tasks": self.memory.session_tasks(identity), "artifacts": self.memory.artifacts(identity)}
