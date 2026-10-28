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
            raise ValueError("请输入1至6000字的问题")
        context = self.engine.context(context)
        session = self.memory.session(session_id)
        if session["context"] != context:
            raise ValueError("回放时刻、模型协议或模型版本已改变，请开始一个新会话，避免混用数据上下文。")
        if model is not None and model not in self.models:
            raise ValueError("请选择当前中转站可用的文本模型")
        if agent not in {"orchestrator", *self.hub.bus.agents}:
            raise ValueError("未知 Agent")
        if execution_policy not in ("review", "execute"):
            raise ValueError("无效的仿真执行策略")
        identity = self.memory.create_task(session_id, context)
        self.memory.message(session_id, "user", message.strip(), identity)
        job = Job(identity, self.memory)
        with self.lock:
            self.jobs[identity] = job
        job.emit("queued", {"message": "任务已接收", "context": context})
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
            prompt += f"\n当前用户选择的入口 Agent：{agent}。优先处理其职责内的问题，需要协作时可调用其他专业工具；入口选择不改变数据时刻。"
            prompt += "\n当前仿真执行策略：" + ("先审阅。只能规划；起飞、导航和提交执行由用户在指令工作区操作，或切换执行权限后再请求。" if execution_policy == "review" else "允许仿真执行。只执行用户当前请求的动作，仍需先检查机队与可行计划。")
            job.emit("agent_selected", {"agent": agent, "execution_policy": execution_policy})
            for round_no in range(1, self.cfg.max_rounds+1):
                if job.cancel.is_set():
                    raise Cancelled("任务已停止")
                job.emit("phase", {"role": "需求与编排 Agent", "message": "正在理解问题并选择技能" if round_no == 1 else "正在根据工具结果继续分析", "round": round_no, "model": model})
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
                        current_text += "\n\n（本次说明达到长度上限，已保留生成部分。可以继续追问其中某一步。）"
                    texts.append(current_text)
                if not functions:
                    if job.cancel.is_set():
                        raise Cancelled("任务已停止")
                    answer = "\n\n".join(texts).strip()
                    if not answer:
                        raise GatewayError("模型没有返回可显示的答复，请重新发送。")
                    job.finish("completed", "done", {"answer": answer, "tool_calls": calls, "model": model,
                                      "seconds": round(time.monotonic()-started, 2), "total_tokens": usage,
                                      "answer_truncated": response.get("status") == "incomplete"}, assistant=answer)
                    return
                for call in functions:
                    if job.cancel.is_set():
                        raise Cancelled("任务已停止")
                    calls += 1
                    if calls > self.cfg.max_tool_calls:
                        raise GatewayError("已达到本次工具调用上限，任务停止；可以缩小问题后重新发送。")
                    name, call_id = call.get("name", ""), call.get("call_id", "")
                    role, title = TOOL_LABELS.get(name, ("执行器", name))
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
            raise GatewayError("模型分析达到最大轮次，已有工具结果保留在页面中，可继续提出更具体的问题。")
        except Cancelled:
            job.finish("cancelled", "cancelled", {"message": "任务已停止，已经取得的计算结果保留。"}, "用户停止任务")
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
            job.finish("cancelled", "cancelled", {"message": "任务已停止，已经取得的计算结果保留。"}, "用户停止任务")

    def session_view(self, identity):
        return {"session": self.memory.session(identity), "messages": self.memory.history(identity),
                "tasks": self.memory.session_tasks(identity), "artifacts": self.memory.artifacts(identity)}
