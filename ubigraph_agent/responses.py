"""Small dependency-free Responses transport with proper function-call continuation."""
import json
import time
import urllib.error
import urllib.request


class GatewayError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


def sse_messages(lines):
    """SSE data may span multiple lines; comments and [DONE] are not JSON events."""
    data = []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                if payload != "[DONE]":
                    yield json.loads(payload)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data and "\n".join(data) != "[DONE]":
        yield json.loads("\n".join(data))


class ResponsesGateway:
    def __init__(self, config):
        self.cfg = config

    def _request(self, path, data=None, timeout=None):
        if not self.cfg.api_key:
            raise GatewayError("尚未配置模型密钥，请在本机 .env.local 设置 OPENAI_API_KEY。")
        headers = {"Authorization": "Bearer "+self.cfg.api_key,
                   "Content-Type": "application/json", "User-Agent": "UbiGraph-Agent/0.5"}
        req = urllib.request.Request(self.cfg.base_url+path,
                    data=None if data is None else json.dumps(data, ensure_ascii=False).encode(), headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.cfg.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(body).get("error", {}).get("message", body)
            except (ValueError, AttributeError):
                message = body
            raise GatewayError(f"模型网关返回 HTTP {exc.code}：{self.cfg.redact(message)[:500]}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GatewayError("模型网关连接失败："+self.cfg.redact(exc)[:240]) from None

    def models(self):
        with self._request("/models", timeout=20) as response:
            data = json.load(response)
        return [r["id"] for r in data.get("data", []) if r.get("id") and
                not any(s in r["id"].lower() for s in ("image", "auto-review", "embedding"))]

    def complete(self, inputs, tools, instructions, emit, cancelled, model=None):
        if cancelled():
            raise Cancelled("任务已停止")
        payload = {"model": model or self.cfg.model, "instructions": instructions,
                   "input": inputs, "tools": tools, "tool_choice": "auto",
                   "stream": True, "store": False, "max_output_tokens": self.cfg.max_output_tokens,
                   "reasoning": {"effort": self.cfg.reasoning_effort, "summary": "auto"}}
        final = None
        started = time.monotonic()
        with self._request("/responses", payload) as stream:
            for event in sse_messages(stream):
                if cancelled():
                    raise Cancelled("任务已停止")
                kind = event.get("type", "")
                if kind == "response.output_text.delta":
                    emit("text_delta", {"text": event.get("delta", "")})
                elif kind == "response.reasoning_summary_text.delta":
                    # Only provider-supplied summaries; never render private reasoning text.
                    emit("reasoning_summary", {"text": event.get("delta", "")})
                elif kind == "response.output_item.added" and event.get("item", {}).get("type") == "function_call":
                    item = event["item"]
                    emit("tool_requested", {"name": item.get("name"), "call_id": item.get("call_id")})
                elif kind == "response.completed":
                    final = event["response"]
                elif kind == "response.incomplete":
                    partial = event.get("response", {})
                    reason = (partial.get("incomplete_details") or {}).get("reason")
                    output = partial.get("output", [])
                    has_text = any(c.get("text") for o in output if o.get("type") == "message"
                                   for c in o.get("content", []) if c.get("type") == "output_text")
                    if reason == "max_output_tokens" and has_text and not any(o.get("type") == "function_call" for o in output):
                        final = partial
                        emit("warning", {"message": "答复达到单次长度上限，已保留已生成说明；可以继续追问某一步。"})
                    else:
                        raise GatewayError("模型未完成本轮响应："+self.cfg.redact(reason or "incomplete"))
                elif kind in ("response.failed", "error"):
                    detail = event.get("error") or event.get("response", {}).get("error") or event.get("response", {}).get("incomplete_details") or kind
                    raise GatewayError("模型未完成本轮响应："+self.cfg.redact(detail)[:300])
        if final is None:
            raise GatewayError("模型流提前结束，未收到 response.completed；可重新发送本次问题。")
        emit("model_completed", {"model": final.get("model", model or self.cfg.model),
                                 "seconds": round(time.monotonic()-started, 3),
                                 "usage": {k: final.get("usage", {}).get(k, 0)
                                           for k in ("input_tokens", "output_tokens", "total_tokens")}})
        return final
