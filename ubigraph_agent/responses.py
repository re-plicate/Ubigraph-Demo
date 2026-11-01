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
            raise GatewayError('No model API key is configured. Set OPENAI_API_KEY in the local .env.local file.')
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
            raise GatewayError(f"The model gateway returned HTTP {exc.code}: {self.cfg.redact(message)[:500]}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GatewayError('Model gateway connection failed:'+self.cfg.redact(exc)[:240]) from None

    def models(self):
        with self._request("/models", timeout=20) as response:
            data = json.load(response)
        return [r["id"] for r in data.get("data", []) if r.get("id") and
                not any(s in r["id"].lower() for s in ("image", "auto-review", "embedding"))]

    def complete(self, inputs, tools, instructions, emit, cancelled, model=None):
        if self.cfg.wire_api() == "chat_completions":
            return self._chat_complete(inputs, tools, instructions, emit, cancelled, model)
        if cancelled():
            raise Cancelled('Task stopped')
        payload = {"model": model or self.cfg.model, "instructions": instructions,
                   "input": inputs, "tools": tools, "tool_choice": "auto",
                   "stream": True, "store": False, "max_output_tokens": self.cfg.max_output_tokens,
                   "reasoning": {"effort": self.cfg.reasoning_effort, "summary": "auto"}}
        final = None
        started = time.monotonic()
        with self._request("/responses", payload) as stream:
            for event in sse_messages(stream):
                if cancelled():
                    raise Cancelled('Task stopped')
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
                        emit("warning", {"message": 'Response limit reached. Generated explanation is preserved; ask a follow-up question.'})
                    else:
                        raise GatewayError('Model response incomplete:'+self.cfg.redact(reason or "incomplete"))
                elif kind in ("response.failed", "error"):
                    detail = event.get("error") or event.get("response", {}).get("error") or event.get("response", {}).get("incomplete_details") or kind
                    raise GatewayError('Model response incomplete:'+self.cfg.redact(detail)[:300])
        if final is None:
            raise GatewayError('The model stream ended before response.completed. Retry the request.')
        emit("model_completed", {"model": final.get("model", model or self.cfg.model),
                                 "seconds": round(time.monotonic()-started, 3),
                                 "usage": {k: final.get("usage", {}).get(k, 0)
                                           for k in ("input_tokens", "output_tokens", "total_tokens")}})
        return final

    def _chat_complete(self, inputs, tools, instructions, emit, cancelled, model):
        """Adapt streaming Chat Completions to the runtime's canonical tool records."""
        if cancelled():
            raise Cancelled("Task stopped")
        messages = [{"role": "system", "content": instructions}]
        for item in inputs:
            kind = item.get("type")
            if kind == "function_call":
                call = {"id": item["call_id"], "type": "function", "function": {
                    "name": item["name"], "arguments": item["arguments"]}}
                if messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                    messages[-1]["tool_calls"].append(call)
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            elif kind == "function_call_output":
                messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
            elif kind == "message" or item.get("role") in ("user", "assistant", "system"):
                content = item.get("content", "")
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content)
                role = item.get("role", "assistant")
                # Text and parallel function calls belong to one assistant turn.
                # A separate assistant message here would interrupt the required
                # tool-call -> tool-result sequence in Chat Completions.
                if role == "assistant" and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                    messages[-1]["content"] = (messages[-1].get("content") or "") + content
                else:
                    messages.append({"role": role, "content": content})
        functions = [{"type": "function", "function": {"name": tool["name"],
            "description": tool.get("description", ""), "parameters": tool["parameters"]}} for tool in tools]
        payload = {"model": model or self.cfg.model, "messages": messages, "stream": True,
                   "max_tokens": self.cfg.max_output_tokens, "stream_options": {"include_usage": True}}
        if "api.deepseek.com" in self.cfg.base_url:
            payload["thinking"] = {"type": "disabled"}
        if functions:
            payload.update(tools=functions, tool_choice="auto")
        started = time.monotonic()
        calls, text, usage, finish = {}, [], {}, None
        with self._request("/chat/completions", payload) as stream:
            for event in sse_messages(stream):
                if cancelled():
                    raise Cancelled("Task stopped")
                if event.get("error"):
                    raise GatewayError(self.cfg.redact(event["error"]))
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        text.append(delta["content"])
                        emit("text_delta", {"text": delta["content"]})
                    for fragment in delta.get("tool_calls", []):
                        call = calls.setdefault(fragment["index"], {"type": "function_call", "call_id": "", "name": "", "arguments": ""})
                        call["call_id"] += fragment.get("id", "")
                        function = fragment.get("function", {})
                        call["name"] += function.get("name", "")
                        call["arguments"] += function.get("arguments", "")
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        if finish not in ("stop", "tool_calls"):
            raise GatewayError("Model response did not finish: " + str(finish or "stream ended"))
        output = []
        for index in sorted(calls):
            call = calls[index]
            if not call["call_id"] or not call["name"]:
                raise GatewayError("The model returned an incomplete tool call")
            json.loads(call["arguments"])
            emit("tool_requested", {"name": call["name"], "call_id": call["call_id"]})
            output.append(call)
        if text:
            output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "".join(text)}]})
        normalized_usage = {"input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)}
        emit("model_completed", {"model": payload["model"], "seconds": round(time.monotonic()-started, 3), "usage": normalized_usage})
        return {"model": payload["model"], "output": output, "usage": normalized_usage}
