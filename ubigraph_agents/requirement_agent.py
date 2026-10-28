"""Requirement Agent: decomposes user intent into skill calls.

Uses an LLM (OpenAI-compatible API) to parse natural language.
If no API key is set, falls back to rule-based keyword matching.

Set your key via environment variable:
    CITYCLAW_LLM_KEY=sk-...
    CITYCLAW_LLM_BASE=https://api.openai.com/v1  (or any OpenAI-compatible endpoint)
    CITYCLAW_LLM_MODEL=gpt-4o-mini
"""
from __future__ import annotations

import json
import os
import urllib.request

from .base import Agent, Envelope

# Available skills the LLM can choose from
SKILLS_CATALOG = """
Available agents and actions:
- temporal.predict: predict parking availability for a garage
- temporal.names: list all garage names
- spatial.neighbors: find most similar garages
- spatial.similarity: get full similarity matrix
- transfer.transfer_predict: predict for a new city/cross-region
- transfer.transfer_gain: compute transfer improvement
- drone.plan: plan a drone mission
- drone.state: get current drone fleet state
- detection.detect: run car detection on an image
- hic.evaluate: evaluate prediction uncertainty and decide if human review needed
"""

SYSTEM_PROMPT = f"""You are the Requirement Agent in the CityClaw parking prediction system.
Given a user's natural language query, decide which agents to call and in what order.

{SKILLS_CATALOG}

Return ONLY a JSON list of skill calls, e.g.:
[{{"agent": "spatial", "action": "neighbors"}}, {{"agent": "temporal", "action": "predict"}}]

If the query mentions a garage name (BRUUNS, KALKVAERKSVEJ, MAGASIN, SALLING, SCANDCENTER, SKOLEBAKKEN), include it as payload.garage.
If the query mentions a city (SFpark, London), set payload.target_city.
Return ONLY the JSON array, no explanation.
"""


class RequirementAgent(Agent):
    name = "requirement"

    def _register(self) -> None:
        self.handlers = {
            "parse": self._parse,
        }

    def _call_llm(self, user_query: str) -> list[dict] | None:
        """Call OpenAI-compatible LLM. Returns parsed skills or None on failure."""
        api_key = os.environ.get("CITYCLAW_LLM_KEY")
        base_url = os.environ.get("CITYCLAW_LLM_BASE", "https://api.openai.com/v1").rstrip("/")
        model = os.environ.get("CITYCLAW_LLM_MODEL", "gpt-4o-mini")

        if not api_key:
            return None

        try:
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                "temperature": 0.1,
            }).encode()
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"].strip()
                # Strip markdown code fences if present
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                skills = json.loads(content)
                return skills if isinstance(skills, list) else None
        except Exception as e:
            print(f"[RequirementAgent] LLM call failed: {e}")
            return None

    def _fallback_rules(self, text: str) -> list[dict]:
        """Rule-based fallback when no LLM is available."""
        text_lower = text.lower()
        skills = []
        if any(k in text_lower for k in ["预测", "prediction", "还有多少", "车位", "available", "停"]):
            skills.append({"agent": "temporal", "action": "predict"})
        if any(k in text_lower for k in ["相似", "neighbor", "哪里像", "spatial", "像哪个"]):
            skills.append({"agent": "spatial", "action": "neighbors"})
        if any(k in text_lower for k in ["掉了", "坏了", "掉线", "fail", "sensor", "故障"]):
            skills.append({"agent": "drone", "action": "plan"})
            skills.append({"agent": "detection", "action": "detect"})
            skills.append({"agent": "transfer", "action": "transfer_predict"})
        if any(k in text_lower for k in ["sfpark", "london", "跨城", "transfer", "迁移", "新城市"]):
            skills.append({"agent": "transfer", "action": "transfer_predict"})
        if not skills:
            skills.append({"agent": "temporal", "action": "predict"})
        return skills

    def _parse(self, env: Envelope) -> dict:
        query = env.payload.get("query", "")
        use_llm = bool(os.environ.get("CITYCLAW_LLM_KEY"))

        skills = None
        if use_llm:
            skills = self._call_llm(query)

        if skills is None:
            skills = self._fallback_rules(query)
            source = "rule-based (set CITYCLAW_LLM_KEY to enable LLM)"
        else:
            source = f"LLM ({os.environ.get('CITYCLAW_LLM_MODEL', 'gpt-4o-mini')})"

        return {
            "original_query": query,
            "decomposed_skills": skills,
            "n_skills": len(skills),
            "parser": source,
        }
