"""Everything that talks to a language model: one adapter, plus the two jobs it does (parse the request, explain the result).
Gemini and OpenAI both speak the OpenAI chat protocol, so one client covers both. The LLM never makes the decision."""
from __future__ import annotations
import json, os
from . import config
from .spec import ProblemSpec, COMPONENTS
from .store import tool


def provider() -> str | None:
    p = os.getenv("LLM_PROVIDER", "").lower()
    if p in ("gemini", "openai"):
        return p if os.getenv("GEMINI_API_KEY" if p == "gemini" else "OPENAI_API_KEY") else None
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return "openai" if os.getenv("OPENAI_API_KEY") else None


def chat(messages: list[dict], temperature: float = 0.0, json_mode: bool = False) -> tuple[str, int, int]:
    from openai import OpenAI
    p = provider()
    if p is None:
        raise RuntimeError("No LLM key found. Put GEMINI_API_KEY=... (no comment on the line) in the .env file.")
    if p == "gemini":
        client, model = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=config.GEMINI_BASE_URL), os.getenv("LLM_MODEL", config.GEMINI_MODEL)
    else:
        client, model = OpenAI(), os.getenv("LLM_MODEL", config.OPENAI_MODEL)
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        r = client.chat.completions.create(model=model, messages=messages, temperature=temperature, **kw)
    except Exception:
        if not json_mode:
            raise
        r = client.chat.completions.create(model=model, messages=messages, temperature=temperature)  # some models reject json_mode
    u = r.usage
    return r.choices[0].message.content, getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0


PARSE_PROMPT = """You convert a football recruitment request into a JSON problem spec. Output ONLY JSON matching this schema:
{schema}
Rules: budgets in EUR (3M€ -> 3000000). Map the request to exactly one Transfermarkt sub-position. Leagues use Transfermarkt ids
(LaLiga=ES1, Premier League=GB1, Bundesliga=L1, Serie A=IT1, Ligue 1=FR1). 'weight_overrides' may only use: {components}.
The data has no style/event information, so if the user asks for a playing style (e.g. 'ball-playing'), do not invent a weight:
add a note to 'assumptions' saying it cannot be applied. List every guess you made in 'assumptions'.
'total_budget_eur' is the combined budget for ALL players. If the request says N players 'under X', assume X is the combined budget
and say so in 'assumptions' (also say it if you assume per-player). If position or budget is missing, output {{"error": "<what to ask>"}}."""


@tool("llm_parse")
def parse_request(text: str) -> tuple[ProblemSpec, int, int]:
    prompt = PARSE_PROMPT.format(schema=json.dumps(ProblemSpec.model_json_schema()), components=", ".join(COMPONENTS))
    content, tin, tout = chat([{"role": "system", "content": prompt}, {"role": "user", "content": text}], 0.0, json_mode=True)
    data = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
    if "error" in data:
        raise ValueError(data["error"])
    return ProblemSpec.model_validate(data), tin, tout


@tool("llm_explain")
def _explain_call(res: dict) -> tuple[str, int, int]:
    system = ("You are a football sporting-director analyst. Explain the decision in <=150 words using ONLY the numbers in the JSON. "
              "Name the main trade-off, the confidence and why, one risk, and cite the source by name. No invented stats. "
              "This is a recommendation, not a done deal: say 'recommend', not 'acquire'. Call a trait strong only if a number shows it. "
              "Describe the confidence using the 'meaning' field; never present top_k as a rank. Mention every player in 'picks' by name.")
    return chat([{"role": "system", "content": system}, {"role": "user", "content": json.dumps(res, default=str)}], 0.2)


def explain(res: dict) -> tuple[str, int, int]:
    try:
        return _explain_call(res)                      # a failure is logged by the @tool wrapper
    except Exception as e:                             # the decision is already made: losing the prose must not lose the answer
        names = ", ".join(p["name"] for p in res["picks"])
        return f"Recommendation: {names} (confidence {res['confidence']['label']}). Explanation unavailable: {type(e).__name__}.", 0, 0
