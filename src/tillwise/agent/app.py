"""FastAPI app for the data agent.

    uv run uvicorn tillwise.agent.app:app --reload
    POST /ask   {"question": "What was net revenue in the UK in 2011?"}
    GET  /health
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from tillwise.agent.graph import ask, build_graph
from tillwise.agent.store import store_from_env
from tillwise.agent.tools import TOOL_NAMES, Toolbox

app = FastAPI(title="Tillwise data agent", version="0.1.0",
              description="Answers business questions from the Tillwise gold layer with fixed, read-only tools.")


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Answer(BaseModel):
    answer: str
    tool_calls: list[dict[str, Any]]
    ungrounded_numbers: list[float]
    provider: str | None = None
    model: str | None = None


@lru_cache(maxsize=1)
def get_toolbox() -> Toolbox:
    return Toolbox(store_from_env())


@lru_cache(maxsize=1)
def get_graph():
    return build_graph(get_toolbox())


@app.get("/health")
def health(box: Annotated[Toolbox, Depends(get_toolbox)]) -> dict[str, Any]:
    cov = box.data_coverage()
    return {"status": "ok", "tools": TOOL_NAMES, "data": {"first_day": cov["first_day"], "last_day": cov["last_day"]}}


@app.post("/ask", response_model=Answer)
def ask_question(q: Question, graph: Annotated[Any, Depends(get_graph)]) -> Answer:
    try:
        out = ask(graph, q.question)
    except Exception as exc:  # noqa: BLE001 - both providers down, or a bug: say so clearly
        raise HTTPException(status_code=503,
                            detail=f"The agent could not answer right now ({type(exc).__name__}).") from exc
    return Answer(**{k: out[k] for k in ("answer", "tool_calls", "ungrounded_numbers", "provider", "model")})
