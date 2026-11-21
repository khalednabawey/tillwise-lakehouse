"""The agent graph: model -> tools loop, then a grounding check before the answer leaves.

    START -> agent --(tool calls)--> tools -> agent ... --(answer)--> ground --(ok)--> END
                                                                         \\--(ungrounded, 1 retry)--> agent
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from tillwise.agent import prompts
from tillwise.agent.config import settings
from tillwise.agent.grounding import ungrounded
from tillwise.agent.llm import chat_model
from tillwise.agent.tools import Toolbox, ToolError


# ---------------------------------------------------------------- tool schemas
class Period(BaseModel):
    start_date: str | None = Field(None, description="YYYY-MM-DD, inclusive. Omit for the start of the data.")
    end_date: str | None = Field(None, description="YYYY-MM-DD, inclusive. Omit for the end of the data.")


class SalesArgs(Period):
    country: str | None = Field(None, description="Country name, e.g. 'United Kingdom', 'France'. Omit for all.")
    stock_code: str | None = Field(None, description="Product stock code, e.g. '85123A'. Omit for all products.")


class TopArgs(Period):
    metric: str = Field("revenue", description="'revenue' (net revenue) or 'units' (units sold)")
    n: int = Field(10, description="How many products, 1-25")
    country: str | None = Field(None, description="Country name. Omit for all.")


class CountryArgs(Period):
    n: int = Field(10, description="How many countries, 1-25")


class TrendArgs(Period):
    country: str | None = Field(None, description="Country name. Omit for all.")


class ReturnsArgs(Period):
    by: str = Field("country", description="'country' or 'product'")
    n: int = Field(10, description="How many rows, 1-25")


class FindArgs(BaseModel):
    text: str = Field(..., description="Words from the product description, or a stock code")
    n: int = Field(10, description="How many matches, 1-25")


class ForecastArgs(BaseModel):
    stock_code: str = Field(..., description="Product stock code (use find_products first if you only have a name)")
    country: str = Field("United Kingdom", description="Country name")
    weeks: int = Field(8, description="Weeks ahead, 1-8")


class NoArgs(BaseModel):
    pass


DESCRIPTIONS = {
    "data_coverage": ("What the data covers: first/last day, countries, numbers of products, customers, orders.",
                      NoArgs),
    "sales_summary": ("Net/gross revenue, returns value, units sold, orders and customers for a period, "
                      "optionally for one country and/or one product.", SalesArgs),
    "top_products": ("Top products by net revenue or units sold for a period, optionally in one country.", TopArgs),
    "country_breakdown": ("Net revenue, orders and customers by country for a period, highest first.", CountryArgs),
    "monthly_trend": ("Net revenue and orders per month for a period, optionally for one country.", TrendArgs),
    "returns_report": ("Value of returned goods and returns as % of sales, by country or by product.", ReturnsArgs),
    "customer_segments": ("Customer segments (RFM): customers, net revenue and revenue share per segment.", NoArgs),
    "find_products": ("Find products by words in their description or by stock code.", FindArgs),
    "demand_forecast": ("Weekly unit forecast (with p10/p90 range) for the next 1-8 weeks after 2011-11-28, "
                        "for one product in one country.", ForecastArgs),
}


def build_tools(box: Toolbox) -> list[StructuredTool]:
    tools = []
    for name, (desc, schema) in DESCRIPTIONS.items():
        fn = getattr(box, name)
        tools.append(StructuredTool.from_function(func=fn, name=name, description=desc, args_schema=schema))
    return tools


# ---------------------------------------------------------------- graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_log: list[dict[str, Any]]
    steps: int
    grounding_retries: int
    ungrounded: list[float]


def build_graph(box: Toolbox, model: Runnable | None = None, max_steps: int | None = None):
    tools = build_tools(box)
    by_name = {t.name: t for t in tools}
    llm = model or chat_model(tools)
    max_steps = max_steps or settings().max_steps

    def agent(state: AgentState) -> dict:
        msgs = [SystemMessage(prompts.SYSTEM), *state["messages"]]
        if state["steps"] >= max_steps:
            msgs.append(HumanMessage("You have used all tool calls. Answer now from the tool results above."))
            reply = llm.invoke(msgs)
            reply.tool_calls = []  # no more tools: force an answer
        else:
            reply = llm.invoke(msgs)
        return {"messages": [reply], "steps": state["steps"] + 1}

    def run_tools(state: AgentState) -> dict:
        last: AIMessage = state["messages"][-1]
        out, log = [], list(state["tool_log"])
        for call in last.tool_calls:
            tool = by_name.get(call["name"])
            try:
                if tool is None:
                    raise ToolError(f"Unknown tool {call['name']}")
                result = tool.invoke(call["args"])
                content = json.dumps(result, default=str)
                ok = True
            except ToolError as exc:
                content, ok = json.dumps({"error": str(exc)}), False
            except Exception as exc:  # noqa: BLE001 - validation errors from bad arguments
                content, ok = json.dumps({"error": f"{type(exc).__name__}: {exc}"}), False
            out.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
            log.append({"tool": call["name"], "args": call["args"], "ok": ok, "result": content})
        return {"messages": out, "tool_log": log}

    def ground(state: AgentState) -> dict:
        answer = state["messages"][-1].content or ""
        bad = ungrounded(str(answer), [t["result"] for t in state["tool_log"] if t["ok"]])
        if bad and state["grounding_retries"] < 1:
            nums = ", ".join(f"{b:,.2f}" for b in bad[:6])
            return {"messages": [HumanMessage(prompts.GROUNDING_RETRY.format(numbers=nums))],
                    "grounding_retries": state["grounding_retries"] + 1, "ungrounded": bad}
        return {"ungrounded": bad}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "ground"

    def after_ground(state: AgentState) -> str:
        return "agent" if isinstance(state["messages"][-1], HumanMessage) else END

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_node("ground", ground)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", "ground": "ground"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("ground", after_ground, {"agent": "agent", END: END})
    return g.compile()


def ask(graph, question: str) -> dict[str, Any]:
    state = graph.invoke({"messages": [HumanMessage(question)], "tool_log": [], "steps": 0,
                          "grounding_retries": 0, "ungrounded": []},
                         {"recursion_limit": 40})
    final: AIMessage = state["messages"][-1]
    meta = final.response_metadata or {}
    return {
        "answer": final.content,
        "tool_calls": [{"tool": t["tool"], "args": t["args"], "ok": t["ok"]} for t in state["tool_log"]],
        "tool_results": [t["result"] for t in state["tool_log"]],
        "ungrounded_numbers": state["ungrounded"],
        "provider": meta.get("provider"),
        "model": meta.get("model"),
    }
