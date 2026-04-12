"""Ad Executor — LangGraph-based ReAct agent for recommendation decisions.

Uses LangGraph's StateGraph with ToolNode to implement the ReAct pattern:
  Observe (user profiles + content pool)
  → Think (analyze audience, match content)
  → Act (call tools: analyze_audience, match_user_content, generate_recommendation)
  → Observe (tool results)
  → ... loop until all recommendations are made

This is a genuine LangGraph implementation with:
- StateGraph for workflow orchestration
- ToolNode for tool execution
- Conditional edges for ReAct loop control
- State persistence across the graph
"""

from __future__ import annotations

import json
import operator
from dataclasses import asdict
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.memory.strategy_lib import StrategyLibrary
from src.models import ContentItem, RecommendAction, UserProfile
from src.tools.ad_tools import get_all_tools


# --- LangGraph State Definition ---

class AgentState(TypedDict):
    """State that flows through the LangGraph."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    users_json: str  # serialized user profiles
    contents_json: str  # serialized content items
    recommendations: list[dict]  # collected recommendations


# --- System Prompt ---

EXECUTOR_SYSTEM_PROMPT = """你是一个智能广告推荐 Agent。你需要通过分析用户画像和内容特征，为每个用户推荐最合适的内容。

## 你的工作流程（ReAct 模式）
1. 调用 analyze_audience 分析用户群体特征
2. 调用 analyze_content_pool 分析内容池
3. 对每个用户调用 match_user_content 找到最佳匹配
4. 对每个用户调用 generate_recommendation 生成最终推荐

## 可用策略
{strategies}

## 重要规则
- 你必须为每个用户生成一条推荐
- 使用工具来分析和匹配，不要跳过工具直接给结果
- 推荐理由要具体，说明用户兴趣与内容的关联
- 如果有可用策略，优先按策略执行"""


class AdExecutor:
    """LangGraph-based recommendation agent with ReAct tool-calling loop."""

    def __init__(self, llm: ChatOpenAI, strategy_lib: StrategyLibrary | None = None):
        self.llm = llm
        self.strategy_lib = strategy_lib
        self.tools = get_all_tools()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph with ReAct pattern."""

        # Bind tools to the LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # --- Node: Agent reasoning ---
        def agent_node(state: AgentState) -> dict:
            """The agent thinks and decides whether to call a tool or finish."""
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        # --- Node: Tool execution ---
        tool_node = ToolNode(self.tools)

        # --- Node: Extract recommendations from conversation ---
        def extract_node(state: AgentState) -> dict:
            """Extract final recommendations from the agent's messages."""
            recommendations = []
            for msg in state["messages"]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc["name"] == "generate_recommendation":
                            recommendations.append(tc["args"])
            return {"recommendations": recommendations}

        # --- Conditional edge: should we continue or stop? ---
        def should_continue(state: AgentState) -> str:
            """Check if the agent wants to call more tools or is done."""
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            return "extract"

        # --- Build the graph ---
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_node("extract", extract_node)

        # Set entry point
        graph.set_entry_point("agent")

        # Add edges
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "extract": "extract"})
        graph.add_edge("tools", "agent")  # After tool execution, go back to agent
        graph.add_edge("extract", END)

        return graph.compile()

    def execute(
        self,
        users: list[UserProfile],
        contents: list[ContentItem],
        scenario: str = "",
    ) -> list[RecommendAction]:
        """Run the LangGraph agent to generate recommendations."""

        # Prepare data
        users_data = [
            {
                "user_id": u.user_id,
                "gender": u.gender.value,
                "age_range": u.age_range,
                "province": u.province,
                "city": u.city,
                "city_level": u.city_level,
                "interests": u.interests,
                "active_degree": u.active_degree,
                "device_price": u.device_price,
            }
            for u in users
        ]
        contents_data = [
            {
                "item_id": c.item_id,
                "caption": c.caption,
                "topic_tags": c.topic_tags,
                "category_l1": c.category_l1,
                "category_l2": c.category_l2,
                "category_l3": c.category_l3,
                "duration_seconds": c.duration_seconds,
            }
            for c in contents
        ]

        users_json = json.dumps(users_data, ensure_ascii=False)
        contents_json = json.dumps(contents_data, ensure_ascii=False)

        # Build initial messages
        strategies_text = self._load_strategies(scenario)
        system_msg = SystemMessage(
            content=EXECUTOR_SYSTEM_PROMPT.format(strategies=strategies_text)
        )
        human_msg = HumanMessage(
            content=(
                f"请为以下 {len(users)} 个用户推荐内容。\n\n"
                f"用户画像 JSON:\n```json\n{users_json}\n```\n\n"
                f"内容池 JSON:\n```json\n{contents_json}\n```\n\n"
                f"请使用工具逐步分析并生成推荐。"
            )
        )

        # Run the graph
        initial_state: AgentState = {
            "messages": [system_msg, human_msg],
            "users_json": users_json,
            "contents_json": contents_json,
            "recommendations": [],
        }

        try:
            final_state = self.graph.invoke(initial_state)
            recommendations = final_state.get("recommendations", [])
        except Exception:
            # Fallback: if graph fails, use simple interest matching
            recommendations = self._fallback_match(users, contents)

        return self._to_actions(recommendations, users, contents)

    def _load_strategies(self, scenario: str) -> str:
        if not self.strategy_lib or self.strategy_lib.count() == 0:
            return "暂无可用策略，请根据用户画像和内容特征做最佳判断。"
        strategies = self.strategy_lib.search(scenario) if scenario else []
        if not strategies:
            index = self.strategy_lib.list_all()
            return "可用策略：\n" + "\n".join(
                f"- {s['name']}: {s['applicable_scenario']}" for s in index[:5]
            )
        parts = []
        for s in strategies[:3]:
            steps = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(s.execution_steps))
            parts.append(
                f"### {s.name}\n适用场景: {s.applicable_scenario}\n"
                f"目标人群: {s.target_audience}\n执行步骤:\n{steps}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _fallback_match(users: list[UserProfile], contents: list[ContentItem]) -> list[dict]:
        """Simple interest-based matching when graph execution fails."""
        results = []
        for u in users:
            best = contents[0]
            for c in contents:
                if any(
                    interest in c.category_l1 or (c.topic_tags and interest in c.topic_tags[0])
                    for interest in u.interests
                ):
                    best = c
                    break
            results.append({
                "user_id": u.user_id,
                "item_id": best.item_id,
                "targeting_reason": f"兴趣匹配: {u.interests} → {best.category_l1}",
            })
        return results

    @staticmethod
    def _to_actions(
        recommendations: list[dict],
        users: list[UserProfile],
        contents: list[ContentItem],
    ) -> list[RecommendAction]:
        """Convert raw recommendation dicts to RecommendAction objects."""
        user_map = {u.user_id: u for u in users}
        content_map = {c.item_id: c for c in contents}

        actions = []
        matched_users = set()
        for r in recommendations:
            uid = r.get("user_id", "")
            iid = r.get("item_id", "")
            user = user_map.get(uid)
            content = content_map.get(iid)
            if user and content:
                actions.append(RecommendAction(
                    user=user,
                    content=content,
                    targeting_reason=r.get("targeting_reason", ""),
                ))
                matched_users.add(uid)

        # Fill in any users that didn't get a recommendation
        for u in users:
            if u.user_id not in matched_users:
                actions.append(RecommendAction(
                    user=u,
                    content=contents[0],
                    targeting_reason="默认推荐（未被工具匹配到）",
                ))

        return actions
