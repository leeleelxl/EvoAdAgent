"""User retrieval tools — L2 FAISS semantic search wired into the ReAct loop.

Factory pattern: the tool closes over a UserProfileStore reference, so the
ReAct agent can actually query the index during recommendation decisions.
Without this, the L2 FAISS index is built but never consumed — dead weight.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.memory.user_profile import UserProfileStore


def build_find_similar_users_tool(store: UserProfileStore):
    """Build a tool that queries the given UserProfileStore via FAISS semantic search.

    The tool is a closure: each call to this factory returns a tool bound to a
    specific store instance. This keeps the tool stateless from LangChain's
    perspective while letting the executor reuse a warm FAISS index.
    """

    @tool
    def find_similar_users(user_id: str, k: int = 3) -> str:
        """查找与给定用户画像语义相似的 k 个历史用户，返回其人口属性和兴趣摘要。

        用途：给某用户做推荐前，先看画像相似的历史用户偏好什么内容，辅助冷启动或
        推理长尾用户偏好。返回的相似用户画像可以作为推荐决策的锚点参考。

        Args:
            user_id: 目标用户的 user_id（必须是已在 L2 索引中的用户）
            k: 返回的相似用户数量，默认 3，最多不超过索引大小

        Returns:
            Markdown 格式的相似用户列表，含人口属性、城市、兴趣标签
        """
        if store.count() == 0:
            return "L2 用户画像索引为空，无法检索相似用户。"

        target = store.get(user_id)
        if target is None:
            return (
                f"用户 {user_id} 不在 L2 索引中。"
                f"当前索引包含 {store.count()} 个用户。"
            )

        # +1 because the query user itself will be the nearest neighbor (dist=0).
        results = store.search_similar(target, k=min(k + 1, store.count()))

        lines: list[str] = []
        for u, dist in results:
            if u.user_id == user_id:
                continue
            lines.append(
                f"- **{u.user_id}** (L2 距离 {dist:.3f})  "
                f"{u.gender.value} · {u.age_range}岁 · "
                f"{u.city_level}城市({u.city}) · 兴趣={u.interests}"
            )
            if len(lines) >= k:
                break

        if not lines:
            return f"未找到与 {user_id} 相似的其他用户。"

        return (
            f"## 与用户 {user_id} 语义最相似的 {len(lines)} 个历史用户\n\n"
            + "\n".join(lines)
            + "\n\n以上用户的兴趣和属性可作为推荐决策的参考锚点。"
        )

    return find_similar_users
