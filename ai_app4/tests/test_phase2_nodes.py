"""
阶段二节点验证脚本

验证内容：
1. analyze_and_route_node — 中英文 Query 拆解、领域分类
2. parallel_retrieval_node — 多路并发检索、结果融合
3. 整图编译 + 端到端运行（使用真实 wealth 索引）
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from rag_framework.core.config import get_settings
from rag_framework.core.logger import setup_logging
from ai_app4.core.config import WealthSettings
from ai_app4.core.container import WealthContainer
from ai_app4.core import context
from ai_app4.graph.builder import graph
from ai_app4.graph.state import WealthState


def _register_wealth_domain() -> None:
    """显式注册 wealth 领域插件（lifespan 未运行时手动调用）。"""
    import sys
    from pathlib import Path

    pkg = Path(__file__).parent.parent.parent / "domains" / "wealth"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))

    from rag_framework.core.registry import register_domain
    from wealth_domain.plugin import WealthDomainPlugin

    register_domain(WealthDomainPlugin)


async def main() -> None:
    setup_logging()
    _register_wealth_domain()
    settings = WealthSettings()
    context.set_settings(settings)

    container = WealthContainer.from_settings(settings)
    context.set_container(container)
    context.set_settings(settings)


    test_queries = [
        "英伟达最新财报和美联储利率政策有什么关系？",
        "NVIDIA earnings and Fed rate decision impact",
        "CPI 数据对科技股的影响",
        "美联储 5 月议息会议说了什么？",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {q}")
        print(f"{'='*60}")

        state: WealthState = {
            "user_message": q,
            "user_id": "test_user",
            "sub_queries": [],
            "rewritten_queries": [],
            "retrieved_context": None,
            "retrieval_iterations": 0,
            "confidence": 0.0,
            "top_ce": 0.0,
            "evaluation_result": None,
            "needs_tool": False,
            "tool_calls": [],
            "tool_results": [],
            "math_result": None,
            "reply": "",
            "history": [],
            "trace": [],
        }

        result = await graph.ainvoke(state, config={"configurable": {"thread_id": "test_user"}})

        # 打印 analyze 结果
        sqs = result.get("sub_queries", [])
        print(f"  sub_queries: {len(sqs)} 路")
        for sq in sqs:
            print(f"    domain={sq.get('domain')}, weight={sq.get('weight')}, text={sq.get('text')[:60]}...")
            if "text_en" in sq:
                print(f"    [EN] {sq['text_en'][:80]}...")

        # 打印检索结果
        ctx = result.get("retrieved_context")
        if ctx:
            print(f"  检索命中: {len(ctx)} 字符")
            print(f"  上下文摘要: {ctx[:200]}...")
        else:
            print("  检索未命中上下文")

        # 打印 trace
        for t in result.get("trace", []):
            print(f"  [{t.get('node')}] { {k:v for k,v in t.items() if k != 'node'} }")

    print("\n✅ 阶段二节点验证完成")


if __name__ == "__main__":
    asyncio.run(main())
