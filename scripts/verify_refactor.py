"""
重构验证脚本

检查：
1. 所有模块文件语法正确
2. rag_framework + android_domain + ai_app1 可正常导入
3. RAGSettings 类型检查
4. AndroidDomainPlugin 接口检查
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "domains" / "android"))


# ─── 1. 语法检查 ────────────────────────────────────────────────────────────────

def check_syntax(files: list[str]) -> bool:
    ok = True
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"  [FAIL] 文件缺失: {f}")
            ok = False
            continue
        try:
            ast.parse(p.read_text())
        except SyntaxError as e:
            print(f"  [FAIL] 语法错误 {f}: {e}")
            ok = False
    if ok:
        print("  [PASS] 所有文件语法正确")
    return ok


# ─── 2. 模块导入检查 ────────────────────────────────────────────────────────────

def check_imports() -> bool:
    errors = []
    try:
        from rag_framework.core.config import RAGSettings, get_settings
        from rag_framework.core.registry import PluginRegistry, register_domain, get_domain
        from rag_framework.core.logger import setup_logging, chat_logger, retrieval_logger
        from rag_framework.core.exceptions import RAGError, ModelLoadError, RetrievalError
        from rag_framework.domain.base import DomainPlugin, CollectionNames, QueryRoute, DomainPrompts
        from rag_framework.embedding.base import Embedder
        from rag_framework.embedding.sentence_transformer import STEmbedder
        from rag_framework.rerank.base import Reranker, RankedDoc
        from rag_framework.rerank.cross_encoder import CrossEncoderReranker
        from rag_framework.retrieval.base import Retriever, RetrievedDoc, RetrievalResult
        from rag_framework.retrieval.dense import DenseStore
        from rag_framework.retrieval.sparse import BM25Store
        from rag_framework.retrieval.fusion import HybridRetriever, HybridConfig
        from rag_framework.retrieval.query_rewriter.base import QueryRewriter
        from rag_framework.llm.base import LLMClient
        from rag_framework.llm.openai_client import OpenAILLMClient
        from rag_framework.llm.tool_registry import register_tool, get_tool_definitions, execute_tool, list_tools  # noqa: F401
        from rag_framework.session.base import SessionStore, SessionData
        from rag_framework.session.memory_store import MemorySessionStore
        from rag_framework.session.manager import SessionManager
        from rag_framework.container import RAGContainer
        from rag_framework.indexing.chunker import chunk_paragraphs, chunk_file, chunk_text  # noqa: F401
        from rag_framework.eval.metrics import (
            EvalMetrics, LatencyStats, compute_latency_stats,
            recall_at_k, reciprocal_rank, hit_at_k, aggregate_metrics,
        )
        from rag_framework.eval.hit_judge import is_hit, ground_truth_ids
        from rag_framework.eval.ranking import run_ranking_eval, evaluate_single_query, load_dataset
        from rag_framework.eval.qa import QAJudge, run_qa_eval, evaluate_single_qa
        from rag_framework.eval.ablation import run_ablation_study
        from rag_framework.eval.experiment import run_experiment
    except Exception as e:
        errors.append(f"rag_framework 导入失败: {e}")

    try:
        from android_domain import AndroidDomainPlugin
    except Exception as e:
        errors.append(f"android_domain 导入失败: {e}")

    try:
        from ai_app1.main import app  # noqa: F401
        from ai_app1.api.chat import router, ChatRequest  # noqa: F401
    except Exception as e:
        errors.append(f"ai_app1 导入失败: {e}")

    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False
    print("  [PASS] 所有模块导入成功")
    return True


# ─── 3. 配置类型检查 ────────────────────────────────────────────────────────────

def check_config_types() -> bool:
    try:
        from rag_framework.core.config import RAGSettings
        s = RAGSettings()
        assert isinstance(s.active_domain, str)
        assert isinstance(s.llm_backend, str)
        assert isinstance(s.embed_model_path, str)
        # rewriter_llm 配置组
        assert isinstance(s.rewriter_llm_backend, str)
        assert isinstance(s.resolved_rewriter_llm_backend, str)
        # 若 rewriter_llm_backend 为空则回退到 llm_backend；若已显式设置则允许不同
        if not s.rewriter_llm_backend.strip():
            assert s.resolved_rewriter_llm_backend == s.llm_backend
        print("  [PASS] RAGSettings 类型正确（含 rewriter_llm）")
        return True
    except Exception as e:
        print(f"  [FAIL] RAGSettings 检查失败: {e}")
        return False


# ─── 4. 领域插件检查 ────────────────────────────────────────────────────────────

def check_domain_plugin() -> bool:
    try:
        from android_domain import AndroidDomainPlugin
        from rag_framework.domain.base import CollectionNames

        plugin = AndroidDomainPlugin()
        assert plugin.name == "android"
        assert isinstance(plugin.system_prompt, str) and len(plugin.system_prompt) > 0
        collections = plugin.get_collection_names()
        assert isinstance(collections, CollectionNames)
        assert collections.parent == "knowledge_base"
        assert collections.child == "knowledge_base_child"
        assert collections.hyde == "knowledge_base_hyde"
        print("  [PASS] AndroidDomainPlugin 接口正确")
        return True
    except Exception as e:
        print(f"  [FAIL] 领域插件检查失败: {e}")
        return False


# ─── 主入口 ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("RAG 框架重构验证")
    print("=" * 60)

    files = [
        # Framework
        "rag_framework/rag_framework/__init__.py",
        "rag_framework/rag_framework/core/config.py",
        "rag_framework/rag_framework/core/exceptions.py",
        "rag_framework/rag_framework/core/logger.py",
        "rag_framework/rag_framework/core/registry.py",
        "rag_framework/rag_framework/domain/base.py",
        "rag_framework/rag_framework/embedding/base.py",
        "rag_framework/rag_framework/embedding/sentence_transformer.py",
        "rag_framework/rag_framework/rerank/base.py",
        "rag_framework/rag_framework/rerank/cross_encoder.py",
        "rag_framework/rag_framework/rerank/fallback.py",
        "rag_framework/rag_framework/retrieval/base.py",
        "rag_framework/rag_framework/retrieval/dense.py",
        "rag_framework/rag_framework/retrieval/sparse.py",
        "rag_framework/rag_framework/retrieval/fusion.py",
        "rag_framework/rag_framework/retrieval/query_rewriter/base.py",
        "rag_framework/rag_framework/retrieval/query_rewriter/llm_rewriter.py",
        "rag_framework/rag_framework/retrieval/query_rewriter/rule_rewriter.py",
        "rag_framework/rag_framework/llm/base.py",
        "rag_framework/rag_framework/llm/openai_client.py",
        "rag_framework/rag_framework/llm/tool_registry.py",
        "rag_framework/rag_framework/session/base.py",
        "rag_framework/rag_framework/session/memory_store.py",
        "rag_framework/rag_framework/session/manager.py",
        "rag_framework/rag_framework/container.py",
        "rag_framework/rag_framework/indexing/chunker.py",
        "rag_framework/rag_framework/indexing/hyde.py",
        "rag_framework/rag_framework/indexing/indexer.py",
        "rag_framework/rag_framework/eval/metrics.py",
        "rag_framework/rag_framework/eval/hit_judge.py",
        "rag_framework/rag_framework/eval/ranking.py",
        "rag_framework/rag_framework/eval/qa.py",
        "rag_framework/rag_framework/eval/ablation.py",
        "rag_framework/rag_framework/eval/experiment.py",
        # Domain
        "domains/android/android_domain/__init__.py",
        "domains/android/android_domain/plugin.py",
        # App layer
        "ai_app1/main.py",
        "ai_app1/api/chat.py",
        "ai_app1/tests/test_api.py",
        "ai_app3/core/llm_provider.py",
        "ai_app3/service/query_engine.py",
        "ai_app3/service/evaluator.py",
        "ai_app3/service/context_compressor.py",
        "ai_app3/graph/builder.py",
    ]

    results = [
        ("语法检查", check_syntax(files)),
        ("模块导入", check_imports()),
        ("配置类型", check_config_types()),
        ("领域插件", check_domain_plugin()),
    ]

    print("=" * 60)
    all_pass = all(r[1] for r in results)
    if all_pass:
        print("全部通过 ✅")
        return 0
    else:
        print("存在失败项 ❌")
        return 1


if __name__ == "__main__":
    sys.exit(main())
