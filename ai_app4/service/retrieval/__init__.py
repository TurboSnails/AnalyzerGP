"""
ai_app4 检索层扩展。

ThreeTrackRetriever：三轨融合检索器。
  - Track A：本地 HybridRetriever（历史知识）
  - Track B：金融 API（实时股价、宏观指标）
  - Track C：网络搜索（突发新闻、市场情绪）
"""

from ai_app4.service.retrieval.three_track_retriever import ThreeTrackRetriever

__all__ = ["ThreeTrackRetriever"]
