"""
PyTorch 轻量任务模型层。

提供意图分类、情感分析、命名实体识别等任务的统一抽象和实现。
依赖 transformers + torch，采用惰性加载策略，ai_app1/2/3 不触发此模块加载。
"""
from rag_framework.torch_models.base import TorchTaskModel, TaskPrediction
from rag_framework.torch_models.intent_classifier import IntentClassifier
from rag_framework.torch_models.sentiment_analyzer import SentimentAnalyzer
from rag_framework.torch_models.entity_recognizer import EntityRecognizer

__all__ = [
    "TorchTaskModel",
    "TaskPrediction",
    "IntentClassifier",
    "SentimentAnalyzer",
    "EntityRecognizer",
]
