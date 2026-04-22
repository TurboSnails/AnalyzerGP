"""
文件缓存模块 - 避免重复请求外部数据源
- 缓存格式: JSON（基础类型）+ Pickle（DataFrame）
- 缓存过期: 基于 TTL（小时），过期自动重新获取
- 缓存目录: output/.cache/
"""
import os
import json
import pickle
import hashlib
import time
from typing import Any, Optional, Callable
from functools import wraps

import pandas as pd

from config import DATA_CONFIG


CACHE_DIR = DATA_CONFIG["cache_dir"]
CACHE_TTL = DATA_CONFIG["cache_ttl_hours"] * 3600   # 转为秒


def _cache_path(key: str, ext: str = "pkl") -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{safe_key}.{ext}")


def _is_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    return (time.time() - os.path.getmtime(path)) < CACHE_TTL


def cache_get(key: str) -> Optional[Any]:
    """从缓存读取，过期返回 None"""
    path = _cache_path(key)
    if not _is_fresh(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def cache_set(key: str, value: Any) -> None:
    """写入缓存"""
    path = _cache_path(key)
    try:
        with open(path, "wb") as f:
            pickle.dump(value, f)
    except Exception:
        pass   # 缓存写失败不影响主流程


def cached(key_prefix: str):
    """
    装饰器：自动缓存函数返回值

    用法:
        @cached("stock_info")
        def get_stock_info(self, symbol):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 用参数构造缓存 key
            arg_str = "_".join(str(a) for a in args[1:]) + "_".join(
                f"{k}={v}" for k, v in sorted(kwargs.items())
            )
            key = f"{key_prefix}_{arg_str}"
            cached_val = cache_get(key)
            if cached_val is not None:
                return cached_val
            result = func(*args, **kwargs)
            # 只缓存非空结果
            if result is not None and not (isinstance(result, pd.DataFrame) and result.empty):
                cache_set(key, result)
            return result
        return wrapper
    return decorator


def clear_cache(symbol: str = None) -> int:
    """
    清除缓存。symbol 为 None 时清除全部。
    返回删除文件数。
    """
    if not os.path.exists(CACHE_DIR):
        return 0
    count = 0
    for f in os.listdir(CACHE_DIR):
        full = os.path.join(CACHE_DIR, f)
        if symbol is None or symbol in f:
            try:
                os.remove(full)
                count += 1
            except OSError:
                pass
    return count
