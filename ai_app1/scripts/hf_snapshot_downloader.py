"""
Hugging Face Hub snapshot 下载封装：预设轻量模型、镜像、磁盘统计、续传锁清理。
"""

from __future__ import annotations

import os
import stat as stat_lib
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.utils.tqdm import tqdm as HfTqdm

# 大陆常用镜像；也可在 shell 里 export HF_ENDPOINT=...
DEFAULT_CN_MIRROR = "https://hf-mirror.com"


def _configure_ide_friendly_progress() -> None:
    """
    IDE / 任务面板运行脚本时，stdout 常不是 TTY，tqdm 会关掉「按字节的」单行进度，
    只能看到外层的 Fetching N files（按已完成文件数）。
    huggingface_hub.utils.tqdm.is_tqdm_disabled：设置了 TQDM_POSITION=-1 时强制绘制 http_get 的进度条。
    """
    os.environ.setdefault("TQDM_POSITION", "-1")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except OSError:
                pass


@dataclass(frozen=True)
class HfModelPreset:
    """下载预设：repo、slim 模式下忽略的 glob（None 表示 slim 也不裁剪）。"""

    repo_id: str
    slim_ignore: tuple[str, ...] | None
    default_subdir: str


# bge-m3 体积大、Hub 易不稳定；bge-base-zh 约数百 MB 级，中文检索常用，更适合弱网先跑通。
PRESETS: dict[str, HfModelPreset] = {
    "bge-m3": HfModelPreset(
        "BAAI/bge-m3",
        ("onnx/**", "imgs/**", "long.jpg"),
        "bge-m3",
    ),
    "bge-base-zh": HfModelPreset(
        "BAAI/bge-base-zh-v1.5",
        None,
        "bge-base-zh-v1.5",
    ),
    "bge-small-zh": HfModelPreset(
        "BAAI/bge-small-zh-v1.5",
        None,
        "bge-small-zh-v1.5",
    ),
    "minilm-m": HfModelPreset(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        None,
        "paraphrase-multilingual-MiniLM-L12-v2",
    ),
}


def _disk_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _int_env(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        v = default
    else:
        try:
            v = int(raw.strip())
        except ValueError:
            v = default
    return max(lo, min(hi, v))


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _hf_token_for_snapshot() -> str | bool | None:
    """
    传给 snapshot_download 的 token。
    未设 HF_TOKEN 时默认 False（不附带本机缓存 token；避免过期 token 打镜像 401）。
    需要 hf auth login 的隐式 token：HF_IMPLICIT_TOKEN=1。
    """
    raw = os.getenv("HF_TOKEN")
    if raw is None:
        implicit = os.getenv("HF_IMPLICIT_TOKEN", "").strip().lower() in ("1", "true", "yes", "on")
        return None if implicit else False
    s = raw.strip()
    if not s or s.lower() in ("false", "0", "no", "anonymous", "none"):
        return False
    return s


class _LiveHubTqdm(HfTqdm):
    """缩短 tqdm 刷新间隔，弱网下避免长时间 0% 像卡死。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mininterval", 0.2)
        kwargs.setdefault("miniters", 1)
        kwargs.setdefault("smoothing", 0.12)
        kwargs.setdefault("file", sys.stderr)
        super().__init__(*args, **kwargs)


def _list_hf_cache_lockfiles(local_dir: Path) -> list[Path]:
    cache = local_dir / ".cache" / "huggingface"
    if not cache.is_dir():
        return []
    out: list[Path] = []
    for dirpath, _names, filenames in os.walk(os.fspath(cache), followlinks=False):
        for name in filenames:
            if name.endswith(".lock"):
                out.append(Path(dirpath, name))
    return out


def clear_hf_cache_lockfiles(local_dir: Path) -> int:
    n = 0
    for p in _list_hf_cache_lockfiles(local_dir):
        try:
            p.unlink()
            n += 1
        except OSError as e:
            _disk_log(f"[unlock] 未删除 {p}: {e}")
    return n


def _tree_bytes(root: Path) -> int:
    root_s = os.fspath(root)
    if not os.path.isdir(root_s):
        return 0
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(root_s, topdown=True, followlinks=False):
            for name in filenames:
                fp = os.path.join(dirpath, name)
                try:
                    st = os.lstat(fp)
                except OSError:
                    continue
                if stat_lib.S_ISREG(st.st_mode):
                    total += st.st_size
    except OSError:
        pass
    return total


@contextmanager
def _disk_stats_loop(root: Path, interval_sec: float) -> Iterator[None]:
    if interval_sec <= 0:
        yield
        return

    stop = threading.Event()

    def _run() -> None:
        t0 = time.monotonic()
        b0 = _tree_bytes(root)
        _disk_log(
            f"[disk-stats] 基准 {b0 / (1024**3):.3f} GiB，每 {interval_sec:g}s 报一次速率（stderr）"
        )
        prev_b, prev_t = b0, t0
        while not stop.wait(interval_sec):
            now = time.monotonic()
            b = _tree_bytes(root)
            dt = max(now - prev_t, 1e-6)
            db = max(b - prev_b, 0)
            inst_mib = (db / dt) / (1024 * 1024)
            span = max(now - t0, 1e-6)
            avg_mib = max(b - b0, 0) / span / (1024 * 1024)
            _disk_log(
                f"[disk-stats] 累计≈{b / (1024**3):.3f} GiB  "
                f"瞬时≈{inst_mib:.2f} MiB/s  全程均≈{avg_mib:.2f} MiB/s"
            )
            prev_b, prev_t = b, now

    th = threading.Thread(target=_run, name="disk-stats", daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=interval_sec + 3.0)


class HfSnapshotDownloader:
    """
    snapshot_download 封装：同一套磁盘统计 / 锁清理 / tqdm / 并发配置。
    """

    def __init__(
        self,
        repo_id: str,
        local_dir: Path,
        *,
        slim_ignore: Sequence[str] | None = None,
        endpoint: str | None = None,
        token: str | bool | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.local_dir = local_dir.expanduser().resolve()
        self._slim_ignore = tuple(slim_ignore) if slim_ignore is not None else None
        self._endpoint_override = endpoint
        self._token = token

    @classmethod
    def from_preset(cls, preset_key: str, models_parent: Path, *, endpoint: str | None = None) -> HfSnapshotDownloader:
        if preset_key not in PRESETS:
            keys = ", ".join(sorted(PRESETS))
            raise ValueError(f"未知 preset: {preset_key!r}，可选: {keys}")
        p = PRESETS[preset_key]
        out = models_parent / p.default_subdir
        return cls(p.repo_id, out, slim_ignore=p.slim_ignore, endpoint=endpoint)

    def resolved_endpoint(self) -> str | None:
        if self._endpoint_override is not None:
            return self._endpoint_override
        return os.getenv("HF_ENDPOINT") or None

    def _ignore_for_slim(self, slim: bool) -> list[str] | None:
        if not slim:
            return None
        if self._slim_ignore is None:
            return None
        return list(self._slim_ignore)

    def maybe_clear_locks(self, unlock: bool) -> int:
        if not unlock:
            return 0
        n = clear_hf_cache_lockfiles(self.local_dir)
        _disk_log(f"[unlock] 已移除 {n} 个 huggingface *.lock")
        return n

    def warn_locks_if_needed(self, endpoint: str | None) -> None:
        locks = _list_hf_cache_lockfiles(self.local_dir)
        if not locks:
            return
        sample = locks[0].name
        mirror = ""
        if not endpoint:
            mirror = f" （未设 HF_ENDPOINT 时脚本默认走镜像；仍直连官方可 export HF_ENDPOINT= 或见 CLI --no-mirror）"
        _disk_log(
            f"[hint] 缓存目录下有 {len(locks)} 个 *.lock（如 {sample}）。"
            " 续传若长期 0%：--unlock 或 HF_DOWNLOAD_UNLOCK=1。"
            f"{mirror}"
        )

    def download(
        self,
        *,
        full: bool = False,
        dry_run: bool = False,
        force: bool = False,
        unlock: bool = False,
        disk_stats_interval: float = 0.0,
    ) -> Path:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        _configure_ide_friendly_progress()

        env_unlock = os.getenv("HF_DOWNLOAD_UNLOCK", "").strip().lower() in ("1", "true", "yes", "on")
        if unlock or env_unlock:
            if not dry_run:
                self.maybe_clear_locks(True)

        endpoint = self.resolved_endpoint()
        workers = _int_env("HF_MAX_WORKERS", 12, 1, 32)
        etag_timeout = _float_env("HF_HUB_ETAG_TIMEOUT", 30.0)

        slim = not full
        self.warn_locks_if_needed(endpoint)

        common_kw: dict[str, Any] = dict(
            repo_id=self.repo_id,
            local_dir=str(self.local_dir),
            endpoint=endpoint,
            max_workers=workers,
            ignore_patterns=self._ignore_for_slim(slim),
            etag_timeout=etag_timeout,
            token=_hf_token_for_snapshot() if self._token is None else self._token,
            tqdm_class=_LiveHubTqdm,
            force_download=force,
        )

        endpoint_note = endpoint or "默认 hub"
        print(f"Repo: {self.repo_id} → {self.local_dir}")
        print(f"endpoint={endpoint_note} max_workers={workers} slim={slim} force={force}")
        if disk_stats_interval > 0:
            print(f"disk-stats 间隔={disk_stats_interval:g}s（stderr）")
        print(
            "提示: 外层「Fetching N files」按已完成文件数跳；当前大文件的字节进度在下一行（若仍无，确认 stderr 在面板里可见）。"
            " 大文件未写完时外层可能停在同一百分比；`du -sh` 或加 --disk-stats 可看是否在涨。\n"
        )

        try:
            if dry_run:
                infos = snapshot_download(**common_kw, dry_run=True)
                n = len(infos)
                to_pull = [i for i in infos if i.will_download]
                n_cached = sum(1 for i in infos if i.is_cached)
                pull_bytes = sum(i.file_size for i in to_pull)
                print(
                    f"dry-run: 共 {n} 项，仍将下载 {len(to_pull)} 个文件 "
                    f"≈ {pull_bytes / (1024**3):.3f} GiB；声称已缓存 {n_cached} 项。"
                )
                for i in sorted(to_pull, key=lambda x: -x.file_size)[:15]:
                    print(f"  - {i.filename}\t{i.file_size / (1024**2):.1f} MiB")
                if len(to_pull) > 15:
                    print(f"  … 另有 {len(to_pull) - 15} 个文件")
                return self.local_dir

            bench_t = time.monotonic()
            bench_b = _tree_bytes(self.local_dir)
            with _disk_stats_loop(self.local_dir, disk_stats_interval):
                snapshot_download(**common_kw)
            if disk_stats_interval > 0:
                b_end = _tree_bytes(self.local_dir)
                dt = max(time.monotonic() - bench_t, 1e-6)
                avg_mib = max(b_end - bench_b, 0) / dt / (1024 * 1024)
                _disk_log(
                    f"[disk-stats] 收尾 累计≈{b_end / (1024**3):.3f} GiB  "
                    f"本进程全程均≈{avg_mib:.2f} MiB/s"
                )
        except KeyboardInterrupt:
            _disk_log("[disk-stats] 已 Ctrl+C；同一目录重跑一般会续传。")
            raise SystemExit(130) from None

        print(f"\n✅ 完成: {self.local_dir}")
        return self.local_dir
