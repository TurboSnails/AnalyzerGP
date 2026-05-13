"""
下载 Hugging Face 上的句向量模型到本地（默认推荐轻量 BGE，弱网友好）。

预设（--preset）：
  bge-base-zh   BAAI/bge-base-zh-v1.5（默认，体积较小，中文常用）
  bge-small-zh  BAAI/bge-small-zh-v1.5（更小）
  bge-m3        BAAI/bge-m3（大、全功能，跳过 onnx/ 配图）
  minilm-m      paraphrase-multilingual-MiniLM-L12-v2（多语轻量）

环境变量：
  HF_TOKEN、HF_ENDPOINT、HF_MAX_WORKERS、HF_HUB_ETAG_TIMEOUT、HF_DISK_STATS_INTERVAL、HF_DOWNLOAD_UNLOCK
  未设 HF_TOKEN 时脚本默认匿名（不用本机过期 token）。要隐式用已 login token：HF_IMPLICIT_TOKEN=1
  HF_TOKEN=具体值 / false / 空：见 hf_snapshot_downloader。

默认（未设置 HF_ENDPOINT）使用大陆镜像 hf-mirror；要直连官方 Hub：加 --no-mirror。

运行（仓库根 fenxiCB）：
  uv run python ai_app1/scripts/download_bge_m3.py
  uv run python ai_app1/scripts/download_bge_m3.py --no-mirror
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# 清理空字符串 HF_ENDPOINT，避免 huggingface_hub 模块导入时把它当作合法 endpoint
if os.getenv("HF_ENDPOINT") == "":
    del os.environ["HF_ENDPOINT"]

from hf_snapshot_downloader import DEFAULT_CN_MIRROR, HfSnapshotDownloader, PRESETS  # noqa: E402


def _resolve_endpoint(no_mirror: bool) -> str | None:
    if no_mirror:
        os.environ.pop("HF_ENDPOINT", None)
        return None
    return os.getenv("HF_ENDPOINT") or None


def _is_transient_network_error(exc: BaseException) -> bool:
    name = exc.__class__.__name__
    if name in {
        "LocalEntryNotFoundError",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "EntryNotFoundError",
    }:
        return False
    if name in {
        "RemoteProtocolError", "ReadTimeout", "ConnectError", "ConnectTimeout",
        "PoolTimeout", "ReadError", "WriteError", "IncompleteRead",
        "ChunkedEncodingError", "ProtocolError", "HfHubHTTPError",
    }:
        return True
    msg = str(exc).lower()
    if "peer closed connection" in msg or "incomplete" in msg or "timed out" in msg:
        return True
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def _download_with_retries(
    dl: HfSnapshotDownloader,
    *,
    full: bool,
    dry_run: bool,
    force: bool,
    unlock: bool,
    disk_stats_interval: float,
    retries: int = 3,
    retry_wait: float = 3.0,
) -> None:
    attempt = 0
    force_now = force
    unlock_now = unlock
    while True:
        try:
            dl.download(
                full=full,
                dry_run=dry_run,
                force=force_now,
                unlock=unlock_now,
                disk_stats_interval=disk_stats_interval,
            )
            return
        except SystemExit:
            raise
        except BaseException as exc:
            if not _is_transient_network_error(exc):
                print(
                    f"\n[error] 不可重试的错误 ({exc.__class__.__name__}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if "LocalEntryNotFoundError" in exc.__class__.__name__:
                    print(
                        "[hint] 该错误通常表示无法连接到 Hugging Face Hub 或镜像站，"
                        "或仓库不存在。请检查网络、代理设置，或尝试 --no-mirror。",
                        file=sys.stderr,
                        flush=True,
                    )
                raise
            if attempt >= retries:
                raise
            attempt += 1
            wait = retry_wait * (2 ** (attempt - 1))
            print(
                f"\n[retry {attempt}/{retries}] 网络中断 ({exc.__class__.__name__}): {exc}\n"
                f"[retry] {wait:.1f}s 后续传重试……",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
            force_now = False
            unlock_now = True


def main() -> None:
    default_models_parent = _PROJECT_ROOT / "models"

    parser = argparse.ArgumentParser(
        description="Download a sentence-transformers model from Hugging Face Hub",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="bge-base-zh",
        help="模型预设（默认 bge-base-zh，下载更快；要 m3 用 bge-m3）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="覆盖保存目录（默认: 仓库根目录 models/<预设子目录>）",
    )
    parser.add_argument("--full", action="store_true", help="不过滤 onnx/配图（仅 bge-m3 有意义）")
    parser.add_argument("--disk-stats", action="store_true")
    parser.add_argument(
        "--disk-stats-interval",
        type=float,
        default=None,
        metavar="SEC",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--unlock", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="直连 Hugging Face（默认走镜像）",
    )

    args = parser.parse_args()

    endpoint = _resolve_endpoint(args.no_mirror)

    if args.output_dir is not None:
        out = args.output_dir.expanduser().resolve()
        pmeta = PRESETS[args.preset]
        dl = HfSnapshotDownloader(
            pmeta.repo_id,
            out,
            slim_ignore=pmeta.slim_ignore,
            endpoint=endpoint,
        )
    else:
        dl = HfSnapshotDownloader.from_preset(
            args.preset,
            default_models_parent,
            endpoint=endpoint,
        )

    stats_interval = args.disk_stats_interval
    if stats_interval is None and os.getenv("HF_DISK_STATS_INTERVAL") is not None:
        stats_interval = float(os.environ["HF_DISK_STATS_INTERVAL"])
    elif stats_interval is None and args.disk_stats:
        stats_interval = 8.0
    elif stats_interval is None:
        stats_interval = 0.0

    _download_with_retries(
        dl,
        full=args.full,
        dry_run=args.dry_run,
        force=args.force,
        unlock=args.unlock,
        disk_stats_interval=stats_interval,
    )


if __name__ == "__main__":
    main()
