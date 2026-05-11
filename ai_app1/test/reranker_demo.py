"""
CrossEncoder Reranker Demo
==========================
流程：
  1. 复用 HfSnapshotDownloader 把 BGE Reranker 下载到本地 models/<name>/
     （含镜像切换、tqdm 进度、磁盘统计、续传锁清理 —— 与 download_bge_m3.py 同款交互）
  2. 用本地路径加载 BgeRerankerService，对 (query, doc) 做语义重排

运行（仓库根 fenxiCB）：
  uv run python ai_app1/test/reranker_demo.py                       # 默认 bge-reranker-base
  uv run python ai_app1/test/reranker_demo.py --repo BAAI/bge-reranker-v2-m3
  uv run python ai_app1/test/reranker_demo.py --disk-stats --unlock
  uv run python ai_app1/test/reranker_demo.py --dry-run             # 只列出待下载文件
  uv run python ai_app1/test/reranker_demo.py --skip-download       # 已下好，直接跑 demo

环境变量（与 download_bge_m3.py 共享）：
  HF_TOKEN / HF_ENDPOINT / HF_MAX_WORKERS / HF_HUB_ETAG_TIMEOUT
  HF_DISK_STATS_INTERVAL / HF_DOWNLOAD_UNLOCK / HF_IMPLICIT_TOKEN
  RERANKER_REPO_ID      覆盖默认 repo

效果对比：
  - Embedding："哪些文档像 query"
  - CrossEncoder："哪个文档最相关"
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

# override=True：IDE/Shell 已存在的 HF_TOKEN 会挡住 ai_app1/.env 里的 HF_TOKEN=false
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from hf_snapshot_downloader import DEFAULT_CN_MIRROR, HfSnapshotDownloader  # noqa: E402

from ai_app1.service.reranker import BgeRerankerService  # noqa: E402


DEFAULT_REPO = os.getenv("RERANKER_REPO_ID", "BAAI/bge-reranker-base")

# Reranker 仓库常同时塞 pytorch_model.bin / model.safetensors / onnx/，三份等价权重 ≈ 3×。
# CrossEncoder/transformers 会优先用 safetensors，所以默认 slim 把 .bin 与 onnx 一并裁掉。
RERANKER_SLIM_IGNORE: tuple[str, ...] = (
    "onnx/**",
    "*.onnx",
    "*.onnx_data",
    "openvino/**",
    "*.bin",
    "tf_model.h5",
    "flax_model.msgpack",
    "imgs/**",
    "*.png",
    "*.jpg",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a BGE reranker and run a CrossEncoder demo",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"HF repo id（默认 {DEFAULT_REPO}；也可用 BAAI/bge-reranker-v2-m3 等）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="覆盖保存目录（默认: ai_app1/models/<repo basename>）",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="不过滤 onnx/.bin/配图（默认 slim，只留 safetensors，可节省 ~2/3 流量）",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="网络中断自动重试次数（默认读 HF_DOWNLOAD_RETRIES，最终 fallback=5）",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=3.0,
        help="重试前等待秒数（指数退避基数，默认 3.0）",
    )
    parser.add_argument("--disk-stats", action="store_true")
    parser.add_argument(
        "--disk-stats-interval",
        type=float,
        default=None,
        metavar="SEC",
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出待下载文件，不真正下载、不跑 demo")
    parser.add_argument("--unlock", action="store_true", help="清理 huggingface *.lock（卡住 0% 时用）")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="不走镜像，直连 Hugging Face（仅当未设置环境变量 HF_ENDPOINT 时有效）",
    )
    parser.add_argument(
        "--use-mirror",
        action="store_true",
        help=f"显式使用镜像（默认已启用；未设 HF_ENDPOINT 时同 {DEFAULT_CN_MIRROR}）",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过下载，直接用既有本地目录跑 demo",
    )
    return parser.parse_args()


def _resolve_endpoint(no_mirror: bool) -> str | None:
    # 大陆直连 hub 常极慢或假死；未显式 HF_ENDPOINT 时默认走镜像
    if not os.getenv("HF_ENDPOINT"):
        if not no_mirror:
            os.environ["HF_ENDPOINT"] = DEFAULT_CN_MIRROR
    return os.getenv("HF_ENDPOINT") or None


def _resolve_stats_interval(cli_val: float | None, want_stats: bool) -> float:
    if cli_val is not None:
        return cli_val
    env = os.getenv("HF_DISK_STATS_INTERVAL")
    if env is not None:
        try:
            return float(env)
        except ValueError:
            pass
    return 8.0 if want_stats else 0.0


def _resolve_retries(cli_val: int | None) -> int:
    if cli_val is not None:
        return max(0, cli_val)
    env = os.getenv("HF_DOWNLOAD_RETRIES")
    if env is not None:
        try:
            return max(0, int(env.strip()))
        except ValueError:
            pass
    return 5


def _is_transient_network_error(exc: BaseException) -> bool:
    """镜像断流 / 超时 / 临时 5xx 这类「重跑就能续传」的错误"""
    name = exc.__class__.__name__
    if name in {
        "RemoteProtocolError",
        "ReadTimeout",
        "ConnectError",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadError",
        "WriteError",
        "IncompleteRead",
        "ChunkedEncodingError",
        "ProtocolError",
        "HfHubHTTPError",
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
    retries: int,
    retry_wait: float,
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
            if attempt >= retries or not _is_transient_network_error(exc):
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
            # 失败后必须关 force（否则会清掉 .incomplete 续传文件）；同时 unlock 清掉 .lock
            force_now = False
            unlock_now = True


def _ensure_model(args: argparse.Namespace) -> Path:
    # 与 download_bge_m3.py 默认输出根一致：ai_app1/models/
    default_models_parent = Path(__file__).resolve().parent.parent / "models"
    subdir = args.repo.split("/")[-1]
    local_dir = (args.output_dir or default_models_parent / subdir).expanduser().resolve()

    if args.skip_download:
        if not local_dir.is_dir():
            raise SystemExit(f"--skip-download 但目录不存在: {local_dir}")
        print(f"[skip-download] 使用本地模型: {local_dir}")
        return local_dir

    endpoint = _resolve_endpoint(args.no_mirror)
    dl = HfSnapshotDownloader(
        args.repo,
        local_dir,
        slim_ignore=RERANKER_SLIM_IGNORE,
        endpoint=endpoint,
        token=False,
    )
    stats_interval = _resolve_stats_interval(args.disk_stats_interval, args.disk_stats)
    _download_with_retries(
        dl,
        full=args.full,
        dry_run=args.dry_run,
        force=args.force,
        unlock=args.unlock,
        disk_stats_interval=stats_interval,
        retries=_resolve_retries(args.retries),
        retry_wait=args.retry_wait,
    )
    return local_dir


def _run_demo(model_dir: Path) -> None:
    query = "苹果价格"
    docs = [
        "iPhone 很贵，最新款 Pro Max 售价超过一万。",
        "水果苹果今年涨价，产区受霜冻影响产量下降。",
        "苹果公司发布了新款 MacBook，性能提升明显。",
    ]

    print(f"\nModel : {model_dir}")
    print(f"Query : {query}")
    print(f"Docs  : {len(docs)} 条")
    print("-" * 60)

    service = BgeRerankerService(model_name_or_path=str(model_dir))
    pairs = [[query, doc] for doc in docs]
    scores = service.predict(pairs, batch_size=8)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    print("\nCrossEncoder 重排结果：")
    for rank, (doc, score) in enumerate(ranked, 1):
        print(f"  Rank {rank} | score={score:.4f} | {doc}")

    print("\n✅ 预期：'水果苹果今年涨价...' 应该排在最前面（真正语义相关）")


def main() -> None:
    args = _parse_args()
    model_dir = _ensure_model(args)
    if args.dry_run:
        print("dry-run 完成；跳过 demo 推理。")
        return
    _run_demo(model_dir)


if __name__ == "__main__":
    main()
