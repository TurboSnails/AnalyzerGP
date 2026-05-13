"""
Qwen2.5-1.5B-Instruct 下载脚本（Query Rewriter 用）
=====================================================
复用 HfSnapshotDownloader 把 Qwen2.5-1.5B-Instruct 下载到本地
models/qwen2.5-1.5b-instruct/，支持镜像切换、续传、重试。

运行（仓库根 fenxiCB）：
  uv run python ai_app1/test/download_qwen_rewriter.py              # 默认下载 + demo
  uv run python ai_app1/test/download_qwen_rewriter.py --dry-run    # 只列出待下载文件
  uv run python ai_app1/test/download_qwen_rewriter.py --skip-download  # 已下好，直接跑 demo
  uv run python ai_app1/test/download_qwen_rewriter.py --no-mirror  # 直连 Hugging Face

环境变量（与其他下载脚本共享）：
  HF_TOKEN / HF_ENDPOINT / HF_MAX_WORKERS / HF_HUB_ETAG_TIMEOUT
  HF_DISK_STATS_INTERVAL / HF_DOWNLOAD_UNLOCK

模型大小：~3.1 GB（safetensors，slim 模式过滤 .bin / onnx）
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

from hf_snapshot_downloader import DEFAULT_CN_MIRROR, HfSnapshotDownloader  # noqa: E402


DEFAULT_REPO = os.getenv("QWEN_REWRITER_REPO_ID", "Qwen/Qwen2.5-1.5B-Instruct")

# Qwen2.5 Hub 仓库同时包含 pytorch .bin 分片和 safetensors 分片；
# transformers 优先加载 safetensors，slim 模式只下 safetensors，节省约一半流量。
QWEN_SLIM_IGNORE: tuple[str, ...] = (
    "*.bin",
    "*.bin.index.json",
    "onnx/**",
    "*.onnx",
    "*.onnx_data",
    "openvino/**",
    "tf_model.h5",
    "flax_model.msgpack",
    "*.png",
    "*.jpg",
    "*.gif",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Qwen2.5-1.5B-Instruct and run a query-rewrite demo",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"HF repo id（默认 {DEFAULT_REPO}）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="覆盖保存目录（默认: models/qwen2.5-1.5b-instruct）",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="不过滤 .bin/onnx（默认 slim，只留 safetensors）",
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
    parser.add_argument("--dry-run", action="store_true", help="只列出待下载文件，不下载")
    parser.add_argument("--unlock", action="store_true", help="清理 huggingface *.lock")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="直连 Hugging Face（默认官方 Hub；要镜像设 HF_ENDPOINT）",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过下载，直接用既有本地目录跑 demo",
    )
    return parser.parse_args()


def _resolve_endpoint(no_mirror: bool) -> str | None:
    if no_mirror:
        os.environ.pop("HF_ENDPOINT", None)
        return None
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
    name = exc.__class__.__name__
    # 以下错误表示仓库不存在、文件缺失或 Hub 完全不可达，重试通常无效
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
            if not _is_transient_network_error(exc):
                # 非网络类错误直接抛出，并附带诊断提示
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


def _ensure_model(args: argparse.Namespace) -> Path:
    default_models_parent = _PROJECT_ROOT / "models"
    subdir = args.repo.split("/")[-1].lower()
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
        slim_ignore=QWEN_SLIM_IGNORE,
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
    """加载模型并跑一条 Android 查询扩写 demo，验证推理链路正常。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nModel : {model_dir}")
    print("加载 tokenizer / model（首次约需 10-30s）…")

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"device={device}, dtype={dtype}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype).to(device)
    print(f"加载完成，耗时 {time.perf_counter() - t0:.1f}s")

    # ── 模拟 query_rewriter 的 prompt ──────────────────────────────────────────
    system_prompt = (
        "你是 Android 开发知识库的检索优化专家。\n"
        "任务：根据当前问题，生成 3~5 条适合向量检索的 query。\n"
        "输出格式（严格遵守）：只输出一个 JSON 数组，每个元素必须是字符串，不加任何解释。\n"
        '正确示例：["Handler 内存泄漏怎么解决", "Android Handler memory leak", "WeakReference fix"]\n'
        '错误示例（禁止）：[["query", "tag"], ["query2", "tag2"]]'
    )
    test_cases = [
        "Handler 内存泄漏怎么解决",
        "页面卡死了",
    ]

    for query in test_cases:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"【当前问题】\n{query}"},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)

        t1 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        elapsed = time.perf_counter() - t1

        new_ids = output_ids[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        print(f"\nQuery : {query}")
        print(f"Output: {response}")
        print(f"耗时  : {elapsed*1000:.0f}ms")

    print("\n✅ demo 完成 — 若输出为 JSON 数组即说明模型工作正常")


def main() -> None:
    args = _parse_args()
    model_dir = _ensure_model(args)
    if args.dry_run:
        print("dry-run 完成；跳过 demo 推理。")
        return
    _run_demo(model_dir)


if __name__ == "__main__":
    main()
