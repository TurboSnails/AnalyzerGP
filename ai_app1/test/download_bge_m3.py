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

运行（当前工作目录不同，路径不同 —— 勿在 ai_app1 里再写一层 ai_app1/）：
  仓库根 fenxiCB:     uv run python ai_app1/test/download_bge_m3.py
                      uv run python ai_app1/run_download.py
  已进入 ai_app1:     uv run python test/download_bge_m3.py
                      uv run python run_download.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from hf_snapshot_downloader import DEFAULT_CN_MIRROR, HfSnapshotDownloader, PRESETS

# override=True：否则 IDE/Shell 里已存在的 HF_TOKEN 会挡住 ai_app1/.env 里的 HF_TOKEN=false
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


def main() -> None:
    default_models_parent = Path(__file__).resolve().parent.parent / "models"

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
        help="覆盖保存目录（默认: 脚本同级 models/<预设子目录>）",
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
        help="不走镜像，直连 Hugging Face（仅当未设置环境变量 HF_ENDPOINT 时有效）",
    )
    parser.add_argument(
        "--use-mirror",
        action="store_true",
        help=f"显式使用镜像（默认已启用；未设 HF_ENDPOINT 时同 {DEFAULT_CN_MIRROR}）",
    )

    args = parser.parse_args()

    # 大陆直连 hub 常极慢或假死；未显式 HF_ENDPOINT 时默认走镜像
    if not os.getenv("HF_ENDPOINT"):
        if args.no_mirror:
            pass  # 保持不设 HF_ENDPOINT → 官方 hub
        else:
            os.environ["HF_ENDPOINT"] = DEFAULT_CN_MIRROR

    endpoint = os.getenv("HF_ENDPOINT") or None

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

    dl.download(
        full=args.full,
        dry_run=args.dry_run,
        force=args.force,
        unlock=args.unlock,
        disk_stats_interval=stats_interval,
    )


if __name__ == "__main__":
    main()
