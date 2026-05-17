#!/usr/bin/env bash
# 新电脑首次克隆后运行此脚本完成环境初始化
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── 1. 检查 uv ─────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[setup] 未检测到 uv，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# ── 2. 安装依赖（含本地 rag-framework / android-domain）────────
echo "[setup] 正在安装依赖 (uv sync)..."
uv sync

# ── 3. 复制 .env 模板（已存在则跳过）───────────────────────────
for app in ai_app1 ai_app2 ai_app3 ai_app4; do
    example="$ROOT/$app/.env.example"
    target="$ROOT/$app/.env"
    if [ -f "$example" ] && [ ! -f "$target" ]; then
        cp "$example" "$target"
        echo "[setup] 已创建 $app/.env（请填入真实 API Key）"
    elif [ -f "$target" ]; then
        echo "[setup] $app/.env 已存在，跳过"
    fi
done

# ── 4. 创建数据目录（如不存在）─────────────────────────────────
mkdir -p "$ROOT/ai_app1/data"
mkdir -p "$ROOT/models"

echo ""
echo "✓ 环境初始化完成"
echo "  下一步："
echo "    1. 编辑 ai_app1/.env，填入你的 OPENAI_API_KEY"
echo "    2. 运行模型下载脚本（见 PROJECT_LAYOUT.md）"
echo "    3. 运行索引构建脚本（见 PROJECT_LAYOUT.md）"
