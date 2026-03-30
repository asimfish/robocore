#!/usr/bin/env bash
# IsaacLab 环境管理脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup     初始化 IsaacLab submodule 并安装依赖"
    echo "  update    更新 IsaacLab submodule 到最新 tag"
    echo "  check     运行 IsaacSim 兼容性检查"
    echo "  demo      运行 IsaacLab demo (quadrupeds)"
    echo ""
}

cmd_setup() {
    echo "==> 初始化 IsaacLab submodule..."
    cd "$PROJECT_ROOT"
    git submodule update --init --recursive

    echo "==> 安装 isaaclab 依赖 (uv sync --extra isaaclab)..."
    uv sync --extra isaaclab
    uv pip install -e .

    echo "==> 安装完成！运行 '$0 check' 验证兼容性。"
}

cmd_update() {
    echo "==> 进入 isaaclab submodule..."
    cd "$PROJECT_ROOT/isaaclab" || { echo "Error: isaaclab/ 目录不存在，先运行 '$0 setup'"; exit 1; }

    echo "==> 拉取最新 tags..."
    git fetch --tags

    LATEST_TAG=$(git tag --sort=-v:refname | head -n 1)
    echo "==> 最新 tag: $LATEST_TAG"

    git checkout "$LATEST_TAG"
    cd "$PROJECT_ROOT"

    echo "==> 提交 submodule 更新..."
    git add isaaclab
    git commit -m "chore: update isaaclab submodule to $LATEST_TAG"

    echo "==> 重新安装依赖..."
    uv sync --extra isaaclab

    echo "==> 完成！IsaacLab 已更新到 $LATEST_TAG"
}

cmd_check() {
    echo "==> 运行 IsaacSim 兼容性检查..."
    python -c "import isaacsim; print(f'IsaacSim version: {isaacsim.__version__}')" 2>/dev/null || \
        isaacsim isaacsim.exp.compatibility_check
}

cmd_demo() {
    local DEMO="${1:-quadrupeds}"
    echo "==> 运行 IsaacLab demo: $DEMO"
    cd "$PROJECT_ROOT"
    python "isaaclab/scripts/demos/${DEMO}.py"
}

# Main dispatcher
case "${1:-}" in
    setup)  cmd_setup ;;
    update) cmd_update ;;
    check)  cmd_check ;;
    demo)   shift; cmd_demo "$@" ;;
    *)      show_help; exit 1 ;;
esac
