#!/usr/bin/env bash
# 从仓库内的压缩包还原「超过 GitHub 100MB 硬限、只能压缩入库」的复现必需文件。
# clone 之后先跑这个，再跑 shop_A 的训练/评测脚本。
#
#   bash scripts/restore_large_artifacts.sh          # 还原全部
#   bash scripts/restore_large_artifacts.sh --check  # 只校验，不写文件
#
# 每一项都带 sha256，还原后自动比对；已存在且校验通过的直接跳过。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
skip() { printf '  \033[36mskip\033[0m  %s (已存在且校验通过)\n' "$1"; }
work() { printf '  \033[33mwrite\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; exit 1; }

verify() {  # verify <file> <sha256>
    local f=$1 want=$2 got
    got=$(sha256sum "$f" | cut -d' ' -f1)
    [ "$got" = "$want" ] || fail "$f sha256 不匹配 (want $want, got $got)"
}

have() {  # have <file> <sha256> —— 文件已在且校验通过
    [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]
}

# ---------------------------------------------------------------- 1. SFT LoRA
# 133MB > 100MB 硬限 → gzip 后仍 129MB → 再按 48MB 分卷。
SFT_DIR=shopsimulator/shopagent-rl/outputs/sft/v1/model/training_output/lora_adapter
SFT_OUT="$SFT_DIR/adapter_model.safetensors"
SFT_SHA=889f240d29d2ce048b7f020882b8b2324a608a64e2561cff40ffe2be0fb59f6c

echo "[1/3] SFT LoRA adapter (3 分卷 → 133MB)"
if have "$SFT_OUT" "$SFT_SHA"; then
    skip "$SFT_OUT"
elif [ "$CHECK_ONLY" = 1 ]; then
    echo "  (check) 缺失或不匹配：$SFT_OUT"
else
    work "$SFT_OUT"
    cat "$SFT_DIR"/adapter_model.safetensors.gz.part[0-9][0-9] | gunzip -c > "$SFT_OUT"
    verify "$SFT_OUT" "$SFT_SHA"
    ok "$SFT_OUT"
fi

# --------------------------------------------------------------- 2. GRPO env16 step 200 LoRA
GRPO200_DIR=shopsimulator/shopagent-rl/outputs/grpo/v1/model/checkpoint_step_200/artifact/adapter
GRPO200_OUT="$GRPO200_DIR/adapter_model.safetensors"
GRPO200_SHA=0d6162d596b92a307a5811f90830e3905961da950ab8308c7838c24bfc540d53

echo "[2/3] GRPO env16 global_step_200 LoRA adapter (67MB)"
if have "$GRPO200_OUT" "$GRPO200_SHA"; then
    skip "$GRPO200_OUT"
elif [ "$CHECK_ONLY" = 1 ]; then
    echo "  (check) 缺失或不匹配：$GRPO200_OUT"
else
    work "$GRPO200_OUT"
    gunzip -c "$GRPO200_DIR/adapter_model.safetensors.gz" > "$GRPO200_OUT"
    verify "$GRPO200_OUT" "$GRPO200_SHA"
    ok "$GRPO200_OUT"
fi

# ------------------------------------------------------- 3. 环境数据与检索索引
# items_eval_train.json：140MB 任务/商品库，被 shop_env/.gitignore 的 data/*.json 排除。
# search_engine/：pyserini Lucene BM25 索引（20MB），env 起不来就是缺它。
ENV_DIR=shopsimulator/ShopSimulator/shop_env
ITEMS_OUT="$ENV_DIR/data/items_eval_train.json"

echo "[3/3] ShopSimulator env 数据 + BM25 索引"
if [ "$CHECK_ONLY" = 1 ]; then
    [ -f "$ITEMS_OUT" ] || echo "  (check) 缺失：$ITEMS_OUT"
    [ -d "$ENV_DIR/search_engine/indexes" ] || echo "  (check) 缺失：$ENV_DIR/search_engine/"
else
    if [ -f "$ITEMS_OUT" ]; then
        skip "$ITEMS_OUT"
    else
        work "$ITEMS_OUT"
        gunzip -c "$ENV_DIR/data/items_eval_train.json.gz" > "$ITEMS_OUT"
        ok "$ITEMS_OUT"
    fi

    if [ -d "$ENV_DIR/search_engine/indexes" ]; then
        skip "$ENV_DIR/search_engine/"
    else
        work "$ENV_DIR/search_engine/"
        tar xzf "$ENV_DIR/search_engine.tar.gz" -C "$ENV_DIR"
        ok "$ENV_DIR/search_engine/"
    fi
fi

echo
echo "完成。下一步见 shopsimulator/shopagent-rl/README.md（环境启动 → 评测 → 训练）。"
