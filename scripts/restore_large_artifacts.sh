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
# Raw adapter weights exceed GitHub's 100MB limit. Store gzip split parts
# and restore each final experiment adapter on demand.
restore_split_adapter() {  # restore_split_adapter <label> <directory> <sha256>
    local label=$1 dir=$2 sha=$3
    local out="$dir/adapter_model.safetensors"
    echo "$label"
    if have "$out" "$sha"; then
        skip "$out"
    elif [ "$CHECK_ONLY" = 1 ]; then
        echo "  (check) 缺失或不匹配：$out"
    else
        work "$out"
        cat "$dir"/adapter_model.safetensors.gz.part[0-9][0-9] | gunzip -c > "$out"
        verify "$out" "$sha"
        ok "$out"
    fi
}

restore_split_adapter "[1/7] SFT v1 LoRA adapter (3 分卷 → 133MB)" \
    shopsimulator/shopagent-rl/outputs/sft/v1/model/training_output/lora_adapter \
    889f240d29d2ce048b7f020882b8b2324a608a64e2561cff40ffe2be0fb59f6c
restore_split_adapter "[2/7] SFT v2_paired LoRA adapter" \
    shopsimulator/shopagent-rl/outputs/sft/v2_paired/model/training_output/lora_adapter \
    4251d6b1997f286fb5abd0c70430abb30472e90440329d0300e7dff0c437a232
restore_split_adapter "[3/7] SFT v3_certified LoRA adapter" \
    shopsimulator/shopagent-rl/outputs/sft/v3_certified/model/training_output/lora_adapter \
    d212c6b56b8abf55f94fb634bd5f12ca3743829c6b1e6876d0dc407d3ceb6314
restore_split_adapter "[4/7] SFT v4_certified_corrective LoRA adapter" \
    shopsimulator/shopagent-rl/outputs/sft/v4_certified_corrective/model/training_output/lora_adapter \
    def8a0b0b8426b0e2046569a4d4916baec2c0bab405f571bc43ac686f18c8be1
restore_split_adapter "[5/7] SFT v5_certified_explicit_clean LoRA adapter" \
    shopsimulator/shopagent-rl/outputs/sft/v5_certified_explicit_clean/model/training_output/lora_adapter \
    ac2eab3d67da07f494299e63799303aa95047d93b4c27a6d2731c2f159ac41aa

# --------------------------------------------------------------- 6. GRPO env16 step 200 LoRA
GRPO200_DIR=shopsimulator/shopagent-rl/outputs/grpo/v1/model/checkpoint_step_200/artifact/adapter
GRPO200_OUT="$GRPO200_DIR/adapter_model.safetensors"
GRPO200_SHA=0d6162d596b92a307a5811f90830e3905961da950ab8308c7838c24bfc540d53

echo "[6/7] GRPO env16 global_step_200 LoRA adapter (67MB)"
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

# ------------------------------------------------------- 7. 环境数据与检索索引
# items_eval_train.json：140MB 任务/商品库，被 shop_env/.gitignore 的 data/*.json 排除。
# search_engine/：pyserini Lucene BM25 索引（20MB），env 起不来就是缺它。
ENV_DIR=shopsimulator/ShopSimulator/shop_env
ITEMS_OUT="$ENV_DIR/data/items_eval_train.json"

echo "[7/7] ShopSimulator env 数据 + BM25 索引"
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
