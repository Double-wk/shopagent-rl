#!/usr/bin/env bash
# Start the ShopSimulator env service (pack_api.py) under shopenv on :5000.
# Blocks. Other scripts (01_collect.sh …) assume this is running.
#
# Preflight auto-builds the two un-shipped assets the service needs:
#   - data/items_eval_train.json  (decompress from the shipped .gz)
#   - search_engine/indexes/      (pyserini Lucene index, one-time)
#
# JVM env: pyserini -> pyjnius must dlopen libjvm.so. Conda's openjdk 21 lays it out
# at <env>/lib/jvm/lib/server/libjvm.so, which pyjnius does NOT guess, so we set
# JAVA_HOME + JVM_PATH explicitly (needed for BOTH indexing and runtime search).
set -euo pipefail

ENV=/workspace/persistent/miniconda3/envs/shopenv
PY="$ENV/bin/python"
REPO=/workspace/persistent/shopsimulator/ShopSimulator/shop_env   # data/, search_engine/, web_agent_site/, shop_env/
SCRIPTS=/workspace/persistent/shopsimulator/shop_A/scripts

if [ ! -x "$PY" ]; then
  echo "ERROR: shopenv not built. Run: bash $SCRIPTS/setup_shopenv.sh"; exit 1
fi

export JAVA_HOME="$ENV/lib/jvm"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export PATH="$JAVA_HOME/bin:$PATH"

# Preflight: product data
if [ ! -f "$REPO/data/items_eval_train.json" ]; then
  echo "[preflight] decompressing product data..."
  gunzip -c "$REPO/data/fine_items_eval_train_all.json.gz" > "$REPO/data/items_eval_train.json"
fi

# Preflight: Lucene search index
if [ ! -d "$REPO/search_engine/indexes" ]; then
  echo "[preflight] building search docs (pure python)..."
  "$PY" "$SCRIPTS/build_search_docs.py"
  echo "[preflight] building Lucene index (one-time)..."
  "$PY" -m pyserini.index.lucene --collection JsonCollection \
    --input "$REPO/search_engine/index_docs" --index "$REPO/search_engine/indexes" \
    --generator DefaultLuceneDocumentGenerator --threads 8 \
    --storePositions --storeDocvectors --storeRaw
fi

# pack_api.py does `sys.path.append("../")` and imports web_agent_site, so it MUST be
# launched with cwd = the nested shop_env/ dir (then "../" -> REPO, where web_agent_site
# lives; the script dir on sys.path[0] covers `import shop_agent`).
cd "$REPO/shop_env"
echo "[service] starting ShopSimulator on http://127.0.0.1:5000 ..."
exec "$PY" pack_api.py
