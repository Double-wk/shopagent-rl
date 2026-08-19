"""Build pyserini JsonCollection docs from the product data, for Lucene indexing.

Converts the service's data/items_eval_train.json (a list of product dicts) into a
pyserini JsonCollection directory (search_engine/index_docs/docs.jsonl): one JSON
object per line, {"id": <asin>, "contents": <searchable text>}. The Lucene index
itself is then built by pyserini's indexer in the shared shopsim environment,
the service loads at boot via engine.init_search_engine.

This converter is pure Python and runs in any env (no pyserini needed here).
Paths target the UPSTREAM service tree (ShopSimulator/shop_env), because that is
where pack_api.py runs and where the code looks for data/ and search_engine/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# shopagent-rl/scripts/build_search_docs.py  ->  parents[2] = .../shopsimulator
SHOP_PARENT = Path(__file__).resolve().parents[2]
SHOP_ENV = SHOP_PARENT / "ShopSimulator" / "shop_env"
DATA_FILE = SHOP_ENV / "data" / "items_eval_train.json"
OUT_DIR = SHOP_ENV / "search_engine" / "index_docs"


def doc_for(p: dict) -> dict | None:
    asin = p.get("asin")
    if not asin or asin == "nan" or len(str(asin)) > 20:
        return None
    parts = [
        str(p.get("title", "")),
        str(p.get("full_description", "")),
        p.get("small_description", "") or "",
        str(p.get("category", "")),
        str(p.get("query", "")),
        str(p.get("shop_name", "")),
    ]
    attrs = p.get("attribute") or []
    if isinstance(attrs, list):
        parts.extend(str(a) for a in attrs)
    contents = " ".join(x for x in parts if x).strip()
    if not contents:
        return None
    return {"id": str(asin), "contents": contents}


def main() -> None:
    if not DATA_FILE.exists():
        sys.exit(f"missing {DATA_FILE} — decompress fine_items_eval_train_all.json.gz first")
    with open(DATA_FILE, encoding="utf-8") as f:
        products = json.load(f)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "docs.jsonl"
    n = 0
    skipped = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for p in products:
            doc = doc_for(p)
            if doc is None:
                skipped += 1
                continue
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} docs (skipped {skipped}) -> {out_file}")
    print(f"now build the index (in shopsim):")
    print(f"  python -m pyserini.index.lucene --collection JsonCollection \\")
    print(f"    --input {OUT_DIR} --index {SHOP_ENV / 'search_engine' / 'indexes'} \\")
    print(f"    --generator DefaultLuceneDocumentGenerator --threads 8 \\")
    print(f"    --storePositions true --storeDocvectors true --storeRaw")


if __name__ == "__main__":
    main()
