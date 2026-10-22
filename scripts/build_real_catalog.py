from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from app.data.amazon_reviews_adapter import adapt_amazon_product


REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
PARQUET_DIRECTORY = "raw_meta_Electronics"
NUM_SHARDS = 10


def get_dataset_revision() -> str:
    api = HfApi()
    info = api.dataset_info(REPO_ID)
    return info.sha or "main"


def get_shard_filename(shard_index: int) -> str:
    return (
        f"{PARQUET_DIRECTORY}/"
        f"full-{shard_index:05d}-of-{NUM_SHARDS:05d}.parquet"
    )


def iter_amazon_rows(*, revision: str, seed: int, max_scan: int):
    """
    不再使用 datasets streaming shuffle。

    之前远程 streaming 在 Python 退出阶段出现过
    Bad file descriptor / PyGILState_Release 崩溃。

    现在改为：
    Hugging Face Hub 按需下载单个 parquet shard到本地缓存
    -> PyArrow 本地 batch 读取
    -> 达到 max_scan 即停止。

    不会下载完整 Electronics 全量数据。
    """
    rng = random.Random(seed)
    shard_order = list(range(NUM_SHARDS))
    rng.shuffle(shard_order)

    scanned = 0

    for shard_index in shard_order:
        filename = get_shard_filename(shard_index)

        print("Downloading/using shard:", filename)

        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )

        parquet_file = pq.ParquetFile(local_path)

        for batch in parquet_file.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                yield row
                scanned += 1

                if scanned >= max_scan:
                    return


def stratified_sample(products, *, target: int, seed: int):
    """在真实商品池上做尽量均衡的类别抽样。"""
    rng = random.Random(seed)
    groups: dict[str, list] = {}

    for product in products:
        groups.setdefault(product.category, []).append(product)

    for items in groups.values():
        rng.shuffle(items)

    category_names = sorted(groups.keys())
    if not category_names:
        return []

    base_quota = max(1, target // len(category_names))

    selected = []
    leftovers = []

    for category in category_names:
        items = groups[category]
        take_count = min(base_quota, len(items))
        selected.extend(items[:take_count])
        leftovers.extend(items[take_count:])

    if len(selected) < target:
        rng.shuffle(leftovers)
        remaining = target - len(selected)
        selected.extend(leftovers[:remaining])

    rng.shuffle(selected)
    return selected[:target]


def build_catalog(
    *,
    target: int,
    pool_size: int,
    max_scan: int,
    min_rating_count: int,
    seed: int,
):
    """
    在 max_scan 范围内对有效商品做 reservoir sampling，
    最后再做类别均衡抽样。
    """
    revision = get_dataset_revision()

    print("=" * 70)
    print("AMAZON REVIEWS 2023 CATALOG BUILDER")
    print("=" * 70)
    print(f"Repository: {REPO_ID}")
    print(f"Revision: {revision}")
    print(f"Target: {target:,}")
    print(f"Reservoir size: {pool_size:,}")
    print(f"Max scan: {max_scan:,}")
    print()

    rng = random.Random(seed)
    reservoir = []
    seen_ids: set[str] = set()

    scanned = 0
    valid_seen = 0

    for raw in iter_amazon_rows(
        revision=revision,
        seed=seed,
        max_scan=max_scan,
    ):
        scanned += 1

        if scanned % 5000 == 0:
            print(
                "scanned="
                f"{scanned:,} "
                "valid_seen="
                f"{valid_seen:,} "
                "reservoir="
                f"{len(reservoir):,}"
            )

        product = adapt_amazon_product(
            raw,
            min_rating_count=min_rating_count,
        )

        if product is None:
            continue

        if product.id in seen_ids:
            continue

        seen_ids.add(product.id)
        valid_seen += 1

        if len(reservoir) < pool_size:
            reservoir.append(product)
        else:
            replacement_index = rng.randint(1, valid_seen)
            if replacement_index <= pool_size:
                reservoir[replacement_index - 1] = product

    print()
    print(f"Scanned rows: {scanned:,}")
    print(f"Valid unique products seen: {valid_seen:,}")
    print(f"Reservoir products: {len(reservoir):,}")

    if len(reservoir) < target:
        raise RuntimeError(
            "Not enough valid products. "
            "Increase --max-scan, reduce --target, "
            "or relax adapter filtering."
        )

    selected = stratified_sample(
        reservoir,
        target=target,
        seed=seed,
    )

    return selected, revision, scanned, valid_seen


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=6000)
    parser.add_argument("--max-scan", type=int, default=100000)
    parser.add_argument("--min-rating-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/products_real.json")
    parser.add_argument("--manifest", default="data/catalog_manifest.json")

    args = parser.parse_args()

    if args.target <= 0:
        raise ValueError("--target must be positive")

    if args.pool_size < args.target:
        args.pool_size = args.target

    products, revision, scanned, valid_seen = build_catalog(
        target=args.target,
        pool_size=args.pool_size,
        max_scan=args.max_scan,
        min_rating_count=args.min_rating_count,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            [product.model_dump() for product in products],
            file,
            ensure_ascii=False,
            indent=2,
        )

    category_counter = Counter(product.category for product in products)

    manifest = {
        "source": "Amazon Reviews 2023",
        "repository": REPO_ID,
        "source_config": PARQUET_DIRECTORY,
        "revision": revision,
        "sampling": {
            "seed": args.seed,
            "target": args.target,
            "pool_size": args.pool_size,
            "max_scan": args.max_scan,
            "min_rating_count": args.min_rating_count,
            "rows_scanned": scanned,
            "valid_unique_products_seen": valid_seen,
            "method": "local-shard reservoir sampling + stratified sample",
        },
        "product_count": len(products),
        "category_distribution": dict(sorted(category_counter.items())),
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("FINAL CATALOG")
    print("=" * 70)
    print(f"Products: {len(products):,}")

    for category, count in sorted(category_counter.items()):
        print(f"{category:<12} {count:>5}")

    print()
    print(f"Catalog: {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
