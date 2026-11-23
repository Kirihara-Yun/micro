"""FineWeb-Edu 子集下载 + GPT-2 预分词 + 文档级 train/val split + packing。

产物（uint32 token ids，memmap 或 .npy）：
- data/train.bin / data/train.npy
- data/val.bin / data/val.npy

用法：
  HF_ENDPOINT=https://hf-mirror.com python data.py --n_shards 2 --target_tokens 60000000
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

CONTEXT = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_shards", type=int, default=2)
    ap.add_argument("--target_tokens", type=int, default=60_000_000,
                    help="目标总 token 数（训练+验证），超出即停")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", type=str, default="sample-10BT")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2")

    train_ids = []
    val_ids = []
    n_train_docs = n_val_docs = 0
    total_tokens = 0

    def flush():
        pass

    for shard in range(args.n_shards):
        fname = f"sample/10BT/000_{shard:05d}.parquet"
        print(f"[shard {shard}] loading {fname} ...", flush=True)
        ds = load_dataset(args.dataset, name=args.subset, split="train",
                          streaming=True, data_files={"train": fname})
        for ex in ds:
            text = ex["text"]
            ids = tok.encode(text, add_special_tokens=False, truncation=False)
            if len(ids) < 8:
                continue
            # 文档级 split：用 id 字段 hash 分桶，确定性且无偏
            doc_id = ex.get("id") or ex.get("url") or text[:64]
            h = int(hashlib.md5(doc_id.encode()).hexdigest(), 16)
            is_train = (h % 100) >= int(args.val_ratio * 100)
            target = train_ids if is_train else val_ids
            target.extend(ids)
            target.append(tok.eos_token_id)
            if is_train:
                n_train_docs += 1
            else:
                n_val_docs += 1
            total_tokens += len(ids) + 1
            if total_tokens >= args.target_tokens:
                break
        if total_tokens >= args.target_tokens:
            break

    # packing 到 context 长度（截断尾部不足一整个 context 的余数）
    def pack(ids):
        n = len(ids)
        usable = (n // CONTEXT) * CONTEXT
        arr = np.array(ids[:usable], dtype=np.uint32).reshape(-1, CONTEXT)
        return arr

    train_arr = pack(train_ids)
    val_arr = pack(val_ids)
    print(f"训练文档 {n_train_docs}, 验证文档 {n_val_docs}")
    print(f"训练序列 {train_arr.shape} ({train_arr.shape[0]*CONTEXT/1e6:.2f}M tokens)")
    print(f"验证序列 {val_arr.shape} ({val_arr.shape[0]*CONTEXT/1e6:.2f}M tokens)")

    np.save(os.path.join(args.out_dir, "train.npy"), train_arr)
    np.save(os.path.join(args.out_dir, "val.npy"), val_arr)
    print(f"saved to {args.out_dir}/train.npy, {args.out_dir}/val.npy")


if __name__ == "__main__":
    main()
