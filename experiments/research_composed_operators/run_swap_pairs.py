"""Run paired swap ablations and report deltas at matched evaluation steps."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


HERE = Path(__file__).resolve().parent
KINDS = ("two", "two_no_swap")
DATA_HASHES = {
    "train.npy": "1960dfe9dedd11709434bcc99c076fef300df32ff1fd15c4a3561e56047865e6",
    "val.npy": "7380e73206904faa0c9fb8f242bb9b298c770c7445c0fd4593dc5fb679c66a9b",
}


def run_dir(root, kind, seed):
    return root / f"v2_swap_ablation_{kind}_{seed}"


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def completed_metrics(path, expected_steps):
    rows = read_jsonl(path)
    if len(rows) != expected_steps or any(
        row["step"] != index for index, row in enumerate(rows, start=1)
    ) or "val_nll" not in rows[-1]:
        raise ValueError(f"{path}: expected {expected_steps} consecutive steps and final validation")
    return rows


def launch(args, hashes, kind, seed):
    out = run_dir(args.out, kind, seed)
    manifest = out / "run.json"
    train = [
        "--kind", kind, "--seed", str(seed), "--data", str(args.data.resolve()),
        "--out", str(out.resolve()), "--steps", str(args.steps), "--warmup", str(args.warmup),
        "--eval-every", str(args.eval_every), "--batch", str(args.batch), "--accum", str(args.accum),
        "--eval-batches", str(args.eval_batches), "--device", args.device,
    ]
    precision = [] if args.fp32 else ["--bf16"]
    train.extend(precision)
    if args.tiny:
        train.append("--tiny")
    decode = [
        "--checkpoint", str(out / "latest.pt"), "--prompts", str(args.data / "val.npy"),
        "--prompt-len", str(args.prompt_len), "--new-tokens", str(args.new_tokens),
        "--requests", str(args.requests), "--rounds", str(args.rounds), "--device", args.device,
        *precision,
    ]
    settings = {"train": train, "decode": decode, "data_sha256": hashes}
    if out.exists():
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != settings:
            raise ValueError(f"{out}: existing run has different settings or no manifest")
        completed_metrics(out / "metrics.jsonl", args.steps)
        if not (out / "latest.pt").is_file():
            raise ValueError(f"{out}: completed log has no checkpoint")
    else:
        out.mkdir(parents=True)
        manifest.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        subprocess.run(
            [sys.executable, str(HERE / "train_v2.py"), *train], stdout=sys.stderr, check=True
        )
        completed_metrics(out / "metrics.jsonl", args.steps)

    destination = out / "decode.jsonl"
    if not destination.exists():
        temporary = destination.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            subprocess.run(
                [sys.executable, str(HERE / "decode_v2.py"), *decode], stdout=stream, check=True
            )
        temporary.replace(destination)


def summarize(root, seeds, steps, rounds):
    pairs = []
    for seed in seeds:
        curves, decode = {}, {}
        decode_setup = set()
        for kind in KINDS:
            out = run_dir(root, kind, seed)
            rows = completed_metrics(out / "metrics.jsonl", steps)
            curves[kind] = {
                (row["step"], row["target_tokens"]): row["val_nll"]
                for row in rows if "val_nll" in row
            }
            timings = read_jsonl(out / "decode.jsonl")
            if [row["round"] for row in timings] != list(range(1, rounds + 1)) or any(
                row["kind"] != kind for row in timings
            ):
                raise ValueError(f"{out}: mismatched decode variants or rounds")
            decode_setup.update(
                (row["device"], row["dtype"], row["requests"], row["prompt_tokens"], row["decode_tokens"])
                for row in timings
            )
            decode[kind] = {
                key: statistics.median(row[key] for row in timings)
                for key in ("p50_ms", "p95_ms")
            }
        if curves["two"].keys() != curves["two_no_swap"].keys():
            raise ValueError(f"Seed {seed}: evaluations are on different step/token budgets")
        if len(decode_setup) != 1:
            raise ValueError(f"Seed {seed}: paired decode settings differ")
        evaluations = []
        for (step, tokens), swap in curves["two"].items():
            no_swap = curves["two_no_swap"][step, tokens]
            if not all(math.isfinite(value) for value in (swap, no_swap)):
                raise ValueError(f"Seed {seed}, step {step}: nonfinite NLL")
            evaluations.append(dict(step=step, target_tokens=tokens, two=swap, two_no_swap=no_swap,
                                    delta_nll=no_swap - swap))
        pairs.append(dict(seed=seed, evaluations=evaluations, decode=decode))
    if len({pair["evaluations"][-1]["target_tokens"] for pair in pairs}) != 1:
        raise ValueError("Final token budgets differ across seeds")
    deltas = [pair["evaluations"][-1]["delta_nll"] for pair in pairs]
    return dict(pairs=pairs, mean_delta_nll=statistics.mean(deltas),
                min_delta_nll=min(deltas), max_delta_nll=max(deltas))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261124, 20261125])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("logs"))
    parser.add_argument("--steps", type=int, default=1529)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Seeds must be distinct and nonnegative")
    if min(args.steps, args.rounds, args.eval_every, args.batch, args.accum, args.eval_batches,
           args.prompt_len, args.requests) < 1 or args.new_tokens < 2 or args.warmup < 0:
        parser.error("Invalid training or decode sizes")
    if not args.summary_only:
        hashes = {}
        for name, expected in DATA_HASHES.items():
            digest = hashlib.sha256()
            with (args.data / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[name] = digest.hexdigest()
            if not args.tiny and hashes[name] != expected:
                parser.error(f"{args.data / name}: SHA256 differs from REPORT_V2.md")
        for seed in args.seeds:
            for kind in KINDS:
                launch(args, hashes, kind, seed)
    print(json.dumps(summarize(args.out, args.seeds, args.steps, args.rounds), indent=2))


if __name__ == "__main__":
    main()
