#!/usr/bin/env python3

import argparse
import csv
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


TRACKED_ENV = (
    "LLAMA_EXPERT_S",
    "LLAMA_EXPERT_TMAX",
    "LLAMA_EXPERT_ADAPT",
    "LLAMA_EXPERT_DECAY",
    "LLAMA_EXPERT_RAMPOOL",
    "LLAMA_EXPERT_PREAD",
    "LLAMA_EXPERT_PREDICT",
    "LLAMA_EXPERT_PREFETCH_GB",
    "LLAMA_EXPERT_PREFETCH_THREADS",
    "LLAMA_EXPERT_ASYNC_H2D",
    "LLAMA_EXPERT_STATS",
)


def csv_values(text, cast=int):
    if text is None:
        return []
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def string_values(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def run_capture(command):
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except OSError as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def hardware_metadata():
    meta = {
        "platform": platform.platform(),
        "python": sys.version,
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    nvidia = run_capture([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap,pcie.link.gen.max,pcie.link.width.max",
        "--format=csv,noheader,nounits",
    ])
    if nvidia["returncode"] == 0:
        meta["nvidia_smi"] = nvidia["stdout"].splitlines()
    else:
        meta["nvidia_smi_error"] = nvidia["stderr"]
    return meta


@contextmanager
def preserved_sidecar(model, mode, capture_path):
    sidecar = Path(str(model) + ".tier")
    backup = sidecar.with_name(sidecar.name + f".bench-backup-{uuid.uuid4().hex}")
    had_original = sidecar.exists()
    if had_original:
        shutil.copy2(sidecar, backup)

    try:
        if mode == "cold" and sidecar.exists():
            sidecar.unlink()
        yield
        if sidecar.exists():
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sidecar, capture_path)
    finally:
        if sidecar.exists():
            sidecar.unlink()
        if had_original:
            os.replace(backup, sidecar)
        elif backup.exists():
            backup.unlink()


def parse_llama_bench_json(text):
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def throughput_rows(rows):
    normalized = []
    for row in rows:
        n_prompt = int(row.get("n_prompt", 0) or 0)
        n_gen = int(row.get("n_gen", 0) or 0)
        if n_prompt > 0 and n_gen == 0:
            mode = "PP"
        elif n_gen > 0 and n_prompt == 0:
            mode = "TG"
        else:
            mode = "PG"
        normalized.append({
            "mode": mode,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "n_depth": int(row.get("n_depth", 0) or 0),
            "avg_ts": row.get("avg_ts"),
            "stddev_ts": row.get("stddev_ts"),
            "n_batch": row.get("n_batch"),
            "n_ubatch": row.get("n_ubatch"),
            "n_threads": row.get("n_threads"),
            "backend": row.get("backend"),
        })
    return normalized


def csv_result_rows(results):
    rows = []
    for result in results:
        throughput = result["throughput"] or [{}]
        for measured in throughput:
            rows.append({
                "run_id": result.get("run_id"),
                "mode": result.get("mode"),
                "tokens": result.get("tokens"),
                "batch": result.get("batch"),
                "ubatch": result.get("ubatch"),
                "threads": result.get("threads"),
                "s": result.get("s"),
                "tmax": result.get("tmax"),
                "sidecar": result.get("sidecar"),
                "returncode": result.get("returncode"),
                "wall_s": result.get("wall_s"),
                "bench_mode": measured.get("mode"),
                "n_prompt": measured.get("n_prompt"),
                "n_gen": measured.get("n_gen"),
                "n_depth": measured.get("n_depth"),
                "bench_n_batch": measured.get("n_batch"),
                "bench_n_ubatch": measured.get("n_ubatch"),
                "bench_n_threads": measured.get("n_threads"),
                "avg_ts": measured.get("avg_ts"),
                "stddev_ts": measured.get("stddev_ts"),
                "backend": measured.get("backend"),
            })
    return rows


def build_cells(args):
    pp_values = csv_values(args.pp)
    tg_values = csv_values(args.tg)
    tests = []
    tests.extend(("PP", value) for value in pp_values)
    tests.extend(("TG", value) for value in tg_values)
    return [
        {
            "mode": mode,
            "tokens": tokens,
            "batch": batch,
            "ubatch": ubatch,
            "threads": threads,
            "s": s,
            "tmax": tmax,
            "sidecar": sidecar,
        }
        for (mode, tokens), batch, ubatch, threads, s, tmax, sidecar in itertools.product(
            tests,
            csv_values(args.batch),
            csv_values(args.ubatch),
            csv_values(args.threads),
            csv_values(args.s),
            csv_values(args.tmax),
            string_values(args.sidecar),
        )
    ]


def build_command(args, cell):
    command = [
        str(args.bench),
        "-m", str(args.model),
        "-o", "json",
        "-r", str(args.repetitions),
        "-b", str(cell["batch"]),
        "-ub", str(cell["ubatch"]),
        "-t", str(cell["threads"]),
        "-ngl", str(args.ngl),
        "-fa", args.flash_attn,
        "-ctk", args.cache_type_k,
        "-ctv", args.cache_type_v,
    ]
    if cell["mode"] == "PP":
        command += ["-p", str(cell["tokens"]), "-n", "0"]
    else:
        command += ["-p", "0", "-n", str(cell["tokens"]), "-d", str(args.depth)]
    if args.no_warmup:
        command.append("--no-warmup")
    command += args.extra
    return command


def main():
    parser = argparse.ArgumentParser(description="Fresh-process wackMall PP/TG benchmark matrix")
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/wackmall-bench"))
    parser.add_argument("--pp", default="16,17,256,512")
    parser.add_argument("--tg", default="128,256")
    parser.add_argument("--batch", default="1024")
    parser.add_argument("--ubatch", default="512")
    parser.add_argument("--threads", default="10")
    parser.add_argument("--s", default="44")
    parser.add_argument("--tmax", default="16")
    parser.add_argument("--sidecar", default="cold,warm", help="comma-separated cold,warm")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--ngl", default="-1")
    parser.add_argument("--flash-attn", default="on")
    parser.add_argument("--cache-type-k", default="q4_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--adapt", choices=("0", "1"), default="1")
    parser.add_argument("--decay", default="0.999")
    parser.add_argument("--rampool", default="0")
    parser.add_argument("--pread", choices=("0", "1"), default="0")
    parser.add_argument("--predict", choices=("0", "1"), default="0")
    parser.add_argument("--prefetch-gb", default="0")
    parser.add_argument("--prefetch-threads", default="2")
    parser.add_argument("--async-h2d", choices=("0", "1"), default="0")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="extra llama-bench arguments after --")
    args = parser.parse_args()

    if not args.dry_run:
        if not args.bench.is_file():
            parser.error(f"llama-bench not found: {args.bench}")
        if not args.model.is_file():
            parser.error(f"model not found: {args.model}")

    sidecar_modes = set(string_values(args.sidecar))
    if not sidecar_modes <= {"cold", "warm"}:
        parser.error("--sidecar accepts only cold,warm")

    cells = build_cells(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "hardware.json").write_text(
        json.dumps(hardware_metadata(), indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "plan.json").write_text(
        json.dumps(cells, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(cells, indent=2))
        return 0

    results = []
    for index, cell in enumerate(cells, 1):
        run_id = f"{index:04d}-{cell['mode'].lower()}-{cell['tokens']}-s{cell['s']}-tmax{cell['tmax']}-{cell['sidecar']}"
        run_dir = args.output / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        stats_path = run_dir / "wackmall-perf.txt"
        sidecar_capture = run_dir / "result.tier"

        env = os.environ.copy()
        env.update({
            "LLAMA_EXPERT_S": str(cell["s"]),
            "LLAMA_EXPERT_TMAX": str(cell["tmax"]),
            "LLAMA_EXPERT_ADAPT": args.adapt,
            "LLAMA_EXPERT_DECAY": args.decay,
            "LLAMA_EXPERT_RAMPOOL": args.rampool,
            "LLAMA_EXPERT_PREAD": args.pread,
            "LLAMA_EXPERT_PREDICT": args.predict,
            "LLAMA_EXPERT_PREFETCH_GB": args.prefetch_gb,
            "LLAMA_EXPERT_PREFETCH_THREADS": args.prefetch_threads,
            "LLAMA_EXPERT_ASYNC_H2D": args.async_h2d,
            "LLAMA_EXPERT_STATS": str(stats_path),
        })
        command = build_command(args, cell)
        command_record = {
            "argv": command,
            "environment": {key: env.get(key) for key in TRACKED_ENV},
            "cell": cell,
        }
        (run_dir / "command.json").write_text(
            json.dumps(command_record, indent=2) + "\n", encoding="utf-8"
        )

        started = time.time()
        with preserved_sidecar(args.model, cell["sidecar"], sidecar_capture):
            proc = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        elapsed = time.time() - started
        (run_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        (run_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
        parsed = parse_llama_bench_json(proc.stdout)
        if parsed:
            (run_dir / "stdout.json").write_text(
                json.dumps(parsed, indent=2) + "\n", encoding="utf-8"
            )

        throughput = throughput_rows(parsed)
        result = {
            **cell,
            "run_id": run_id,
            "returncode": proc.returncode,
            "wall_s": elapsed,
            "throughput": throughput,
        }
        results.append(result)
        print(f"[{index}/{len(cells)}] {run_id}: rc={proc.returncode}")

    (args.output / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = args.output / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "run_id", "mode", "tokens", "batch", "ubatch", "threads", "s", "tmax",
            "sidecar", "returncode", "wall_s", "bench_mode", "n_prompt", "n_gen", "n_depth",
            "bench_n_batch", "bench_n_ubatch", "bench_n_threads", "avg_ts", "stddev_ts", "backend",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_result_rows(results))

    return 0 if all(result["returncode"] == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
