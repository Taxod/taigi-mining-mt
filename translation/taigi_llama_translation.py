#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch translation with Bohanlu/Taigi-Llama-2-Translator-{7B,13B}.

Output CSV columns:
    Source, Reference, Prediction, Prediction_Raw, Status,
    XCOMET_XL, XCOMET_Error_Spans
XCOMET columns are written empty; fill them in a separate scoring pass.
"""

import argparse
import csv
import os
import sys
import time
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
PROMPT_TEMPLATE = "[TRANS]\n{source_sentence}\n[/TRANS]\n[{target_language}]\n"

FIELDNAMES = [
    "Source",
    "Reference",
    "Prediction",
    "Prediction_Raw",
    "Status",
    "XCOMET_XL",
    "XCOMET_Error_Spans",
]

VALID_LANGS = {"HAN", "POJ", "HL", "ZH", "EN"}

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_pipeline(model_dir: str, dtype: torch.dtype, load_in_4bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

    # decoder-only batched generation requires left padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = dict(
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    model.eval()

    # greedy decoding; clear sampling params so transformers stops warning
    model.generation_config.do_sample = False
    for key in ("temperature", "top_p", "top_k", "typical_p"):
        if hasattr(model.generation_config, key):
            setattr(model.generation_config, key, None)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
    )
    return pipe, tokenizer


def postprocess(raw: str):
    """Cut at the closing tag. Returns (prediction, found_tag)."""
    idx = raw.find("[/")
    if idx == -1:
        return normalize(raw.strip()), False
    return normalize(raw[:idx].strip()), True


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """NFC. POJ / Tai-lo tone marks compare unequal across NFC and NFD."""
    return unicodedata.normalize("NFC", text)


def read_lines(path: Path):
    with path.open(encoding="utf-8") as fh:
        return [normalize(line.strip()) for line in fh if line.strip()]


def read_parallel(src_path: Path, ref_path: Path):
    """Two plain-text files, one sentence per line, same line count."""
    sources = read_lines(src_path)
    references = read_lines(ref_path)
    if len(sources) != len(references):
        raise SystemExit(
            f"data mismatch: {len(sources)} sources vs {len(references)} references"
        )
    return list(zip(sources, references))


def read_rows(path: Path, source_col: str, ref_col: str):
    """Read csv/tsv (with header), or plain txt with no reference."""
    suffix = path.suffix.lower()

    if suffix not in (".csv", ".tsv", ".tab"):
        return [(s, "") for s in read_lines(path)]

    delimiter = "\t" if suffix in (".tsv", ".tab") else ","
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        header = {k.lower(): k for k in reader.fieldnames or []}
        src_key = header.get(source_col.lower())
        ref_key = header.get(ref_col.lower())
        if src_key is None:
            raise SystemExit(
                f"column '{source_col}' not found in {path}; "
                f"available: {reader.fieldnames}"
            )
        rows = []
        for rec in reader:
            src = normalize((rec.get(src_key) or "").strip())
            ref = normalize((rec.get(ref_key) or "").strip()) if ref_key else ""
            rows.append((src, ref))
    return rows


def count_written(path: Path) -> int:
    """Number of data rows already in the output file (0 if absent)."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return max(sum(1 for _ in csv.reader(fh)) - 1, 0)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="csv/tsv with a Source column, or a plain-text source file")
    p.add_argument("--ref-file", type=Path, default=None,
                   help="parallel reference file; use with a plain-text --input")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model-dir", default="Bohanlu/Taigi-Llama-2-Translator-7B")
    p.add_argument("--target-lang", required=True, choices=sorted(VALID_LANGS))
    p.add_argument("--source-col", default="Source")
    p.add_argument("--ref-col", default="Reference")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=512,
                   help="rows grouped for length-sorting and flushed together")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    p.add_argument("--load-in-4bit", action="store_true",
                   help="NF4 quantization; needed for 13B on a 16 GB GPU")
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.ref_file is not None:
        rows = read_parallel(args.input, args.ref_file)
    else:
        rows = read_rows(args.input, args.source_col, args.ref_col)
    if args.limit:
        rows = rows[: args.limit]

    # contiguous sharding, so merging is plain concatenation in shard order
    if args.num_shards > 1:
        total = len(rows)
        per = -(-total // args.num_shards)
        start = args.shard_id * per
        rows = rows[start : start + per]
        print(f"[shard {args.shard_id}/{args.num_shards}] "
              f"rows {start}..{start + len(rows) - 1} of {total}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    done = 0 if args.no_resume else count_written(args.output)
    if done >= len(rows) and len(rows) > 0:
        print(f"nothing to do: {done} rows already in {args.output}", flush=True)
        return
    if done:
        print(f"resuming after {done} rows", flush=True)
    rows = rows[done:]

    pipe, tokenizer = build_pipeline(args.model_dir, DTYPES[args.dtype],
                                     load_in_4bit=args.load_in_4bit)

    mode = "a" if done and not args.no_resume else "w"
    fout = args.output.open(mode, encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES,
                            quoting=csv.QUOTE_MINIMAL)
    if mode == "w":
        writer.writeheader()
        fout.flush()

    t0 = time.time()
    processed = 0

    for block_start in range(0, len(rows), args.block_size):
        block = rows[block_start : block_start + args.block_size]

        prompts = [
            PROMPT_TEMPLATE.format(source_sentence=src,
                                   target_language=args.target_lang)
            for src, _ in block
        ]

        # sort by token length inside the block to cut padding waste,
        # then restore the original order before writing
        lengths = [len(tokenizer(p).input_ids) for p in prompts]
        order = sorted(range(len(block)), key=lambda i: lengths[i])
        sorted_prompts = [prompts[i] for i in order]

        results = [None] * len(block)
        for bstart in range(0, len(sorted_prompts), args.batch_size):
            chunk_idx = order[bstart : bstart + args.batch_size]
            chunk = sorted_prompts[bstart : bstart + args.batch_size]
            try:
                with torch.inference_mode():
                    outs = pipe(
                        chunk,
                        batch_size=len(chunk),
                        return_full_text=False,
                        do_sample=False,
                        repetition_penalty=args.repetition_penalty,
                        max_new_tokens=args.max_new_tokens,
                    )
                for pos, out in zip(chunk_idx, outs):
                    raw = out[0]["generated_text"]
                    pred, closed = postprocess(raw)
                    if not pred:
                        status = "EMPTY"
                    elif not closed:
                        status = "NO_END_TAG"
                    else:
                        status = "OK"
                    results[pos] = (pred, raw, status)
            except Exception as exc:  # keep row alignment on OOM / decode errors
                msg = f"ERROR: {type(exc).__name__}: {exc}"[:500]
                print(msg, file=sys.stderr, flush=True)
                for pos in chunk_idx:
                    results[pos] = ("", "", msg)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for (src, ref), (pred, raw, status) in zip(block, results):
            writer.writerow({
                "Source": src,
                "Reference": ref,
                "Prediction": pred,
                "Prediction_Raw": raw,
                "Status": status,
                "XCOMET_XL": "",
                "XCOMET_Error_Spans": "",
            })
        fout.flush()
        os.fsync(fout.fileno())

        processed += len(block)
        rate = processed / max(time.time() - t0, 1e-6)
        remain = (len(rows) - processed) / max(rate, 1e-6)
        print(f"{done + processed}/{done + len(rows)}  "
              f"{rate:.2f} sent/s  eta {remain / 60:.1f} min", flush=True)

    fout.close()
    print(f"wrote {args.output} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()