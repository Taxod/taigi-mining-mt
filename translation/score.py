#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score finished translation CSVs: BLEU, chrF2, and XCOMET-XL for -> en.

Input CSVs must carry: Source, Reference, Prediction, Prediction_Raw, Status.
Filenames are parsed as {src}_{tgt}_translation_{run}.csv.

Writes:
  - per-segment XCOMET_XL and XCOMET_Error_Spans back into each translation CSV
  - evaluation_scores_{run}.csv per run, plus evaluation_scores_all.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import unicodedata

# ============================================================================
# Config
# ============================================================================

XCOMET_MODEL = "Unbabel/XCOMET-XL"
XCOMET_BATCH_SIZE = 32
XCOMET_TARGET_LANGS = {"en"}      # only score directions translating into English

SCORE_HEADER = [
    "Run", "Model_Path", "Prompt_Mode", "Source_Lang", "Target_Lang",
    "BLEU_Score", "chrF2_Score", "XCOMET_XL", "BLEU_Tokenizer",
    "N", "Well_Formed", "Unclosed", "Untagged", "Empty",
]

# translate_taigi.py writes uppercase statuses; the Qwen eval script writes
# lowercase ones. Both are mapped onto the same four summary columns.
STATUS_MAP = {
    "OK": "Well_Formed", "ok": "Well_Formed",
    "NO_END_TAG": "Unclosed", "no_close": "Unclosed",
    "no_tag": "Untagged", "raw": "Well_Formed",
    "EMPTY": "Empty", "empty": "Empty",
}

FNAME_RE = re.compile(r"^(?P<src>[a-z]+)_(?P<tgt>[a-z]+)_translation_(?P<run>.+)\.csv$")

os.environ.setdefault("HF_HOME")

# ============================================================================
# Text processing
# ============================================================================

def normalize(text: str) -> str:
    """NFC. POJ / Tai-lo tone marks compare unequal across NFC and NFD."""
    return unicodedata.normalize("NFC", text or "")


def get_bleu_tokenizer(tgt_lang: str) -> str:
    """'zh' collapses a romanized sentence into one token, so Latin targets
    need '13a'."""
    if tgt_lang in ("zh", "han", "hanlo"):
        return "zh"
    if tgt_lang in ("en", "poj", "tl"):
        return "13a"
    return "char"


# ============================================================================
# IO
# ============================================================================

def parse_filename(path: str):
    """{src}_{tgt}_translation_{run}.csv -> (src, tgt, run), or None."""
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return m.group("src"), m.group("tgt"), m.group("run")


def read_translation_csv(path: str):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    preds = [normalize(r.get("Prediction", "")) for r in rows]
    refs = [normalize(r.get("Reference", "")) for r in rows]
    statuses = [r.get("Status", "") for r in rows]
    return preds, refs, statuses


def status_counts(statuses):
    counts = {"Well_Formed": 0, "Unclosed": 0, "Untagged": 0, "Empty": 0}
    for s in statuses:
        key = STATUS_MAP.get(s)
        if key is None:
            # ERROR: OutOfMemoryError... and anything else unrecognised
            key = "Untagged"
        counts[key] += 1
    return counts


def append_column(path: str, column: str, values) -> None:
    """Overwrite `column` in place if present, else append it."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if column in header:
        col = header.index(column)
    else:
        header.append(column)
        for row in rows:
            row.append("")
        col = len(header) - 1

    for row, val in zip(rows, values):
        row[col] = val

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ============================================================================
# Surface metrics
# ============================================================================

def evaluate_translations(predictions, references, tgt_lang):
    bleu_tok = get_bleu_tokenizer(tgt_lang)

    from sacrebleu.metrics import BLEU, CHRF
    bleu_metric = BLEU(tokenize=bleu_tok)
    bleu = bleu_metric.corpus_score(predictions, [references])
    chrf = CHRF().corpus_score(predictions, [references])

    print(f"SacreBLEU tokenizer: '{bleu_tok}' (target: {tgt_lang})")
    print(f"BLEU Score: {bleu.score:.2f}")
    print(f"Signature: {bleu_metric.get_signature()}")
    print(f"chrF2 Score: {chrf.score:.2f}")

    return bleu.score, chrf.score, bleu_tok


# ============================================================================
# XCOMET-XL
# ============================================================================

def load_xcomet(half: bool):
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("COMET not installed (pip install unbabel-comet); skipping XCOMET-XL.")
        return None

    try:
        print(f"\nLoading {XCOMET_MODEL} ...")
        model = load_from_checkpoint(download_model(XCOMET_MODEL))
        model.eval()
        if half:
            model = model.half()
            print("Running XCOMET-XL in fp16 (limited VRAM).")
        return model
    except Exception as exc:
        print(f"Could not load {XCOMET_MODEL} ({exc}); skipping XCOMET-XL.")
        return None


def score_xcomet(model, predictions, references, batch_size):
    """Reference-based mode (mt + ref, src omitted). Scores are on 0-1."""
    import torch

    data = [
        # COMET rejects empty hypotheses; a single space keeps the row aligned
        # and scores near zero, which is the intended reading of empty output.
        {"mt": (p if p.strip() else " "), "ref": r}
        for p, r in zip(predictions, references)
    ]

    gpus = 1 if torch.cuda.is_available() else 0
    if gpus == 0:
        print("No CUDA device: XCOMET-XL runs on CPU and will be slow.")

    out = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=True)

    # error_spans is xCOMET-only; plain COMET metrics carry no such metadata.
    spans = getattr(getattr(out, "metadata", None), "error_spans", None) \
        or [[] for _ in data]

    return float(out.system_score), [float(x) for x in out.scores], list(spans)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="./results/*_translation_*.csv",
                   help="glob for the finished translation CSVs")
    p.add_argument("--out-dir", default=None,
                   help="where the scores CSVs go; defaults to each input's folder")
    p.add_argument("--model-path", default="Bohanlu/Taigi-Llama-2-Translator-7B",
                   help="recorded in the Model_Path column")
    p.add_argument("--prompt-mode", default="taigi-llama",
                   help="recorded in the Prompt_Mode column")
    p.add_argument("--no-xcomet", action="store_true")
    p.add_argument("--xcomet-batch-size", type=int, default=XCOMET_BATCH_SIZE)
    p.add_argument("--xcomet-fp16", action="store_true",
                   help="force fp16; auto-enabled when VRAM < 24 GB")
    return p.parse_args()


def main():
    args = parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        sys.exit(f"No files match {args.pattern}")

    jobs = []
    for path in paths:
        parsed = parse_filename(path)
        if parsed is None:
            print(f"[skip] {path}: name is not {{src}}_{{tgt}}_translation_{{run}}.csv")
            continue
        src_lang, tgt_lang, run_name = parsed
        jobs.append({"path": path, "src": src_lang, "tgt": tgt_lang, "run": run_name})

    if not jobs:
        sys.exit("No filenames could be parsed.")

    print(f"{len(jobs)} file(s) to score")

    # ------------------------------------------------------------ pass 1
    summary_rows = []
    for job in jobs:
        print(f"\n{'=' * 50}")
        print(f"{job['src']} -> {job['tgt']}  [{job['run']}]  {job['path']}")
        print("=" * 50)

        preds, refs, statuses = read_translation_csv(job["path"])
        counts = status_counts(statuses)

        bleu, chrf, bleu_tok = evaluate_translations(preds, refs, job["tgt"])

        row = [
            job["run"], args.model_path, args.prompt_mode, job["src"], job["tgt"],
            f"{bleu:.2f}", f"{chrf:.2f}", "", bleu_tok,
            len(preds), counts["Well_Formed"], counts["Unclosed"],
            counts["Untagged"], counts["Empty"],
        ]
        job["row"] = row
        job["preds"], job["refs"] = preds, refs
        summary_rows.append(row)

    # ------------------------------------------------------------ pass 2
    xcomet_jobs = [j for j in jobs
                   if not args.no_xcomet and j["tgt"] in XCOMET_TARGET_LANGS]
    skipped = [j for j in jobs if j not in xcomet_jobs and not args.no_xcomet]
    for j in skipped:
        print(f"\n[XCOMET skipped] {j['src']} -> {j['tgt']}: "
              f"target not in {sorted(XCOMET_TARGET_LANGS)}")

    if xcomet_jobs:
        half = args.xcomet_fp16
        try:
            import torch
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                half = half or vram < 24
        except ImportError:
            pass

        comet_model = load_xcomet(half)
        if comet_model is not None:
            for job in xcomet_jobs:
                print(f"\nXCOMET-XL: {job['run']} {job['src']} -> {job['tgt']}")
                system_score, seg_scores, spans = score_xcomet(
                    comet_model, job["preds"], job["refs"], args.xcomet_batch_size
                )
                n_flagged = sum(1 for s in spans if s)
                print(f"XCOMET-XL system score: {system_score:.4f}  (0-1 scale)")
                print(f"Segments with error spans: {n_flagged}/{len(spans)}")

                append_column(job["path"], "XCOMET_XL",
                              [f"{s:.4f}" for s in seg_scores])
                append_column(job["path"], "XCOMET_Error_Spans",
                              [json.dumps(s, ensure_ascii=False) for s in spans])

                job["row"][SCORE_HEADER.index("XCOMET_XL")] = f"{system_score:.4f}"

    # ------------------------------------------------------------ output
    by_run = {}
    for job in jobs:
        out_dir = args.out_dir or os.path.dirname(job["path"]) or "."
        by_run.setdefault((out_dir, job["run"]), []).append(job["row"])

    for (out_dir, run_name), rows in by_run.items():
        os.makedirs(out_dir, exist_ok=True)
        scores_file = os.path.join(out_dir, f"evaluation_scores_{run_name}.csv")
        with open(scores_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(SCORE_HEADER)
            writer.writerows(rows)
        print(f"\n-> {scores_file}")

    summary_dir = args.out_dir or os.path.dirname(jobs[0]["path"]) or "."
    summary_file = os.path.join(summary_dir, "evaluation_scores_all.csv")
    with open(summary_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SCORE_HEADER)
        writer.writerows(summary_rows)
    print(f"-> {summary_file}")


if __name__ == "__main__":
    main()