#!/usr/bin/env python3
"""Analyse translation CSVs: error spans, output language, repetition.

Reads the per-run translation files produced by the evaluation script
(columns Source, Reference, Prediction, Prediction_Raw, Status, XCOMET_XL,
XCOMET_Error_Spans) and reports three things that are already latent in them:

  1. xCOMET error spans   how much of each hypothesis the metric marks as
                          wrong, and at what severity
  2. Output language      GlotLID over the Prediction column, which turns
                          "the model returns Japanese" into a count
  3. Repetition           share of outputs dominated by a repeated n-gram,
                          which is what fills an unclosed generation

Sections 1 and 3 need nothing beyond the CSVs. Section 2 needs a GlotLID
binary and is skipped if GLOTLID_PATH does not exist.

Writes span_stats.csv, language_stats.csv and repetition_stats.csv into OUT_DIR.

Install:
    pip install fasttext        # only for the language section
"""

import csv
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from huggingface_hub import hf_hub_download
import sacrebleu

# =============================================================================
# EDIT 1 - the runs to analyse.
# =============================================================================
RUNS = [
    {"label": "CPT mined", "direction": "poj-en",
     "path": "./qwen/CP_wikiplus10_035/poj_en_translation_ckpt11500.csv"},
    {"label": "Simple Match", "direction": "poj-en",
     "path": "./qwen/simple_match_2M/poj_en_translation_ckpt8500.csv"},
    {"label": "LaBSE mined", "direction": "poj-en",
     "path": "./qwen/LaBSE_wikiplus10/poj_en_translation_ckpt13500.csv"},
    {"label": "NLLB CPT", "direction": "poj-en",
     "path": "./nllb/nllbcpt/poj_en_translation_ft_b_ckpt46500.csv"},
    {"label": "NLLB 080", "direction": "poj-en",
     "path": "./nllb/nllb_finetuned_080/poj_en_translation_ft_a_ckpt3000.csv"},
    {"label": "CPT mined", "direction": "en-poj",
     "path": "./qwen/CP_wikiplus10_035/en_poj_translation_ckpt11500.csv"},
    {"label": "Simple Match", "direction": "en-poj",
     "path": "./qwen/simple_match_2M/en_poj_translation_ckpt8500.csv"},
    {"label": "LaBSE mined", "direction": "en-poj",
     "path": "./qwen/LaBSE_wikiplus10/en_poj_translation_ckpt13500.csv"},
    {"label": "NLLB CPT", "direction": "en-poj",
     "path": "./nllb/nllbcpt/en_poj_translation_ft_b_ckpt46500.csv"},
    {"label": "NLLB 080", "direction": "en-poj",
     "path": "./nllb/nllb_finetuned_080/en_poj_translation_ft_a_ckpt3000.csv"},
]

# =============================================================================
# EDIT 3 - repetition threshold. An output whose distinct-4 ratio falls below
# this is counted as degenerate. 0.5 means half its 4-grams are duplicates.
# =============================================================================
DISTINCT_4_THRESHOLD = 0.5
NGRAM_N = 4
P95_WORDS = 22
OUT_DIR = Path("out")


# -----------------------------------------------------------------------------


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def read_rows(path: str) -> list[dict]:
    csv.field_size_limit(10 ** 7)  # Prediction_Raw can be very long
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))

def length_split_stats(rows: list[dict], direction: str) -> dict:
    """chrF2 on sentences above and below the training p95, by English length."""
    english_col = "Reference" if direction == "poj-en" else "Source"
    result = {}
    for name, keep in (("long", lambda k: k > P95_WORDS), ("short", lambda k: k <= P95_WORDS)):
        subset = [r for r in rows if keep(len((r.get(english_col) or "").split()))]
        hyps = [normalize((r.get("Prediction") or "").strip()) for r in subset]
        refs = [normalize((r.get("Reference") or "").strip()) for r in subset]
        result[f"{name}_n"] = len(subset)
        result[f"{name}_chrf2"] = sacrebleu.corpus_chrf(hyps, [refs]).score if subset else float("nan")
    return result
# --- 1. error spans ----------------------------------------------------------

def span_stats(rows: list[dict]) -> dict:
    """Aggregate the XCOMET_Error_Spans column.

    Coverage is the summed span length over the hypothesis length. The offsets
    xCOMET returns index its own detokenised string, in which characters
    outside the encoder vocabulary appear as <unk>; POJ diacritics do collapse
    that way in this data. Coverage is therefore approximate and is clipped at
    1.0.
    """
    n = 0
    with_spans = 0
    n_spans = 0
    severities = Counter()
    coverage_sum = 0.0
    score_with, score_without = [], []

    for row in rows:
        raw = (row.get("XCOMET_Error_Spans") or "").strip()
        prediction = normalize((row.get("Prediction") or "").strip())
        if not raw:
            continue
        try:
            spans = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  skipped an unparsable span cell", file=sys.stderr)
            continue

        n += 1
        try:
            score = float(row.get("XCOMET_XL") or "nan")
        except ValueError:
            score = float("nan")

        if spans:
            with_spans += 1
            n_spans += len(spans)
            for span in spans:
                severities[span.get("severity", "unspecified")] += 1
            covered = sum(max(0, int(s.get("end", 0)) - int(s.get("start", 0)))
                          for s in spans)
            if prediction:
                coverage_sum += min(1.0, covered / len(prediction))
            score_with.append(score)
        else:
            score_without.append(score)

    def mean(values):
        values = [v for v in values if v == v]  # drop NaN
        return sum(values) / len(values) if values else float("nan")

    return {
        "segments": n,
        "with_spans": with_spans,
        "with_spans_pct": 100 * with_spans / n if n else float("nan"),
        "spans_total": n_spans,
        "critical": severities.get("critical", 0),
        "major": severities.get("major", 0),
        "minor": severities.get("minor", 0),
        "mean_coverage_pct": 100 * coverage_sum / with_spans if with_spans else float("nan"),
        "mean_score_with_spans": mean(score_with),
        "mean_score_without_spans": mean(score_without),
    }


# --- 2. output language ------------------------------------------------------

def language_stats(rows: list[dict], model) -> Counter:
    labels = Counter()
    for row in rows:
        text = normalize((row.get("Prediction") or "").strip())
        text = " ".join(text.split())  # fastText rejects embedded newlines
        if not text:
            labels["<empty>"] += 1
            continue
        prediction, _ = model.predict(text, k=1)
        labels[prediction[0].replace("__label__", "")] += 1
    return labels


# --- 3. repetition -----------------------------------------------------------

def distinct_ratio(text: str, n: int) -> float:
    """Unique n-grams over total n-grams. 1.0 means no repetition at all."""
    tokens = text.split()
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_stats(rows: list[dict]) -> dict:
    ratios = []
    degenerate = 0
    degenerate_unclosed = 0

    for row in rows:
        text = normalize((row.get("Prediction") or "").strip())
        ratio = distinct_ratio(text, NGRAM_N)
        ratios.append(ratio)
        if ratio < DISTINCT_4_THRESHOLD:
            degenerate += 1
            if (row.get("Status") or "").strip() == "no_close":
                degenerate_unclosed += 1

    n = len(ratios)
    return {
        "segments": n,
        "mean_distinct_4": sum(ratios) / n if n else float("nan"),
        "degenerate": degenerate,
        "degenerate_pct": 100 * degenerate / n if n else float("nan"),
        "degenerate_and_unclosed": degenerate_unclosed,
    }


# -----------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    model = None
    try:
        import fasttext
        print("Downloading or loading GlotLID from Hugging Face cache...")
        model_path = hf_hub_download(
            repo_id="cis-lmu/glotlid",
            filename="model.bin",
            cache_dir=None
        )
        model = fasttext.load_model(model_path)
    except ImportError:
        print("Missing required packages. Please install fasttext and huggingface_hub.")
    except Exception as e:
        print(f"Failed to load GlotLID model. Skipping the language section. Error: {e}")
    span_rows, lang_rows, rep_rows = [], [], []

    for spec in RUNS:
        print(f"\n{spec['label']} ({spec['direction']}) <- {spec['path']}")
        rows = read_rows(spec["path"])
        print(f"  {len(rows)} rows")

        spans = span_stats(rows)
        print(f"  spans: {spans['with_spans']}/{spans['segments']} segments "
              f"({spans['with_spans_pct']:.1f}%), coverage "
              f"{spans['mean_coverage_pct']:.1f}%, "
              f"critical={spans['critical']} major={spans['major']} "
              f"minor={spans['minor']}")
        print(f"  mean xCOMET with spans {spans['mean_score_with_spans']:.4f}, "
              f"without {spans['mean_score_without_spans']:.4f}")
        span_rows.append({"label": spec["label"], "direction": spec["direction"],
                          **spans})

        rep = repetition_stats(rows)
        print(f"  repetition: mean distinct-{NGRAM_N} {rep['mean_distinct_4']:.3f}, "
              f"{rep['degenerate']} degenerate ({rep['degenerate_pct']:.1f}%), "
              f"{rep['degenerate_and_unclosed']} of them unclosed")
        rep_rows.append({"label": spec["label"], "direction": spec["direction"],
                         **rep})
        split = length_split_stats(rows, spec["direction"])
        print(f"  length split: long n={split['long_n']} chrF2={split['long_chrf2']:.2f}, "
              f"short n={split['short_n']} chrF2={split['short_chrf2']:.2f}")

        if model is not None:
            labels = language_stats(rows, model)
            total = sum(labels.values())
            top = labels.most_common(6)
            print("  languages: " + ", ".join(
                f"{name} {100 * count / total:.1f}%" for name, count in top))
            for name, count in labels.items():
                lang_rows.append({
                    "label": spec["label"],
                    "direction": spec["direction"],
                    "language": name,
                    "count": count,
                    "pct": f"{100 * count / total:.2f}",
                })

    def write(path, rows):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path}")

    print()
    write(OUT_DIR / "span_stats.csv", span_rows)
    write(OUT_DIR / "repetition_stats.csv", rep_rows)
    write(OUT_DIR / "language_stats.csv", lang_rows)


if __name__ == "__main__":
    main()
