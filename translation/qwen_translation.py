"""Evaluate Qwen models on Taiwanese Hokkien translation (base vs fine-tuned)."""

import os
import re
import csv
import json
import unicodedata

import torch
import sacrebleu
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
# ============================================================================
# Config
# ============================================================================

BASE_MODEL = "Qwen/Qwen3.5-0.8B"

# Checkpoints can come from a local directory or from a Hub repo whose
# subfolders are the checkpoint-N directories. Set USE_HF_HUB accordingly.
USE_HF_HUB = True
HF_REPO = "w-y-kuo/CP_wikiplus10_035"
CKPT_DIR = ""

MODEL_TAG = (HF_REPO.rsplit("/", 1)[-1] if USE_HF_HUB
             else os.path.basename(CKPT_DIR.rstrip("/")))

# checkpoint-1000 ... checkpoint-23000 every 1000, plus the final checkpoint-23300.
# CKPT_STEPS = list(range(1000, 23001, 1000)) + [23300]
CKPT_STEPS = [11500,6000]
# (run_name, model_path, subfolder, prompt_mode). run_name is the file suffix;
# subfolder is None for local paths, which already point at the checkpoint.
if USE_HF_HUB:
    # RUNS = [("base", BASE_MODEL, None, "baseline")] + [
    #     (f"ckpt{step}", HF_REPO, f"checkpoint-{step}", "sft")
    #     for step in CKPT_STEPS
    # ]
    RUNS = [
        (f"ckpt{step}", HF_REPO, f"checkpoint-{step}", "sft")
        for step in CKPT_STEPS
    ]
else:
    RUNS = [
        (f"ckpt{step}", f"{CKPT_DIR}/checkpoint-{step}", None, "sft")
        for step in CKPT_STEPS
    ]

SKIP_MISSING_CKPT = True          # local mode only: warn and continue

TOKENIZER_FALLBACK = BASE_MODEL   # used if a checkpoint has no tokenizer files
TORCH_DTYPE = "auto"              # torch.float32 on MPS (incomplete bf16)
MAX_NEW_TOKENS = 1024
VERIFY_PROMPT_FORMAT = True       # print first prompt per run

# Batched generation. Left padding plus a padded attention mask means results
# are numerically close to, but not bit-identical with, BATCH_SIZE = 1.
BATCH_SIZE = 64
SORT_BY_LENGTH = True             # group similar lengths to cut padding waste

PROMPT_MODE = "baseline"          # set per run by main; do not edit

# ---------------------------------------------------------------------------
# XCOMET-XL (reference-based mode: mt + ref, src omitted)
# ---------------------------------------------------------------------------
# Requires: pip install unbabel-comet
# unbabel-comet pins transformers<5, which this checkpoint needs 5.x for, so
# the two cannot share one env. Leave XCOMET_ENABLED off here and score the
# saved CSVs from a separate comet env when that conflict applies.
# The weights are gated on the Hub: accept the licence and run `hf auth login`
# once, otherwise download_model() raises a 403.

XCOMET_ENABLED = False
XCOMET_MODEL = "Unbabel/XCOMET-XL"
XCOMET_BATCH_SIZE = 32
XCOMET_TARGET_LANGS = {"en"}      # only score directions translating into English

# ============================================================================
# Language tables
# ============================================================================

# Baseline mode: system prompts for the un-tuned model.
CUSTOM_PROMPTS = {
    ("poj", "zh"): "You are a linguistic expert. Translate the following Taiwanese Hokkien Pe̍h-ōe-jī (POJ) text into Taiwanese Traditional Chinese. Output ONLY the translation without any explanations.",
    ("poj", "en"): "You are a linguistic expert. Translate the following Taiwanese Hokkien Pe̍h-ōe-jī (POJ) text into English. Output ONLY the translation without any explanations.",
    ("hanlo",
     "en"): "You are a linguistic expert. Translate the following Taiwanese Hokkien Hàn-lô Tâi-bûn, Hàn-lô (Hàn-lô) text into English. Output ONLY the translation without any explanations.",
    ("tl", "zh"): "You are a linguistic expert. Translate the following Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案) text into Taiwanese Traditional Chinese. Output ONLY the translation without any explanations.",
    ("han", "zh"): "You are a linguistic expert. Translate the following Taiwanese Hokkien Han characters (Taigi Hanji) into Taiwanese Traditional Chinese. Output ONLY the translation without any explanations.",
    ("tl", "en"): "You are a linguistic expert. Translate the following Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案) text into English. Output ONLY the translation without any explanations.",
    ("tl", "han"): "You are a linguistic expert. Translate the following Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案) text into Taiwanese Hokkien Han characters (Taigi Hanji). Output ONLY the translation without any explanations.",
    ("zh", "poj"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Taiwanese Traditional Chinese text into Taiwanese Pe̍h-ōe-jī (POJ). "
        "Note: POJ is a specific romanization system. Ensure correct use of tone diacritics and use hyphens to connect syllables within a single word. "
        "Output ONLY the translation without any explanations."
    ),
    ("zh", "tl"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Taiwanese Traditional Chinese text into the official Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案). "
        "Note: Follow the official spelling conventions promulgated by the Ministry of Education in Taiwan. Maintain correct hyphenation. "
        "Output ONLY the translation without any explanations."
    ),
    ("zh", "han"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Taiwanese Traditional Chinese text into proper Taiwanese Han characters (Taigi Hanji). "
        "Note: Use authentic Taiwanese Hokkien vocabulary and officially recognized Han characters. Do not output standard Mandarin. "
        "Output ONLY the translation without any explanations."
    ),
    ("en", "poj"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following English text into Taiwanese Pe̍h-ōe-jī (POJ). "
        "Ensure correct use of tone diacritics and use hyphens to connect syllables. Output ONLY the translation without any explanations."
    ),("en", "hanlo"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following English text into Taiwanese Hokkien Hàn-lô Tâi-bûn, Hàn-lô (Hàn-lô). "
        "Ensure correct use of tone diacritics and use hyphens to connect syllables. Output ONLY the translation without any explanations."
    ),
    ("en", "tl"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following English text into the official Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案). "
        "Maintain correct hyphenation according to the official standards. Output ONLY the translation without any explanations."
    ),
    ("han", "tl"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Taiwanese Hokkien Han characters (Taigi Hanji) into the official Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案). "
        "Maintain correct hyphenation. Output ONLY the translation without any explanations."
    ),
}

BASELINE_NAMES = {
    "zh": "Taiwanese Traditional Chinese",
    "en": "English",
    "poj": "Taiwanese Hokkien Pe̍h-ōe-jī (POJ)",
    "tl": "Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案)",
    "han": "Taiwanese Hokkien Han characters (Taigi Hanji)",
    "hanlo": "Taiwanese Hokkien Hàn-lô Tâi-bûn, Hàn-lô (Hàn-lô)",
}

# SFT mode: (display_name, tag) as they appear in the training data.
# poj/en verified against the training log. zh/tl/han are guesses and are only
# relevant if the training data covers them.
SFT_LANG = {
    "poj": ("Taiwanese Hokkien Pe̍h-ōe-jī (POJ)", "poj"),
    "en": ("English", "eng"),
    "tl": ("Taiwanese Romanization System (Tâi-lô)", "tl"),
    "han": ("Taiwanese Hokkien Han characters (Taigi Hanji)", "han"),
    "zh": ("Mandarin (Traditional Chinese characters)", "zh"),
    "hanlo": ("Taiwanese Hokkien Hàn-lô Tâi-bûn, Hàn-lô (Hàn-lô)", "hanlo"),
}

# ============================================================================
# Prompt building
# ============================================================================

def build_sft_prompt(text: str, src_lang: str, tgt_lang: str) -> str:
    src_name, src_tag = SFT_LANG[src_lang]
    tgt_name, tgt_tag = SFT_LANG[tgt_lang]
    return (
        f"Translate the following {src_name} text to {tgt_name}. "
        f"Put it in this format <{tgt_tag}> {tgt_name} translation </{tgt_tag}>.\n"
        f"<{src_tag}> {text} </{src_tag}>"
    )


def build_baseline_messages(text: str, src_lang: str, tgt_lang: str) -> list[dict]:
    fallback = (
        f"You are a professional translator. Translate the following text from "
        f"{BASELINE_NAMES.get(src_lang, src_lang)} to {BASELINE_NAMES.get(tgt_lang, tgt_lang)}. "
        f"Output ONLY the translation without any explanations or additional text."
    )
    return [
        {"role": "system", "content": CUSTOM_PROMPTS.get((src_lang, tgt_lang), fallback)},
        {"role": "user", "content": text},
    ]


def render_prompt(tokenizer, text: str, src_lang: str, tgt_lang: str) -> str:
    # SFT mode builds chatml by hand: training used LLaMA-Factory's `chatml`
    # template, which emits no system block, but Qwen's built-in template
    # inserts a default one when no system message is passed.
    if PROMPT_MODE == "sft":
        user = build_sft_prompt(text, src_lang, tgt_lang)
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

    messages = build_baseline_messages(text, src_lang, tgt_lang)
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

# ============================================================================
# Text processing
# ============================================================================

def normalize(text: str) -> str:
    """NFC. POJ/Tai-lo tone marks compare unequal across NFC/NFD."""
    return unicodedata.normalize("NFC", text)


def strip_sft_tags(raw: str, tgt_lang: str) -> tuple[str, str]:
    """Returns (text, status) with status in ok / no_close / no_tag / empty."""
    tag = SFT_LANG[tgt_lang][1]

    m = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL)
    if m:
        out = m.group(1).strip()
        return out, ("ok" if out else "empty")

    # Opening tag only: usually truncation. Keep the text rather than dropping it.
    m = re.search(rf"<{tag}>(.*)", raw, re.DOTALL)
    if m:
        out = m.group(1).strip()
        return out, ("no_close" if out else "empty")

    out = re.sub(r"</?[A-Za-z][A-Za-z0-9_-]*>", "", raw).strip()
    return out, ("no_tag" if out else "empty")


def postprocess(raw: str, tgt_lang: str) -> tuple[str, str]:
    # Baseline output is scored as-is: the base model was not trained to emit a
    # wrapper, so cleaning only one side would bias the comparison.
    if PROMPT_MODE == "sft":
        out, status = strip_sft_tags(raw, tgt_lang)
        return normalize(out), status
    return raw, ("empty" if not raw else "raw")


def get_bleu_tokenizer(tgt_lang: str) -> str:
    """'zh' collapses a romanized sentence into one token, so Latin targets need '13a'."""
    if tgt_lang in ("zh", "han", "hanlo"):
        return "zh"
    if tgt_lang in ("en", "poj", "tl"):
        return "13a"
    return "char"

# ============================================================================
# Data
# ============================================================================

def load_data(source_file: str, reference_file: str) -> tuple[list[str], list[str]]:
    for path in (source_file, reference_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: '{path}'")

    with open(source_file, encoding="utf-8") as f:
        sources = [normalize(line.strip()) for line in f if line.strip()]
    with open(reference_file, encoding="utf-8") as f:
        references = [normalize(line.strip()) for line in f if line.strip()]

    if len(sources) != len(references):
        raise ValueError(
            f"Data mismatch! Sources: {len(sources)}, References: {len(references)}"
        )
    return sources, references

# ============================================================================
# Model
# ============================================================================

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def enable_cache(cfg, path="(top)") -> None:
    """Training sets use_cache=False for gradient checkpointing and it persists
    into config.json. Left off, generation recomputes the prefix every token."""
    if getattr(cfg, "use_cache", None) is False:
        print(f"  {path}.use_cache was False; re-enabling")
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = True
    for name in ("text_config", "llm_config", "language_config", "decoder"):
        sub = getattr(cfg, name, None)
        if sub is not None and hasattr(sub, "to_dict"):
            enable_cache(sub, f"{path}.{name}")


def setup_translator(model_path: str, subfolder: str | None = None):
    """model_path is a local checkpoint dir, or a Hub repo id with `subfolder`
    naming the checkpoint directory inside it."""
    device = pick_device()
    print(f"Using device: {device}")

    where = {"subfolder": subfolder} if subfolder else {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **where)
    except (OSError, ValueError):
        print(f"No tokenizer in {model_path}/{subfolder or ''}, using {TOKENIZER_FALLBACK}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FALLBACK)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=TORCH_DTYPE, **where
        )
    except TypeError:
        # transformers < 5 spells the argument torch_dtype.
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=TORCH_DTYPE, **where
        )
    except (ValueError, KeyError) as exc:
        # Qwen3.5 carries a vision tower; the causal-LM class may not cover it.
        print(f"AutoModelForCausalLM failed ({exc}); trying AutoModel")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path, dtype=TORCH_DTYPE, **where)

    enable_cache(model.config)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True

    model = model.to(device)
    model.eval()

    tokenizer.padding_side = "left"   # required for batched decoder-only generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        raise ValueError("Tokenizer vocab exceeds model embeddings; mismatched checkpoint.")

    return tokenizer, model, device

# ============================================================================
# Translation
# ============================================================================

def translate_and_save(tokenizer, model, device,
                       sources: list[str], references: list[str],
                       output_file: str, src_lang: str, tgt_lang: str,
                       batch_size: int = BATCH_SIZE) -> tuple[list[str], dict]:
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    prompts = [render_prompt(tokenizer, s, src_lang, tgt_lang) for s in sources]

    if VERIFY_PROMPT_FORMAT and prompts:
        print(f"\n--- prompt ({PROMPT_MODE}) ---")
        print(json.dumps(prompts[0], ensure_ascii=False))
        print("--- end ---\n")

    # Length-sorted batching: shorter prompts stop padding the long ones.
    # Results are written back through `order`, so the file keeps input order.
    order = list(range(len(prompts)))
    if SORT_BY_LENGTH and len(prompts) > 1:
        lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        order.sort(key=lambda i: lengths[i])

    raw_by_index: list[str] = [""] * len(prompts)
    stop_strings = [f"</{SFT_LANG[tgt_lang][1]}>"] if PROMPT_MODE == "sft" else None

    with torch.no_grad():
        for start in tqdm(range(0, len(order), batch_size),
                          desc=f"Translating {src_lang} -> {tgt_lang}"):
            idx = order[start:start + batch_size]
            batch_prompts = [prompts[i] for i in idx]

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

            # kwargs = {
            #     "max_new_tokens": MAX_NEW_TOKENS,
            #     "do_sample": False,
            #     "pad_token_id": tokenizer.pad_token_id,
            #     "use_cache": True,
            # }
            kwargs = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "use_cache": True,
            }

            if stop_strings:
                try:
                    generated = model.generate(
                        **inputs, **kwargs,
                        stop_strings=stop_strings,
                        tokenizer=tokenizer,
                    )
                except (TypeError, ValueError):
                    generated = model.generate(**inputs, **kwargs)
            else:
                generated = model.generate(**inputs, **kwargs)

            # Left padding means every row shares one prompt length.
            new_tokens = generated[:, inputs["input_ids"].shape[1]:]

            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for i, raw in zip(idx, decoded):
                raw_by_index[i] = normalize(raw.strip())

    predictions, statuses = [], []
    for raw in raw_by_index:
        pred, status = postprocess(raw, tgt_lang)
        predictions.append(pred)
        statuses.append(status)

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Reference", "Prediction", "Prediction_Raw", "Status"])
        writer.writerows(zip(sources, references, predictions, raw_by_index, statuses))

    stats = {
        "n": len(predictions),
        "ok": statuses.count("ok"),
        "no_close": statuses.count("no_close"),
        "no_tag": statuses.count("no_tag"),
        "empty": statuses.count("empty"),
    }

    print(f"\nSaved to {output_file}")
    if PROMPT_MODE == "sft":
        print(f"Format: {stats['ok']}/{stats['n']} ok, {stats['no_close']} unclosed, "
              f"{stats['no_tag']} untagged, {stats['empty']} empty")
    else:
        print(f"Output: {stats['n']} sentences, {stats['empty']} empty")

    return predictions, stats

# ============================================================================
# Evaluation: surface metrics
# ============================================================================

def evaluate_translations(predictions: list[str], references: list[str],
                          tgt_lang: str) -> tuple[float, float, str]:
    print("\n--- Evaluation Results ---")

    bleu_tok = get_bleu_tokenizer(tgt_lang)
    print(f"SacreBLEU tokenizer: '{bleu_tok}' (target: {tgt_lang})")

    # The metrics API carries the signature; corpus_bleu() returns a bare score.
    try:
        from sacrebleu.metrics import BLEU, CHRF
        bleu_metric = BLEU(tokenize=bleu_tok)
        bleu = bleu_metric.corpus_score(predictions, [references])
        chrf_metric = CHRF()
        chrf = chrf_metric.corpus_score(predictions, [references])
        signature = str(bleu_metric.get_signature())
    except ImportError:
        bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize=bleu_tok)
        chrf = sacrebleu.corpus_chrf(predictions, [references])
        signature = f"tok:{bleu_tok}|version:{sacrebleu.__version__}"

    print(f"BLEU Score: {bleu.score:.2f}")
    print(f"Signature: {signature}")
    print(f"chrF2 Score: {chrf.score:.2f}")

    return bleu.score, chrf.score, bleu_tok

# ============================================================================
# Evaluation: XCOMET-XL
# ============================================================================

def load_xcomet():
    """Loads XCOMET-XL once. Returns None if COMET or the weights are unavailable."""
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("COMET not installed (pip install unbabel-comet); skipping XCOMET-XL.")
        return None

    try:
        print(f"\nLoading {XCOMET_MODEL} ...")
        ckpt = download_model(XCOMET_MODEL)
        model = load_from_checkpoint(ckpt)
        model.eval()
        return model
    except Exception as exc:
        print(f"Could not load {XCOMET_MODEL} ({exc}); skipping XCOMET-XL.")
        return None


def score_xcomet(model, predictions: list[str],
                 references: list[str]) -> tuple[float, list[float], list]:
    """Reference-based mode (mt + ref, src omitted).

    Returns (system_score, segment_scores, error_spans); scores are on COMET's
    native 0-1 scale.
    """
    data = [
        # COMET chokes on empty hypotheses; a single space keeps the row aligned
        # and scores near zero, which is the intended reading of an empty output.
        {"mt": (p if p.strip() else " "), "ref": r}
        for p, r in zip(predictions, references)
    ]

    gpus = 1 if torch.cuda.is_available() else 0
    if gpus == 0:
        print("No CUDA device: XCOMET-XL runs on CPU and will be slow.")

    out = model.predict(data, batch_size=XCOMET_BATCH_SIZE, gpus=gpus,
                        progress_bar=True)

    # error_spans is xCOMET-only; plain COMET metrics carry no such metadata.
    spans = getattr(getattr(out, "metadata", None), "error_spans", None) or [[] for _ in data]

    return float(out.system_score), [float(x) for x in out.scores], list(spans)


def read_translation_csv(path: str) -> tuple[list[str], list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["Prediction"] for r in rows], [r["Reference"] for r in rows]


def append_segment_scores(path: str, column: str, values: list[float]) -> None:
    """Rewrites a translation CSV with one extra per-segment score column."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if column in header:
        col = header.index(column)
        for row, val in zip(rows, values):
            row[col] = f"{val:.4f}"
    else:
        header.append(column)
        for row, val in zip(rows, values):
            row.append(f"{val:.4f}")

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def append_error_spans(path: str, spans: list, column: str = "XCOMET_Error_Spans") -> None:
    """Adds one JSON column holding this segment's detected error spans.

    Each span is a dict with `start`, `end`, `text`, `confidence`, `severity`.
    Offsets index into `mt`. Stored as JSON because CSV has no nested structure.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    encoded = [json.dumps(s, ensure_ascii=False) for s in spans]

    if column in header:
        col = header.index(column)
        for row, val in zip(rows, encoded):
            row[col] = val
    else:
        header.append(column)
        for row, val in zip(rows, encoded):
            row.append(val)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def set_score_cell(scores_file: str, run_name: str, src_lang: str, tgt_lang: str,
                   column: str, value: str) -> None:
    """Fills the XCOMET column of the matching row in a per-run scores CSV."""
    with open(scores_file, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    col = header.index(column)
    i_run, i_src, i_tgt = header.index("Run"), header.index("Source_Lang"), header.index("Target_Lang")
    for row in rows:
        if row[i_run] == run_name and row[i_src] == src_lang and row[i_tgt] == tgt_lang:
            row[col] = value

    with open(scores_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

# ============================================================================
# Main
# ============================================================================

SRC_LIST = [
    ("poj", "en"),    # in-distribution for the fine-tuned model
    ("en", "poj"),
    ("hanlo", "en"),    # in-distribution for the fine-tuned model
    ("en", "hanlo"),
    # ("poj", "zh"),    # the rest are zero-shot
    # ("tl", "zh"),
    # ("han", "zh"),
    # ("tl", "en"),
    # ("tl", "han"),
    # ("zh", "poj"),
    # ("zh", "tl"),
    # ("zh", "han"),
    # ("en", "tl"),
    # ("han", "tl"),
]

SCORE_HEADER = [
    "Run", "Model_Path", "Prompt_Mode", "Source_Lang", "Target_Lang",
    "BLEU_Score", "chrF2_Score", "XCOMET_XL", "BLEU_Tokenizer",
    "N", "Well_Formed", "Unclosed", "Untagged", "Empty",
]


def main(out_dir: str = f"qwen/{MODEL_TAG}") -> None:
    global PROMPT_MODE

    summary_rows: list[list] = []
    xcomet_jobs: list[dict] = []   # filled in pass 1, scored in pass 2

    # ---------------------------------------------------------------- pass 1
    for run_name, model_path, subfolder, mode in RUNS:
        if (not USE_HF_HUB and SKIP_MISSING_CKPT
                and not os.path.isdir(model_path)):
            print(f"\n[skip] {run_name}: {model_path} not found")
            continue

        PROMPT_MODE = mode
        model_id = f"{model_path}/{subfolder}" if subfolder else model_path

        print(f"\n{'#' * 60}")
        print(f"# {run_name} | {mode} | {model_id}")
        print(f"{'#' * 60}")

        scores_file = f"./{out_dir}/evaluation_scores_{run_name}.csv"
        os.makedirs(os.path.dirname(scores_file), exist_ok=True)
        with open(scores_file, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(SCORE_HEADER)

        tokenizer, model, device = setup_translator(model_path, subfolder)

        for src_lang, tgt_lang in SRC_LIST:
            path = f"./input/raw_{src_lang}_{tgt_lang}/{src_lang}-{tgt_lang}"
            output_file = f"./{out_dir}/{src_lang}_{tgt_lang}_translation_{run_name}.csv"

            print(f"\n{'=' * 40}")
            print(f"{src_lang} -> {tgt_lang}  [{run_name}]")
            print(f"{'=' * 40}")

            sources, references = load_data(f"{path}.{src_lang}", f"{path}.{tgt_lang}")

            predictions, stats = translate_and_save(
                tokenizer, model, device, sources, references,
                output_file, src_lang, tgt_lang,
            )
            bleu, chrf, bleu_tok = evaluate_translations(predictions, references, tgt_lang)

            row = [
                run_name, model_id, mode, src_lang, tgt_lang,
                f"{bleu:.2f}", f"{chrf:.2f}", "", bleu_tok,
                stats["n"], stats["ok"], stats["no_close"],
                stats["no_tag"], stats["empty"],
            ]
            with open(scores_file, "a", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow(row)
            summary_rows.append(row)

            if XCOMET_ENABLED and tgt_lang in XCOMET_TARGET_LANGS:
                xcomet_jobs.append({
                    "run_name": run_name,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "translation_file": output_file,
                    "scores_file": scores_file,
                    "row": row,
                })

        print(f"\nRun '{run_name}' done -> {scores_file}")

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- pass 2
    if xcomet_jobs:
        comet_model = load_xcomet()
        if comet_model is not None:
            for job in xcomet_jobs:
                print(f"\nXCOMET-XL: {job['run_name']} "
                      f"{job['src_lang']} -> {job['tgt_lang']}")
                preds, refs = read_translation_csv(job["translation_file"])
                system_score, seg_scores, spans = score_xcomet(comet_model, preds, refs)

                n_flagged = sum(1 for s in spans if s)
                print(f"XCOMET-XL system score: {system_score:.4f}  (0-1 scale)")
                print(f"Segments with error spans: {n_flagged}/{len(spans)}")

                append_segment_scores(job["translation_file"], "XCOMET_XL", seg_scores)
                append_error_spans(job["translation_file"], spans)
                set_score_cell(job["scores_file"], job["run_name"],
                               job["src_lang"], job["tgt_lang"],
                               "XCOMET_XL", f"{system_score:.4f}")
                job["row"][SCORE_HEADER.index("XCOMET_XL")] = f"{system_score:.4f}"

            del comet_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------- summary
    summary_file = f"./{out_dir}/evaluation_scores_all.csv"
    with open(summary_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SCORE_HEADER)
        writer.writerows(summary_rows)

    print(f"\nAll runs completed -> {summary_file}")


if __name__ == "__main__":
    main()