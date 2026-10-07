import os
import csv
import unicodedata

import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

BASE_NLLB = "facebook/nllb-200-distilled-600M"

# Fine-tuned repos whose checkpoint-N subfolders hold the models. The subfolder
# is fetched to a local path so from_pretrained can be pointed straight at it.
FT_REPO_A = "w-y-kuo/nllb_finetuned_080"
FT_STEP_A = 3000

FT_REPO_B = "w-y-kuo/nllbcpt"
FT_STEP_B = 46500

# Inference needs the weights, the configs and the tokenizer; optimizer.pt,
# scheduler.pt, rng_state.pth and trainer_state.json are training state only.
CKPT_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]

BATCH_SIZE = 256
MAX_NEW_TOKENS = 1024
NUM_BEAMS = 1  # NLLB's own config defaults to 4; 1 is greedy

# Same columns as the Qwen script, so both summaries share one schema and the
# separate XCOMET pass can fill XCOMET_XL in place.
SCORE_HEADER = [
    "Run", "Model_Path", "Prompt_Mode", "Source_Lang", "Target_Lang",
    "BLEU_Score", "chrF2_Score", "XCOMET_XL", "BLEU_Tokenizer",
    "N", "Well_Formed", "Unclosed", "Untagged", "Empty",
]

TRANSLATION_HEADER = ["Source", "Reference", "Prediction", "Prediction_Raw", "Status"]


def fetch_checkpoint(repo_id: str, step: int) -> str:
    """Downloads one checkpoint-N subfolder and returns its local path."""
    from huggingface_hub import snapshot_download

    sub = f"checkpoint-{step}"
    root = snapshot_download(
        repo_id,
        allow_patterns=[f"{sub}/{p}" for p in CKPT_PATTERNS],
    )
    path = os.path.join(root, sub)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"'{sub}' not found in {repo_id}")
    return path


def normalize(text: str) -> str:
    """NFC. POJ/Tai-lo tone marks compare unequal across NFC/NFD."""
    return unicodedata.normalize("NFC", text)


def get_bleu_tokenizer(tgt_lang: str) -> str:
    """'zh' collapses a romanized sentence into one token, so Latin targets need '13a'."""
    if tgt_lang in ("zh", "han", "hanlo"):
        return "zh"
    if tgt_lang in ("en", "poj", "tl"):
        return "13a"
    return "char"


def load_data(source_file: str, reference_file: str) -> tuple[list[str], list[str]]:
    """
    Reads source and reference files, removes empty lines, and validates lengths.
    """
    for path in (source_file, reference_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: '{path}'")

    with open(source_file, "r", encoding="utf-8") as f:
        sources = [normalize(line.strip()) for line in f if line.strip()]

    with open(reference_file, "r", encoding="utf-8") as f:
        references = [normalize(line.strip()) for line in f if line.strip()]

    if len(sources) != len(references):
        raise ValueError(f"Data mismatch! Sources: {len(sources)} lines, References: {len(references)} lines.")

    return sources, references


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_translator(src_lang,
                     tgt_lang,
                     model_name: str = BASE_NLLB):
    """
    Loads an NLLB model directly: transformers 5.x dropped the `translation`
    pipeline, so generate() is called by hand.

    src_lang and tgt_lang are passed through as given: the source codes used
    with the base model are deliberately not FLORES-200 codes, so they resolve
    to the unknown-token id.
    """
    device = pick_device()
    print(f"Using device: {device}  |  model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = src_lang

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    # NLLB steers the target language through the first decoder token.
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    print(f"src_lang={src_lang} -> id {tokenizer.convert_tokens_to_ids(src_lang)}  |  "
          f"tgt_lang={tgt_lang} -> id {forced_bos}  (unk id {tokenizer.unk_token_id})")

    return tokenizer, model, forced_bos


def translate_and_save(translator,
                       sources: list[str],
                       references: list[str],
                       output_file: str,
                       batch_size: int = BATCH_SIZE) -> tuple[list[str], dict]:
    """
    Batched translation. Writes the same five columns as the Qwen script.
    """
    tokenizer, model, forced_bos = translator

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Length-sorted batching: shorter inputs stop padding the long ones.
    # Results are written back through `order`, so the file keeps input order.
    order = sorted(
        range(len(sources)),
        key=lambda i: len(tokenizer(sources[i], add_special_tokens=False)["input_ids"]),
    )

    predictions: list[str] = [""] * len(sources)

    with torch.no_grad():
        for start in tqdm(range(0, len(order), batch_size), desc="Translating"):
            idx = order[start:start + batch_size]
            inputs = tokenizer([sources[i] for i in idx],
                               return_tensors="pt", padding=True,
                               truncation=True, max_length=1024).to(model.device)

            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                do_sample=False,
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i, raw in zip(idx, decoded):
                predictions[i] = normalize(raw.strip())

    statuses = ["empty" if not p else "raw" for p in predictions]

    with open(output_file, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(TRANSLATION_HEADER)
        # NLLB emits no wrapper to strip, so Prediction_Raw repeats Prediction
        # and Status only records whether anything came out.
        writer.writerows(
            (s, r, p, p, st)
            for s, r, p, st in zip(sources, references, predictions, statuses)
        )

    stats = {
        "n": len(predictions),
        "ok": 0,
        "no_close": 0,
        "no_tag": 0,
        "empty": statuses.count("empty"),
    }

    print(f"\nTranslation finished! Results saved to {output_file}")
    print(f"Output: {stats['n']} sentences, {stats['empty']} empty")
    return predictions, stats


def evaluate_translations(predictions: list[str], references: list[str],
                          tgt_lang: str) -> tuple[float, float, str]:
    """
    Evaluates the translated predictions against the references using SacreBLEU
    and returns the BLEU score, the chrF2 score and the tokenizer used.
    """
    print("\n--- Evaluation Results ---")

    bleu_tok = get_bleu_tokenizer(tgt_lang)
    print(f"Using Tokenizer: '{bleu_tok}' (target: {tgt_lang})")

    bleu_result = sacrebleu.corpus_bleu(predictions, [references], tokenize=bleu_tok)
    print(f"BLEU Score: {bleu_result.score:.2f}")

    chrf_result = sacrebleu.corpus_chrf(predictions, [references])
    print(f"chrF2 Score: {chrf_result.score:.2f}")

    return bleu_result.score, chrf_result.score, bleu_tok


if __name__ == "__main__":
    # (src, tgt, src_code, tgt_code, repo_id, step, run_tag).
    # src_code / tgt_code are the language codes handed to the tokenizer;
    # step is None for the base model since it does not have checkpoints.
    src_list = [
        # ("poj", "zh", "poj", "zho_Hant", BASE_NLLB, None, "base"),
        # ("poj", "en", "poj", "eng_Latn", BASE_NLLB, None, "base"),
        # ("tl", "zh", "tl", "zho_Hant", BASE_NLLB, None, "base"),
        # ("han", "zh", "han", "zho_Hant", BASE_NLLB, None, "base"),
        # ("tl", "en", "tl", "eng_Latn", BASE_NLLB, None, "base"),
        # ("tl", "han", "tl", "zho_Hant", BASE_NLLB, None, "base"),
        # ("hanlo", "en", "hanlo", "eng_Latn", BASE_NLLB, None, "base"),
        # ("hanlo", "zh", "hanlo", "zho_Hant", BASE_NLLB, None, "base"),

        # Fine-tuned models
        # ("hanlo", "en", "poj_Latn", "eng_Latn", FT_REPO_A, FT_STEP_A, f"ft_a_ckpt{FT_STEP_A}"),
        ("hanlo", "en", "poj_Latn", "eng_Latn", FT_REPO_B, FT_STEP_B, f"ft_b_ckpt{FT_STEP_B}"),

        # ("poj", "zh", "poj_Latn", "zho_Hant", FT_REPO_A, FT_STEP_A, f"ft_a_ckpt{FT_STEP_A}"),
        # ("poj", "en", "poj_Latn", "eng_Latn", FT_REPO_A, FT_STEP_A, f"ft_a_ckpt{FT_STEP_A}"),
        # ("en", "poj", "eng_Latn", "poj_Latn", FT_REPO_A, FT_STEP_A, f"ft_a_ckpt{FT_STEP_A}"),
        # ("zh", "poj", "zho_Hant", "poj_Latn", FT_REPO_A, FT_STEP_A, f"ft_a_ckpt{FT_STEP_A}"),

        ("poj", "zh", "poj_Latn", "zho_Hant", FT_REPO_B, FT_STEP_B, f"ft_b_ckpt{FT_STEP_B}"),
        ("poj", "en", "poj_Latn", "eng_Latn", FT_REPO_B, FT_STEP_B, f"ft_b_ckpt{FT_STEP_B}"),
        ("en", "poj", "eng_Latn", "poj_Latn", FT_REPO_B, FT_STEP_B, f"ft_b_ckpt{FT_STEP_B}"),
        ("zh", "poj", "zho_Hant", "poj_Latn", FT_REPO_B, FT_STEP_B, f"ft_b_ckpt{FT_STEP_B}"),
    ]

    for src_lang, tgt_lang, src_code, tgt_code, repo_id, step, run_tag in src_list:
        # Extract folder name after the slash, e.g., "nllb_finetuned_080"
        model_folder = repo_id.split("/")[-1]
        out_dir = f"./nllb/{model_folder}"
        os.makedirs(out_dir, exist_ok=True)

        scores_file = f"{out_dir}/evaluation_scores.csv"
        # Initialize the scores file with headers if it doesn't exist
        if not os.path.exists(scores_file):
            with open(scores_file, mode='w', encoding='utf-8-sig', newline='') as f:
                csv.writer(f).writerow(SCORE_HEADER)

        path = f"./input/raw_{src_lang}_{tgt_lang}/{src_lang}-{tgt_lang}"
        source_file = f"{path}.{src_lang}"
        reference_file = f"{path}.{tgt_lang}"
        output_file = f"{out_dir}/{src_lang}_{tgt_lang}_translation_{run_tag}.csv"

        if not (os.path.exists(source_file) and os.path.exists(reference_file)):
            print(f"\n[skip] {src_lang} -> {tgt_lang} [{run_tag}]: input files not found")
            continue

        print(f"\n{'=' * 40}")
        print(f"Start translating {src_lang} -> {tgt_lang} [{run_tag}]")
        print(f"{'=' * 40}")

        sources, references = load_data(source_file, reference_file)
        print(f"Loaded {len(sources)} sentences for translation and evaluation.")

        # Determine the local model path
        model_path = fetch_checkpoint(repo_id, step) if step is not None else repo_id

        translator = setup_translator(src_code, tgt_code, model_path)
        predictions, stats = translate_and_save(translator, sources, references, output_file)

        del translator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bleu_score, chrf2_score, bleu_tok = evaluate_translations(
            predictions, references, tgt_lang
        )

        row = [
            run_tag, model_path, f"{src_code}->{tgt_code}", src_lang, tgt_lang,
            f"{bleu_score:.2f}", f"{chrf2_score:.2f}", "", bleu_tok,
            stats["n"], stats["ok"], stats["no_close"],
            stats["no_tag"], stats["empty"],
        ]

        with open(scores_file, mode='a', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerow(row)

    print("\nAll tasks completed!")