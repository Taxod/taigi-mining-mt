import os
import csv
import time
import sacrebleu
from tqdm import tqdm
from openai import OpenAI

CUSTOM_PROMPTS = {
    ("poj", "zh"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Taiwanese Hokkien Pe̍h-ōe-jī (POJ) text into Mandarin (Traditional Chinese characters). "
        "Output ONLY the translation without any explanations."
    ),
    ("poj", "en"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Taiwanese Hokkien Pe̍h-ōe-jī (POJ) text into English. "
        "Output ONLY the translation without any explanations."
    ),
    ("tl", "zh"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案) text into Mandarin (Traditional Chinese characters). "
        "Output ONLY the translation without any explanations."
    ),
    ("han", "zh"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Taiwanese Hokkien Han characters (Taigi Hanji) into Mandarin (Traditional Chinese characters). "
        "Output ONLY the translation without any explanations."
    ),
    ("tl", "en"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案) text into English. "
        "Output ONLY the translation without any explanations."
    ),
    ("tl", "han"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案) text into Taiwanese Hokkien Han characters (Taigi Hanji). "
        "Output ONLY the translation without any explanations."
    ),
    ("zh", "poj"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Mandarin (Traditional Chinese characters) text into Taiwanese Pe̍h-ōe-jī (POJ). "
        "Output ONLY the translation without any explanations."
    ),
    ("zh", "tl"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Mandarin (Traditional Chinese characters) text into the Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案). "
        "Output ONLY the translation without any explanations."
    ),
    ("zh", "han"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Mandarin (Traditional Chinese characters) text into Taiwanese Han characters (Taigi Hanji). "
        "Output ONLY the translation without any explanations."
    ),
    ("en", "poj"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following English text into Taiwanese Pe̍h-ōe-jī (POJ). "
        "Output ONLY the translation without any explanations."
    ),
    ("en", "tl"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following English text into Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案). "
        "Output ONLY the translation without any explanations."
    ),
    ("han", "tl"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Taiwanese Hokkien Han characters (Taigi Hanji) into Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案). "
        "Output ONLY the translation without any explanations."
    ),
    ("te", "st"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following Mandarin (Traditional Chinese characters) text into Taiwanese Han characters (Taigi Hanji). "
        "Output ONLY the translation without any explanations."
    ),("en", "hanlo"): (
        "You are a linguistic expert in Taiwanese Hokkien. Translate the following English text into Taiwanese Hokkien Hàn-lô Tâi-bûn, Hàn-lô (Hàn-lô). "
        "Ensure correct use of tone diacritics and use hyphens to connect syllables. Output ONLY the translation without any explanations."
    ), ("hanlo",
        "en"): "You are a linguistic expert. Translate the following Taiwanese Hokkien Hàn-lô Tâi-bûn, Hàn-lô (Hàn-lô) text into English. Output ONLY the translation without any explanations.",

}


def get_tokenizer(filename: str) -> str:
    """
    Determine the correct BLEU tokenizer based on the target language
    inferred from the filename. Supports _reverse and _custom suffixes.
    """
    name = os.path.basename(filename).lower()

    # Strip known suffixes to normalize the filename
    normalized = name.replace(".csv", "")
    for suffix in ["_custom", "_reverse", "_gemini"]:
        normalized = normalized.replace(suffix, "")

    # Extract target language from normalized name
    parts = normalized.replace("_translation", "").split("_")
    tgt_lang = parts[-1] if parts else ""

    if tgt_lang in ["zh", "han","hanlo"]:
        return 'zh'
    elif tgt_lang in ["en", "poj", "tl"]:
        return '13a'
    else:
        return 'char'  # Fallback

def get_bleu_tokenizer(tgt_lang: str) -> str:
    if tgt_lang in ("zh", "han", "hanlo"):
        return "zh"
    if tgt_lang in ("en", "poj", "tl"):
        return "13a"
    return "char"
def get_language_name(lang_code: str) -> str:
    """
    Helper function to map language codes to explicit language names for the fallback prompt.
    """
    mapping = {
        "zh": "Taiwanese Traditional Chinese",
        "en": "English",
        "poj": "Taiwanese Hokkien Pe̍h-ōe-jī (POJ)",
        "tl": "Taiwanese Romanization System (Tâi-lô, 臺灣台語羅馬字拼音方案)",
        "han": "Taiwanese Hokkien Han characters (Taigi Hanji)",
        "hanlo": "Taiwanese Hokkien Hàn-lô Tâi-bûn, Hàn-lô (Hàn-lô)",
    }
    return mapping.get(lang_code, lang_code)


def load_data(source_file: str, reference_file: str) -> tuple[list[str], list[str]]:
    """
    Reads source and reference files, removes empty lines, and validates lengths.
    """
    if not os.path.exists(source_file) or not os.path.exists(reference_file):
        raise FileNotFoundError(f"Ensure both '{source_file}' and '{reference_file}' exist.")

    with open(source_file, "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f if line.strip()]

    with open(reference_file, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f if line.strip()]

    if len(sources) != len(references):
        raise ValueError(f"Data mismatch! Sources: {len(sources)} lines, References: {len(references)} lines.")

    return sources, references


def setup_openrouter_client():
    """
    Initializes the OpenAI client configured for OpenRouter.
    Ensure OPENROUTER_API_KEY is set in your environment variables.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client


def translate_and_save(client: OpenAI,
                       model_name: str,
                       sources: list[str],
                       references: list[str],
                       output_file: str,
                       src_lang: str,
                       tgt_lang: str) -> list[str]:
    """
    Translates sentences one by one using OpenRouter API.
    """
    predictions = []

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    src_lang_name = get_language_name(src_lang)
    tgt_lang_name = get_language_name(tgt_lang)
    fallback_prompt = f"You are a professional translator. Translate the following text from {src_lang_name} to {tgt_lang_name}. Output ONLY the translation without any explanations or additional text."

    system_prompt = CUSTOM_PROMPTS.get((src_lang, tgt_lang), fallback_prompt)

    with open(output_file, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Reference", "Prediction"])

        iterator = tqdm(zip(sources, references), total=len(sources), desc=f"Translating {src_lang} -> {tgt_lang}")

        for orig, ref in iterator:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": orig}
            ]

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,  # Deterministic outputs for translation
                    max_tokens=512
                )
                pred = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"\nAPI Error during translation of '{orig}': {e}")
                pred = ""  # Handle error gracefully, or you can implement a retry mechanism here

            predictions.append(pred)
            writer.writerow([orig, ref, pred])

            # Rate limiting for free tier models (adjust as needed based on OpenRouter limits)
            # time.sleep(1)

    print(f"\nTranslation finished! Results saved to {output_file}")
    return predictions


def evaluate_translations(predictions: list[str], references: list[str],
                          tgt_lang: str) -> tuple[float, float]:
    """
    Evaluates the translated predictions against the references using SacreBLEU.
    Dynamically selects the tokenizer based on the output filename.
    """
    print("\n--- Evaluation Results ---")

    # 1. 使用新增的函式來決定 Tokenizer
    tokenizer_type = get_bleu_tokenizer(tgt_lang)
    print(f"Using Tokenizer: '{tokenizer_type}' for {tgt_lang}")

    # 2. 將 tokenizer_type 傳入 sacrebleu.corpus_bleu
    bleu_result = sacrebleu.corpus_bleu(predictions, [references], tokenize=tokenizer_type)
    print(f"BLEU Score: {bleu_result.score:.2f}")

    # chrF2 的計算原本就不依賴詞彙級別的 tokenizer，所以保持原樣即可
    chrf_result = sacrebleu.corpus_chrf(predictions, [references])
    print(f"chrF2 Score: {chrf_result.score:.2f}")

    return bleu_result.score, chrf_result.score


if __name__ == "__main__":
    postfix = "gemini"
    folder_name = "gemini-flashLite"
    src_list = [
        ("poj", "zh", "zho_Hant"),
        ("poj", "en", "eng_Latn"),
        ("tl", "zh", "zho_Hant"),
        ("en", "hanlo", "nan_Latn"),
        ("hanlo", "en", "eng_Latn"),
        ("han", "zh", "zho_Hant"),
        ("tl", "en", "eng_Latn"),
        ("tl", "han", "zho_Hant"),
        ("zh", "poj", "nan_Latn"),
        ("en", "poj", "nan_Latn"),
        ("zh", "tl", "nan_Latn"),
        ("zh", "han", "nan_Hant"),
        ("en", "tl", "nan_Latn"),
        ("han", "tl", "nan_Latn"),
        # ("te", "st", "nan_Hant"),
    ]

    OPENROUTER_MODEL = "google/gemini-3.1-flash-lite"

    scores_file = f"./{folder_name}/evaluation_scores_{postfix}.csv"
    os.makedirs(os.path.dirname(scores_file), exist_ok=True)

    with open(scores_file, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Source_Lang", "Target_Lang", "BLEU_Score", "chrF2_Score"])

    # Initialize client once
    client = setup_openrouter_client()

    for src_lang, tgt_lang, tgt_iso in src_list:
        path = f"./input/raw_{src_lang}_{tgt_lang}/{src_lang}-{tgt_lang}"
        source_file = f"{path}.{src_lang}"
        reference_file = f"{path}.{tgt_lang}"
        output_file = f"./{folder_name}/{src_lang}_{tgt_lang}_translation_{postfix}.csv"

        print(f"\n{'=' * 40}")
        print(f"Start translating {src_lang} -> {tgt_lang} using {OPENROUTER_MODEL}")
        print(f"{'=' * 40}")

        sources, references = load_data(source_file, reference_file)

        predictions = translate_and_save(
            client, OPENROUTER_MODEL,
            sources, references, output_file,
            src_lang, tgt_lang
        )

        bleu_score, chrf2_score = evaluate_translations(predictions, references, tgt_lang)
        with open(scores_file, mode='a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([src_lang, tgt_lang, f"{bleu_score:.2f}", f"{chrf2_score:.2f}"])

    print(f"\nAll tasks completed! Summary scores saved to {scores_file}")
