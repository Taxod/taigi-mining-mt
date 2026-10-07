import argparse

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

import utils as utils

### MODIFY PATH ###
PRETRAINING_PATH = ''

EMBEDDING_SIZE = 768
LAYER = 8


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='Sentence file (format: <ID>\\t<sentence>)')
    parser.add_argument('-o', '--output_file', type=str, required=True,
                        help='Output file path')
    parser.add_argument('-m', '--model_name', type=str, required=True,
                        choices=['xlmr', 'glot500', 'pretrained', 'LaBSE'],
                        help='Embedding model')
    parser.add_argument('-mp', '--model_path', type=str, default=None,
                        help='Path to the model if model_name is pretrained')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size for embedding (default: 64)')
    return parser.parse_args()


def resolve_model_name(model_name, model_path=None):
    """Map the CLI model name to a HuggingFace model id / local path."""
    if model_name == 'xlmr':
        return 'xlm-roberta-base'
    elif model_name == 'glot500':
        return 'cis-lmu/glot500-base'
    elif model_name == 'pretrained':
        if model_path:
            print(f'Using a pretrained model from: {model_path}')
            return model_path
        print(f'Using a pretrained model from: {PRETRAINING_PATH}')
        return PRETRAINING_PATH
    raise ValueError(f'Unknown model name: {model_name}')


def parse_id_sentence_lines(lines):
    """Parse '<ID>\\t<sentence>' lines into (ids, sentences)."""
    ids, sentences = [], []
    for line_no, line in enumerate(lines, 1):
        split_sentence = line.split('\t')
        assert len(split_sentence) == 2, (
            f'Line {line_no} does not have exactly 2 tab-separated fields '
            f'(got {len(split_sentence)}): {line[:80]!r}'
        )
        ids.append(split_sentence[0])
        # Guard against empty sentences: the tokenizer would still emit
        # <s></s>, and masking both out would leave zero tokens.
        sentences.append(split_sentence[1] if split_sentence[1].strip() else '.')
    return ids, sentences


def format_embedding_line(sent_id, embedding):
    str_embedding = ' '.join(f'{v:.6f}' for v in embedding)
    return f'{sent_id} {str_embedding}'


# ---------------------------------------------------------------------------
# XLM-R / Glot500 / custom pretrained models (masked mean pooling on layer 8)
# ---------------------------------------------------------------------------

def embed_batch_xlmr(sentences, tokenizer, model, device, layer=LAYER):
    """
    Embed a batch of sentences with masked mean pooling over hidden layer
    `layer`, excluding <s>, </s>, and all padding tokens.

    Equivalent (up to float rounding) to the original single-sentence
    outputs[:, 1:-1, :].mean(axis=0) behaviour, but correct under padding.
    """
    inputs = tokenizer(
        sentences,
        is_split_into_words=False,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        hidden = model(**inputs)["hidden_states"]
        if layer >= len(hidden):
            raise ValueError(
                f"Specified to take embeddings from layer {layer}, "
                f"but model has only {len(hidden)} layers."
            )
        outputs = hidden[layer]  # (B, L, H)

    attention_mask = inputs["attention_mask"]          # (B, L)
    mask = attention_mask.clone()

    # Zero out <s> (position 0) and each sentence's own </s> (last real token).
    mask[:, 0] = 0
    seq_lens = attention_mask.sum(dim=1)               # real lengths incl. specials
    batch_idx = torch.arange(mask.size(0), device=mask.device)
    mask[batch_idx, seq_lens - 1] = 0

    mask = mask.unsqueeze(-1).to(outputs.dtype)        # (B, L, 1)
    summed = (outputs * mask).sum(dim=1)               # (B, H)
    counts = mask.sum(dim=1).clamp(min=1.0)            # (B, 1)
    mean_pooled = summed / counts

    return mean_pooled.cpu().numpy()                   # (B, H)


def to_xlmr_sentence_embeddings(path, sentence_list, model_name,
                                model_path=None, batch_size=64):
    """Save XLM-R-style embeddings in a txt file (same format as fastText)."""
    resolved_name = resolve_model_name(model_name, model_path)

    device = get_device()
    print(f"*** Currently using device: {device} ***")

    config = AutoConfig.from_pretrained(resolved_name, output_hidden_states=True)
    model = AutoModel.from_pretrained(resolved_name, config=config)
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(resolved_name)

    ids, sentences = parse_id_sentence_lines(sentence_list)
    n = len(sentences)

    with open(path, 'w', encoding='utf8') as out_text:
        out_text.write(f'{n} {EMBEDDING_SIZE}\n')

        buffer = []
        for start in tqdm(range(0, n, batch_size)):
            batch_ids = ids[start:start + batch_size]
            batch_sents = sentences[start:start + batch_size]

            embeddings = embed_batch_xlmr(batch_sents, tokenizer, model, device)
            assert embeddings.shape[1] == EMBEDDING_SIZE, (
                f'The embedding size is different: {embeddings.shape[1]}'
            )

            for sent_id, emb in zip(batch_ids, embeddings):
                buffer.append(format_embedding_line(sent_id, emb))

            if len(buffer) >= 10000:
                out_text.write('\n'.join(buffer) + '\n')
                buffer = []

        if buffer:
            out_text.write('\n'.join(buffer) + '\n')


# ---------------------------------------------------------------------------
# LaBSE (SentenceTransformer has correct built-in batching)
# ---------------------------------------------------------------------------

def to_labse_sentence_embeddings(path, sentence_list, batch_size=64):
    device = get_device()
    print(f"*** Currently using device: {device} ***")

    labse_model = SentenceTransformer('sentence-transformers/LaBSE', device=str(device))

    ids, sentences = parse_id_sentence_lines(sentence_list)
    n = len(sentences)

    with open(path, 'w', encoding='utf8') as out_text:
        out_text.write(f'{n} {EMBEDDING_SIZE}\n')

        buffer = []
        chunk_size = batch_size * 100  # encode in larger chunks, write periodically
        for start in tqdm(range(0, n, chunk_size)):
            chunk_ids = ids[start:start + chunk_size]
            chunk_sents = sentences[start:start + chunk_size]

            embeddings = labse_model.encode(
                chunk_sents,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            assert embeddings.shape[1] == EMBEDDING_SIZE, (
                f'The embedding size is different: {embeddings.shape[1]}'
            )

            for sent_id, emb in zip(chunk_ids, embeddings):
                buffer.append(format_embedding_line(sent_id, emb))

            out_text.write('\n'.join(buffer) + '\n')
            buffer = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.input_file, 'r', encoding='utf8') as f:
        input_file = f.read()
    split_file = utils.text_to_line(input_file)

    model_name = args.model_name
    print(f'Model to use: {model_name}')
    print(f'Batch size: {args.batch_size}')
    print(f'Total sentences: {len(split_file)}')

    if model_name in ['xlmr', 'glot500', 'pretrained']:
        to_xlmr_sentence_embeddings(
            args.output_file, split_file, model_name,
            model_path=args.model_path, batch_size=args.batch_size,
        )
    elif model_name == 'LaBSE':
        to_labse_sentence_embeddings(
            args.output_file, split_file, batch_size=args.batch_size,
        )
    return 0


if __name__ == '__main__':
    main()