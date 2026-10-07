from transformers import NllbTokenizer, M2M100ForConditionalGeneration, AutoConfig
from tokenizers import AddedToken
import torch
from pathlib import Path
import os
import json
from collections import Counter

def add_language_token(tokenizer, new_lang_code):
	"""
	Register a new language tag with the NLLB tokenizer
	"""
	if new_lang_code in tokenizer.get_vocab():
		print(f"Language {new_lang_code} already exists in tokenizer")
		return tokenizer

	print(f"Adding new language token: {new_lang_code}")
	tokenizer.add_tokens([AddedToken(new_lang_code, special=True, normalized=False)])
	new_id = tokenizer.convert_tokens_to_ids(new_lang_code)
	if new_id == tokenizer.unk_token_id:
		raise RuntimeError(f"{new_lang_code} was not registered as a token")
	table = getattr(tokenizer, "lang_code_to_id", None)
	if isinstance(table, dict):
		table[new_lang_code] = new_id
	print(f"  -> id {new_id}, len(tokenizer)={len(tokenizer)}")
	return tokenizer

def scan_characters(tokenizer, corpus_path, limit=50000):
	"""
	Report the characters in the corpus that the tokenizer cannot reproduce.

	A character is lost either because it maps to <unk>, or because
	SentencePiece's normalizer silently rewrites it before the vocabulary
	lookup, which is what happens to U+207F: it becomes a plain "n".  A
	tokenize/decode round trip catches both.
	"""
	if corpus_path is None:
		return Counter()
	char_freq = Counter()
	with open(corpus_path, encoding="utf-8") as fh:
		for n, line in enumerate(fh):
			if n >= limit:
				break
			record = json.loads(line)
			for text in record["translation"].values():
				if text:
					char_freq.update(text)

	lost = Counter()
	for ch, count in char_freq.items():
		if ch.isspace():
			continue
		ids = tokenizer(ch, add_special_tokens=False).input_ids
		if not ids or tokenizer.unk_token_id in ids:
			lost[ch] = count
		elif tokenizer.decode(ids).strip() != ch.strip():
			lost[ch] = count

	for ch, count in lost.most_common():
		ids = tokenizer(ch, add_special_tokens=False).input_ids
		how = "unk" if (not ids or tokenizer.unk_token_id in ids) else \
		      f"normalized to {tokenizer.decode(ids).strip()!r}"
		print(f"  lost: U+{ord(ch):04X} x{count} ({how})")
	return lost

def initialize_embeddings(model, tokenizer, new_lang_code, similar_lang_code,
                          new_chars, char_donors):
	"""
	Resize the embedding matrix and initialize every newly added token
	"""
	original_rows = model.get_input_embeddings().weight.shape[0]
	model.resize_token_embeddings(len(tokenizer))

	with torch.no_grad():
		embeddings = model.get_input_embeddings().weight

		def assign(token, donor):
			token_id = tokenizer.convert_tokens_to_ids(token)
			donor_id = tokenizer.convert_tokens_to_ids(donor) if donor else tokenizer.unk_token_id
			if not donor or donor_id == tokenizer.unk_token_id:
				print(f"Kept {token!r}({token_id}) at the resize initialization")
				return
			embeddings[token_id] = embeddings[donor_id].clone()
			print(f"Initialized {token!r}({token_id}) from {donor}({donor_id})")

		if tokenizer.convert_tokens_to_ids(similar_lang_code) == tokenizer.unk_token_id:
			raise RuntimeError(f"{similar_lang_code} is not supported by this model")
		assign(new_lang_code, similar_lang_code)
		for ch in new_chars:
			assign(ch, char_donors.get(ch))

		out = model.get_output_embeddings()
		tied = out is not None and out.weight.data_ptr() == embeddings.data_ptr()
		if out is not None and not tied:
			for token in [new_lang_code, *new_chars]:
				token_id = tokenizer.convert_tokens_to_ids(token)
				out.weight[token_id] = embeddings[token_id].clone()

	print(f"Embedding rows {original_rows} -> {model.get_input_embeddings().weight.shape[0]}, "
	      f"config.tie_word_embeddings={model.config.tie_word_embeddings}, actually tied={tied}")

def verify(tokenizer, lang_code, samples):
	"""
	Check that the language tag is used as the source prefix and that the
	sample sentences survive a tokenize/decode round trip
	"""
	tokenizer.src_lang = lang_code
	ids = tokenizer("test").input_ids
	expected = tokenizer.convert_tokens_to_ids(lang_code)
	print(f"[verify] src_lang={lang_code} prefix={ids[0]} expected={expected} "
	      f"-> {'ok' if ids[0] == expected else 'WRONG'}")
	if ids[0] != expected:
		raise RuntimeError("the added language token is not used as the source prefix")
	for text in samples:
		piece_ids = tokenizer(text, add_special_tokens=False).input_ids
		back = tokenizer.decode(piece_ids, skip_special_tokens=False)
		if back.strip() == text.strip():
			status = "ok"
		elif "".join(back.split()) == "".join(text.split()):
			# An added token splits the string, so the following piece is encoded
			# as a new word and decodes with a leading space.  No character is
			# lost, and hypotheses and references pick it up alike, so metrics
			# are unaffected; only final output needs the space removed.
			status = "spacing"
		else:
			status = "LOSSY"
		print(f"[verify] {status:7s} {len(piece_ids):3d} pieces | {text} -> {back}")

def prepare_nllb_model(model_name, new_langs, similar_langs, corpus_path, extra_tokens, char_donors):
	"""
	Prepare NLLB model with new language tags
	"""
	print(f"Loading model and tokenizer from {model_name}")

	# Load base model and tokenizer
	tokenizer = NllbTokenizer.from_pretrained(model_name)
	config = AutoConfig.from_pretrained(model_name)
	model = M2M100ForConditionalGeneration.from_pretrained(model_name, config=config)

	# Add new languages
	for new_lang in new_langs:
		tokenizer = add_language_token(tokenizer, new_lang)

	# Report every character the tokenizer loses, but only add the POJ ones.
	# Everything else in that report is corpus noise; turning it into vocabulary
	# would add untrained embedding rows the model can emit at generation time.
	lost = scan_characters(tokenizer, corpus_path)
	new_chars = [ch for ch in extra_tokens if ch not in tokenizer.get_vocab()]
	untouched = [ch for ch in lost if ch not in extra_tokens]
	if untouched:
		print(f"{len(untouched)} lost characters left as <unk>, "
		      f"{sum(lost[c] for c in untouched)} occurrences")
	if new_chars:
		tokenizer.add_tokens([AddedToken(ch, special=False, normalized=False) for ch in new_chars])
		print(f"Added {len(new_chars)} character tokens: "
		      + ", ".join(f"U+{ord(c):04X}" for c in new_chars))

	for new_lang, similar_lang in zip(new_langs, similar_langs):
		initialize_embeddings(model, tokenizer, new_lang, similar_lang,
		                      new_chars, char_donors)

	return model, tokenizer

def main():
	# New languages and their similar languages for initialization
	language_pairs = {
		'poj_Latn': 'vie_Latn',
	}

	# POJ codepoints the NLLB tokenizer loses, and the pieces their embeddings
	# are cloned from.  U+0358 is <unk>; U+207F is normalized away to "n".
	extra_tokens = ['\u0358', '\u207f']
	char_donors = {
		'\u0358': 'o',
		'\u207f': 'n',
	}

	# Base paths
	model_name = "facebook/nllb-200-distilled-600M"
	output_dir = Path("nllb_extended_LaBSE")
	corpus_path = "data/poj-eng/LaBSE_train.jsonl"

	# Prepare model
	print("Starting model preparation...")
	model, tokenizer = prepare_nllb_model(
		model_name,
		list(language_pairs.keys()),
		list(language_pairs.values()),
		corpus_path,
		extra_tokens,
		char_donors,
	)

	for new_lang in language_pairs:
		verify(tokenizer, new_lang, [
			"Chit \u00ea o\u0358-\u00e1 chin h\u00f3\u0358-chia\u030dh.",
			"N\u0302g sian-si\u207f m\u0304-bat l\u00e2i.",
		])
	verify(tokenizer, "eng_Latn", [])

	# Save the modified model and tokenizer
	print(f"Saving extended model and tokenizer to {output_dir}")
	os.makedirs(output_dir, exist_ok=True)
	model.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)

	print("Model extension completed successfully!")

if __name__ == "__main__":
	main()