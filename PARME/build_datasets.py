#!/usr/bin/env python3
"""Turn two line-aligned monolingual files into bidirectional NLLB training data.

Train-only by default: the whole corpus becomes train.jsonl.

No preprocessing: text is passed through byte for byte.  No normalisation, no
filtering, no deduplication, no transliteration.  Filter the mined bitext
beforehand if you want to, so that this stage stays identical to whatever the
other experiments feed their models.

Optional held-out splits
------------------------
--valid-size N carves N pairs off for validation, which is what makes
`--load_best_model_at_end --metric_for_best_model avg_bleu` usable; --test-size N
does the same for a test set.  When either is requested, pairs sharing a source
or a target sentence are kept inside one split (PARME 3.5 / Appendix C
criterion 1) unless --allow-split-overlap is passed.

Output
------
    train.jsonl            both directions, shuffled
    valid.poj2eng.jsonl    written only when --valid-size > 0; one direction per
    valid.eng2poj.jsonl    file, so each can be decoded with its own
    test.poj2eng.jsonl     forced_bos_token_id
    test.eng2poj.jsonl
    stats.json

Usage
-----
python build_dataset.py \
    --poj-file mined.poj.txt --eng-file mined.eng.txt --out-dir data/poj-eng
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


class DisjointSet:
    """Groups pairs that share a sentence, so a split boundary never cuts one."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_records(pairs, poj_lang, eng_lang, directions):
    records = []
    for poj, eng in pairs:
        translation = {poj_lang: poj, eng_lang: eng}
        if "poj2eng" in directions:
            records.append(
                {"translation": translation, "src_lang": poj_lang, "tgt_lang": eng_lang}
            )
        if "eng2poj" in directions:
            records.append(
                {"translation": translation, "src_lang": eng_lang, "tgt_lang": poj_lang}
            )
    return records


def split_pairs(pairs, valid_size, test_size, allow_overlap, rng):
    """Returns (train, valid, test).  Everything lands in train when both sizes
    are 0, and no grouping work is done in that case."""
    if valid_size <= 0 and test_size <= 0:
        return list(pairs), [], []

    if allow_overlap:
        groups = [[pair] for pair in pairs]
    else:
        dsu = DisjointSet()
        for poj, eng in pairs:
            dsu.union("P\t" + poj, "E\t" + eng)
        buckets: dict[str, list[tuple[str, str]]] = {}
        for poj, eng in pairs:
            buckets.setdefault(dsu.find("P\t" + poj), []).append((poj, eng))
        groups = list(buckets.values())
    rng.shuffle(groups)

    test: list[tuple[str, str]] = []
    valid: list[tuple[str, str]] = []
    train: list[tuple[str, str]] = []
    for group in groups:
        if len(test) + len(group) <= test_size:
            test.extend(group)
        elif len(valid) + len(group) <= valid_size:
            valid.extend(group)
        else:
            train.extend(group)
    return train, valid, test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poj-file", type=Path, required=True)
    ap.add_argument("--eng-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--poj-lang", default="poj_Latn")
    ap.add_argument("--eng-lang", default="eng_Latn")
    ap.add_argument("--valid-size", type=int, default=0,
                    help="0 (default) means train only, and no checkpoint selection.")
    ap.add_argument("--test-size", type=int, default=0)
    ap.add_argument("--allow-split-overlap", action="store_true",
                    help="Split at random instead of keeping shared sentences together.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    poj_lines = read_lines(args.poj_file)
    eng_lines = read_lines(args.eng_file)
    if len(poj_lines) != len(eng_lines):
        sys.exit(f"line count mismatch: {len(poj_lines)} vs {len(eng_lines)}")

    # A blank side cannot be turned into a training example; count and skip.
    pairs: list[tuple[str, str]] = []
    blank = 0
    for poj, eng in zip(poj_lines, eng_lines):
        if not poj.strip() or not eng.strip():
            blank += 1
            continue
        pairs.append((poj, eng))

    rng = random.Random(args.seed)
    train, valid, test = split_pairs(
        pairs, args.valid_size, args.test_size, args.allow_split_overlap, rng
    )

    train_records = make_records(train, args.poj_lang, args.eng_lang, {"poj2eng", "eng2poj"})
    rng.shuffle(train_records)
    write_jsonl(args.out_dir / "train.jsonl", train_records)

    for name, split in (("valid", valid), ("test", test)):
        if not split:
            continue
        for direction in ("poj2eng", "eng2poj"):
            write_jsonl(
                args.out_dir / f"{name}.{direction}.jsonl",
                make_records(split, args.poj_lang, args.eng_lang, {direction}),
            )

    stats = {
        "input_lines": len(poj_lines),
        "blank_side_skipped": blank,
        "pairs": len(pairs),
        "train_pairs": len(train),
        "valid_pairs": len(valid),
        "test_pairs": len(test),
        "train_examples_both_directions": len(train_records),
        "split_overlap_allowed": args.allow_split_overlap,
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()