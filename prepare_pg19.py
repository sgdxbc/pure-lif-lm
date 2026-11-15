"""PG-19 for the long-text track (README 5.22): train a byte-level BPE tokenizer on part of the training books and write
each split as one uint16 token stream, books separated by <|endoftext|> (id 0), plus the offset where each book starts.

  python3 prepare_pg19.py                      # reads data/pg19/raw/*.parquet (slurm/pg19_download.sh)

Writes data/pg19/tok8192.json, data/pg19/{train,valid,test}.bin and data/pg19/{train,valid,test}_books.npy (int64
start offsets). The text is lightly normalised: Project Gutenberg wraps lines at about 70 characters, so a single line
break inside a paragraph becomes a space; paragraph breaks (an empty line) stay, and longer runs of empty lines and
the blank lines at the start of a book are shortened.
"""
import argparse, glob, os, re, time
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

HERE = os.path.dirname(os.path.abspath(__file__))
EOS = "<|endoftext|>"
SPLITS = {"valid": "validation", "test": "test", "train": "train"}   # our name: parquet file prefix


def normalise(text):
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # line wrap inside a paragraph
    text = re.sub(r"\n{3,}", "\n\n", text)               # at most one empty line
    return re.sub(r"[ \t]{2,}", " ", text)


def books(raw, prefix):
    for f in sorted(glob.glob(os.path.join(raw, f"{prefix}-*.parquet"))):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for t in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                yield normalise(t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.join(HERE, "data", "pg19"))
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--tok_books", type=int, default=400, help="training books used to train the tokenizer")
    p.add_argument("--splits", default="valid,test,train")
    a = p.parse_args()
    t0 = time.time()
    raw = os.path.join(a.dir, "raw")
    tok_path = os.path.join(a.dir, f"tok{a.vocab}.json")
    if not os.path.exists(tok_path):
        sample = []
        for i, b in enumerate(books(raw, "train")):
            if i >= a.tok_books:
                break
            sample.append(b)
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=[EOS],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tok.train_from_iterator(sample, trainer=trainer)
        tok.save(tok_path)
        print(f"tokenizer trained on {len(sample)} books, {sum(map(len, sample)) / 1e6:.0f}M characters "
              f"({time.time() - t0:.0f}s)", flush=True)
        del sample
    tok = Tokenizer.from_file(tok_path)
    assert tok.token_to_id(EOS) == 0 and tok.get_vocab_size() <= 65535
    for name in a.splits.split(","):
        out_path = os.path.join(a.dir, f"{name}.bin")
        starts, pos, chars, batch = [], 0, 0, []
        with open(out_path + ".tmp", "wb") as out:
            def flush():
                nonlocal pos
                for e in tok.encode_batch(batch):
                    ids = np.asarray(e.ids + [0], dtype=np.uint16)
                    starts.append(pos)
                    out.write(ids.tobytes())
                    pos += len(ids)
                batch.clear()
            for b in books(raw, SPLITS[name]):
                batch.append(b)
                chars += len(b)
                if len(batch) == 64:
                    flush()
            flush()
        os.replace(out_path + ".tmp", out_path)
        np.save(os.path.join(a.dir, f"{name}_books.npy"), np.asarray(starts, dtype=np.int64))
        print(f"{name}: {len(starts):,} books, {pos:,} tokens, {chars / max(pos, 1):.2f} characters per token, "
              f"{pos / max(len(starts), 1):,.0f} tokens per book ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
