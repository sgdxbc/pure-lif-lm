"""General text for the stage goals of README section 0: FineWeb-Edu (the 10-billion-token sample, educational web
pages filtered from Common Crawl, the data of recent small-model reproductions such as llm.c), tokenized with
GPT-2's tokenizer, so validation losses are comparable with GPT-2-sized reproductions and GPT-2-family models can
serve as teachers. Every document is preceded by <|endoftext|> (id 50256), as in GPT-2's training data.

  bash slurm/fineweb_download.sh            # data/fineweb/raw/*.parquet (14 files, 28.5 GB)
  sbatch slurm/prepare_fineweb.sbatch       # this script on a CPU node

Writes data/fineweb/valid.bin (the first documents of the first file, about --valid_tokens tokens) and
data/fineweb/train.bin (all the rest), as uint16 token streams. The tokenizer, data/fineweb/gpt2.json, is GPT-2's
(openai-community/gpt2 on Hugging Face) and is in the repository.
"""
import argparse, glob, os, time
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def docs(raw):
    for f in sorted(glob.glob(os.path.join(raw, "*.parquet"))):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            yield from pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.join(HERE, "data", "fineweb"))
    p.add_argument("--valid_tokens", type=int, default=10_000_000)
    p.add_argument("--max_docs", type=int, default=0, help="stop after this many documents (a quick test)")
    a = p.parse_args()
    t0 = time.time()
    tok = Tokenizer.from_file(os.path.join(a.dir, "gpt2.json"))
    eot = tok.token_to_id("<|endoftext|>")
    assert eot == 50256 and tok.get_vocab_size() <= 65535
    outs = {s: open(os.path.join(a.dir, f"{s}.bin.tmp"), "wb") for s in ("valid", "train")}
    count = {"valid": [0, 0, 0], "train": [0, 0, 0]}   # documents, tokens, characters
    batch = []

    def flush():
        for text, e in zip(batch, tok.encode_batch(batch)):
            s = "valid" if count["valid"][1] < a.valid_tokens else "train"
            ids = np.asarray([eot] + e.ids, dtype=np.uint16)
            outs[s].write(ids.tobytes())
            c = count[s]
            c[0] += 1; c[1] += len(ids); c[2] += len(text)
        batch.clear()

    n = 0
    for text in docs(os.path.join(a.dir, "raw")):
        batch.append(text)
        n += 1
        if len(batch) == 4096:
            flush()
            if n % (4096 * 100) == 0:
                print(f"{n:,} documents, {count['train'][1] / 1e9:.2f}B training tokens ({time.time() - t0:.0f}s)",
                      flush=True)
        if a.max_docs and n >= a.max_docs:
            break
    flush()
    for s, f in outs.items():
        f.close()
        os.replace(os.path.join(a.dir, f"{s}.bin.tmp"), os.path.join(a.dir, f"{s}.bin"))
        d, t, c = count[s]
        print(f"{s}: {d:,} documents, {t:,} tokens, {c / max(t, 1):.2f} characters per token, "
              f"{t / max(d, 1):,.0f} tokens per document ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
