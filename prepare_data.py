"""Train a small byte-level BPE tokenizer on TinyStories and dump token streams as uint16 .bin files."""
import os, time
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VOCAB = 4096
EOS = "<|endoftext|>"


def read_stories(path, limit=None):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    stories = [s.strip() for s in text.split(EOS)]
    stories = [s for s in stories if s]
    return stories[:limit] if limit else stories


def main():
    t0 = time.time()
    tok_path = os.path.join(DATA, "tok4096.json")
    train_stories = read_stories(os.path.join(DATA, "TinyStoriesV2-GPT4-train.txt"))
    valid_stories = read_stories(os.path.join(DATA, "TinyStoriesV2-GPT4-valid.txt"))
    print(f"stories: train {len(train_stories):,}  valid {len(valid_stories):,}  ({time.time()-t0:.0f}s)")

    if not os.path.exists(tok_path):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOS],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      show_progress=False)
        tok.train_from_iterator(train_stories[:400_000], trainer=trainer)
        tok.save(tok_path)
        print(f"tokenizer trained ({time.time()-t0:.0f}s)")
    tok = Tokenizer.from_file(tok_path)
    eos_id = tok.token_to_id(EOS)
    assert eos_id == 0

    for name, stories in [("valid", valid_stories), ("train", train_stories)]:
        out = np.memmap(os.path.join(DATA, f"{name}.bin.tmp"), dtype=np.uint16, mode="w+",
                        shape=(sum(len(s) for s in stories) + len(stories),))  # upper bound (>= #tokens)
        pos = 0
        bs = 20_000
        for i in range(0, len(stories), bs):
            encs = tok.encode_batch(stories[i:i + bs])
            ids = np.fromiter((x for e in encs for x in (e.ids + [eos_id])), dtype=np.uint16)
            out[pos:pos + len(ids)] = ids
            pos += len(ids)
        out.flush()
        final = np.memmap(os.path.join(DATA, f"{name}.bin"), dtype=np.uint16, mode="w+", shape=(pos,))
        final[:] = out[:pos]
        final.flush()
        del out
        os.remove(os.path.join(DATA, f"{name}.bin.tmp"))
        chars = sum(len(s) for s in stories)
        print(f"{name}: {pos:,} tokens, {chars/pos:.2f} chars/token ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
