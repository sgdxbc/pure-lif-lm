"""Paired FineWeb passages for the learned binding probe."""

from dataclasses import dataclass

import numpy as np


BLOCK = 128
SEEDS = {"train": 3101, "dev": 3103, "held": 3104}
NAMES = {
    "train": "Alice Bob Carol David Emma Frank Grace Henry Isaac Jack Kate Liam Maria Noah Olivia Peter "
             "Quinn Rachel Sam Tom Victor Wendy Adam Sarah John Julia James Anna Mark Paul Amy Ben",
    "dev": "Aaron Anthony Ashley Brian Charlie Daniel Edward Elizabeth George Helen Ian Jane Kevin Laura "
           "Mason Nancy Oscar Patrick Robert Scott Tina Vincent Walter Zoe",
    "held": "Alex Arthur Austin Barbara Betty Bruce Chloe Claire Derek Emily Eric Ethan Felix Fiona Gary "
            "Hannah Harry Iris Jacob Jessica Joseph Kelly Leo Lucas Lucy Max Megan Mia Neil Nora Owen Rose",
}
PLACES = "Boston Chicago Denver Miami Sacramento Seattle Portland Dallas Phoenix Toronto Ottawa Atlanta".split()

# Each template contains two roles and one completion for each. The text before
# the answer is identical under the name swap.
TEMPLATES = {
    "train": (
        (" In {p}, {a} edited the article and {b} checked the figures.",
         " The article editor in {p} was", " The figures in {p} were checked by"),
        (" At {p}, {a} baked the bread while {b} made soup.",
         " The bread served at {p} was baked by", " The soup served at {p} was made by"),
        (" In {p}, {a} coached the team and {b} kept score.",
         " The team in {p} was coached by", " The person keeping score in {p} was"),
        (" At {p}, {a} painted the sign and {b} installed it.",
         " The sign at {p} was painted by", " The sign at {p} was installed by"),
    ),
    "dev": (
        (" In {p}, {a} designed the poster and {b} printed it.",
         " The poster designer in {p} was", " When the poster in {p} needed printing, it went to"),
        (" In {p}, {a} sailed the boat and {b} repaired it.",
         " The person who sailed the boat in {p} was", " The boat in {p} was repaired by"),
        (" At {p}, {a} wrote the lyrics and {b} sang them.",
         " The lyrics performed at {p} were written by", " The person singing at {p} was"),
        (" In {p}, {a} bought the timber and {b} cut it.",
         " The timber buyer in {p} was", " The timber in {p} was cut by"),
    ),
    "held": (
        (" In {p}, {a} drew the map and {b} delivered it.",
         " The map used in {p} was drawn by", " The map arrived in {p} thanks to its courier,"),
        (" At {p}, {a} mixed the paint and {b} covered the wall.",
         " The paint at {p} had been mixed by", " The person who painted the wall at {p} was"),
        (" In {p}, {a} composed a song and {b} recorded it.",
         " The composer of the song from {p} was", " The studio recording in {p} was made by"),
        (" At {p}, {a} found the archive and {b} catalogued it.",
         " The archive at {p} was discovered by", " The archive at {p} was catalogued by"),
    ),
}


@dataclass
class Example:
    tokens: np.ndarray
    other: np.ndarray
    position: int
    correct: int
    swapped: int
    original: np.ndarray


class Passages:
    def __init__(self, tokenizer, path, split):
        self.tok = tokenizer
        self.split = split
        self.names = NAMES[split].split()
        self.ids = {name: self.name_id(name) for name in self.names}
        self.rng = np.random.default_rng(SEEDS[split])
        self.data = np.memmap(path, mode="r", dtype=np.uint16)
        if split == "train":
            self.data = self.data[:64_000_000]
        eos = self.tok.token_to_id("<|endoftext|>")
        if eos is None:
            raise ValueError("expected FineWeb end-of-text token")
        boundaries = np.flatnonzero(self.data == eos)
        self.starts = boundaries[:-1] + 1
        # Discard the final document, which can be truncated by the prefix cap.
        self.ends = boundaries[1:]
        self.used = np.zeros(len(self.starts), dtype=bool)
        if split != "train":
            self.used[int(split == "dev")::2] = True

    def name_id(self, name):
        ids = self.encode(" " + name)
        if len(ids) != 1 or self.tok.decode(ids) != " " + name:
            raise ValueError(f"{name!r} is not one leading-space token")
        return ids[0]

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def documents(self, length):
        available = np.flatnonzero(~self.used & (self.ends - self.starts >= length))
        for doc in self.rng.permutation(available):
            start = self.rng.integers(self.starts[doc], self.ends[doc] - length + 1)
            self.used[doc] = True
            yield self.data[start:start + length].copy()

    def vignette(self, template, place, a, b):
        ids = self.encode(template.format(p=place, a=a, b=b))
        if any(ids.count(self.ids[name]) != 1 for name in (a, b)):
            raise ValueError("a name changed its tokenisation in a source sentence")
        return ids

    def make_example(self, original, distance, competitors):
        rng = self.rng
        names = rng.choice(self.names, 2 * (competitors + 1), replace=False)
        places = rng.choice(PLACES, competitors + 1, replace=False)
        templates = TEMPLATES[self.split]
        a, b = names[:2]
        role = int(rng.integers(2))
        correct, wrong = self.ids[names[role]], self.ids[names[1 - role]]
        if correct in original or wrong in original:
            return None
        source, *prompts = templates[rng.integers(len(templates))]
        fact = self.vignette(source, places[0], a, b)
        swapped = self.vignette(source, places[0], b, a)
        offset = fact.index(correct)
        if len(fact) != len(swapped) or swapped.index(wrong) != offset:
            raise ValueError("source names changed the positions of the paired story")
        query = prompts[role].format(p=places[0])
        prefix = self.encode(query)
        for name in (a, b):
            if self.encode(query + " " + name) != prefix + [self.ids[name]]:
                raise ValueError("a name joined a token at the query boundary")
        answer = int(rng.integers(len(original) - 64, len(original) - 1))
        fact_start = answer - distance - offset
        query_start = answer - len(prefix)
        first_free = fact_start + len(fact)
        if first_free > ((answer - 1) // BLOCK) * BLOCK:
            return None
        x = original[:-1].copy()
        x[answer] = self.encode(".")[0]
        x[query_start:answer] = prefix
        edges = np.linspace(first_free, query_start, competitors + 1, dtype=int)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), 1):
            other = templates[rng.integers(len(templates))][0]
            decoy = self.vignette(other, places[i], *names[2 * i:2 * i + 2])
            start = (lo + hi - len(decoy)) // 2
            if start < lo:
                raise ValueError("competing writes do not fit between source and query")
            x[start:start + len(decoy)] = decoy
        other = x.copy()
        x[fact_start:first_free] = fact
        other[fact_start:first_free] = swapped
        return Example(x, other, answer - 1, correct, wrong, original)

    def pairs(self, distance, competitors, count):
        rows = []
        for original in self.documents(distance + BLOCK + 1):
            example = self.make_example(original, distance, competitors)
            if example is not None:
                rows.append(example)
            if len(rows) == count:
                return rows
        raise ValueError(f"insufficient distinct {self.split} documents for gap {distance}")

    def ordinary(self, count):
        rows = []
        for original in self.documents(3 * BLOCK + 1):
            rows.append(original)
            if len(rows) == count:
                return np.stack(rows)
        raise ValueError(f"insufficient distinct {self.split} documents for ordinary NLL")
