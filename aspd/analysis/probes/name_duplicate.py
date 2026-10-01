"""Names-and-repeated-names probe: does a V-matrix component fire on names, on repeats, or more on
repeated names?

Tokens are labelled name_first, name_rep, word_first or word_rep. With e-bar_k the mean |e_{t,c}|
over class k:

    Pi_name = e-bar(name_rep) - e-bar(name_first)
    Pi_word = e-bar(word_rep) - e-bar(word_first)
    Delta   = Pi_name - Pi_word

A pure name detector or a pure duplicate detector has Delta ~ 0; Delta > 0 means repetition raises
the component more on names than on other words.

Prompt A: [eos] followed by `--n` slots alternating a name and a common noun, each drawn from a pool
of 12 single-token names and 12 nouns (`--prompts` seeds). Prompt B: six hand-written passages, each
with one name and one noun that occur twice.

    python -m aspd.analysis.probes.name_duplicate --run <run id> --layer 8 --comp 101
"""

import argparse
import os

import torch

from aspd.analysis.probes.common import EOS, Probe

NAMES = ["John", "Mary", "James", "Anna", "Peter", "Laura", "David", "Sarah", "Michael", "Emma",
         "Robert", "Alice"]
WORDS = [" table", " river", " garden", " window", " market", " bridge", " letter", " coffee",
         " basket", " ticket", " mirror", " candle"]

# (passage, recurring name, recurring noun)
PASSAGES = [
    ("The morning meeting ran long, and John brought coffee for everyone waiting there. "
     "By the time John sat down, the coffee had gone cold.", ["John"], [" coffee"]),
    ("Mary left a letter on the kitchen table before sunrise. "
     "Nobody read the letter until Mary came back that evening.", ["Mary"], [" letter"]),
    ("A storm took out the bridge north of town, so Peter drove the long way around. "
     "Peter said the bridge had been failing for years.", ["Peter"], [" bridge"]),
    ("Sarah kept a mirror in the hallway of the old house. "
     "Guests always stopped at the mirror, which amused Sarah.", ["Sarah"], [" mirror"]),
    ("The market opens at six, and David walks there most mornings. "
     "David says the market is quieter before the tourists arrive.", ["David"], [" market"]),
    ("Anna found a ticket folded inside her coat pocket. "
     "The ticket was three years old, which made Anna laugh.", ["Anna"], [" ticket"]),
]

CELLS = ["name_first", "name_rep", "word_first", "word_rep"]


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layer", type=int, default=8, help="layer of the V matrix under test")
    p.add_argument("--comp", type=int, default=101, help="the component to test")
    p.add_argument("--n", type=int, default=40, help="slots per Prompt A sequence")
    p.add_argument("--prompts", type=int, default=4, help="Prompt A sequences (seeds)")
    p.add_argument("--run", required=True,
                   help="run id under $PARAM_DECOMP_OUT_DIR/runs (the model-wide ASPD run)")
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 8)))
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    return a


class Cells:
    """Per-class sums of |e_{t,c}| for one component, pooled over prompts."""

    def __init__(self):
        self.sum = dict.fromkeys(CELLS, 0.0)
        self.fired = dict.fromkeys(CELLS, 0)
        self.n = dict.fromkeys(CELLS, 0)
        self.peak = dict.fromkeys(CELLS, 0.0)

    def add(self, e: torch.Tensor, labels: dict[int, str]) -> None:
        for pos, cell in labels.items():
            v = float(e[pos])
            self.sum[cell] += v
            self.fired[cell] += int(v > 0)
            self.n[cell] += 1
            self.peak[cell] = max(self.peak[cell], v)

    def mean(self, cell: str) -> float:
        """Mean |e| over every token of the class."""
        return self.sum[cell] / self.n[cell] if self.n[cell] else float("nan")

    def when_fired(self, cell: str) -> float:
        """Mean |e| over the tokens of the class where the component fired."""
        return self.sum[cell] / self.fired[cell] if self.fired[cell] else float("nan")

    def show(self, title: str) -> None:
        print(f"\n=== {title} ===")
        print(f"{'cell':>12} {'n':>5} {'mean |e|':>9} {'fires':>11} {'rate':>6} "
              f"{'mean|fired':>11} {'peak':>8}")
        for cell in CELLS:
            n = self.n[cell]
            rate = f"{self.fired[cell]}/{n}" if n else "-"
            pct = f"{100 * self.fired[cell] / n:.0f}%" if n else "-"
            print(f"{cell:>12} {n:>5} {self.mean(cell):>9.3f} {rate:>11} {pct:>6} "
                  f"{self.when_fired(cell):>11.3f} {self.peak[cell]:>8.3f}")
        pi_name = self.mean("name_rep") - self.mean("name_first")
        pi_word = self.mean("word_rep") - self.mean("word_first")
        print(f"  Pi_name {pi_name:+.3f}   Pi_word {pi_word:+.3f}   Delta {pi_name - pi_word:+.3f}")


def list_prompt(p: Probe, a: argparse.Namespace, seed: int):
    """Prompt A: names and nouns alternating; each slot labelled by kind and first/repeat."""
    g = torch.Generator().manual_seed(seed)
    names = [p.tok(" " + x)["input_ids"][0] for x in NAMES if len(p.tok(" " + x)["input_ids"]) == 1]
    words = [p.tok(x)["input_ids"][0] for x in WORDS if len(p.tok(x)["input_ids"]) == 1]
    seq, kinds = [], []
    for i in range(a.n):
        pool, kind = (names, "name") if i % 2 == 0 else (words, "word")
        seq.append(pool[int(torch.randint(len(pool), (1,), generator=g))])
        kinds.append(kind)
    ids = torch.tensor([EOS] + seq).unsqueeze(0)
    seen, labels = set(), {}
    for i, (t, kind) in enumerate(zip(seq, kinds, strict=True)):
        labels[i + 1] = f"{kind}_{'rep' if t in seen else 'first'}"
        seen.add(t)
    return ids, labels


def passage_prompt(p: Probe, text: str, names: list[str], words: list[str]):
    """Prompt B: label the passage's recurring name and noun (first / repeat); nothing else."""
    ids = torch.tensor(p.tok(text)["input_ids"]).unsqueeze(0)
    flat = ids[0].tolist()
    labels = {}
    for kind, targets in (("name", names), ("word", words)):
        for w in targets:
            # A word opening the passage is tokenized without its leading space.
            ids_of = {p.tok(form)["input_ids"][0] for form in (" " + w.strip(), w.strip())
                      if len(p.tok(form)["input_ids"]) == 1}
            hits = sorted(i for i, t in enumerate(flat) if t in ids_of)
            for j, pos in enumerate(hits):
                labels[pos] = f"{kind}_{'first' if j == 0 else 'rep'}"
    got = sorted(labels.values())
    assert got == sorted(CELLS), f"{text[:40]!r} labelled {got}"
    return ids, labels


def main() -> None:
    a = parse()
    p = Probe(a)
    v_mod = None

    cells = Cells()
    for seed in range(a.prompts):
        ids, labels = list_prompt(p, a, seed)
        tr = p.trace(ids)
        v_mod = v_mod or tr.module_at(a.layer, "attn.v")
        cells.add(tr.effective(v_mod)[:, a.comp].abs(), labels)
    cells.show(f"Prompt A (random co-occurrence) -- {v_mod}:{a.comp}")

    passages = Cells()
    for text, names, words in PASSAGES:
        ids, labels = passage_prompt(p, text, names, words)
        passages.add(p.trace(ids).effective(v_mod)[:, a.comp].abs(), labels)
    passages.show(f"Prompt B (natural co-occurrence) -- {v_mod}:{a.comp}")
    p.say("done")


if __name__ == "__main__":
    main()
