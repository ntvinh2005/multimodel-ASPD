"""Duplicate-token probes.

A: [eos] r_1..r_n followed by a random permutation of the same tokens (probe: second half).
B: a pool of single-token names followed by names resampled from the pool (probe: repeats).
"""

import torch

from aspd.analysis.probes.common import EOS, Probe, Tally, parse

NAMES = ["John", "Mary", "James", "Anna", "Peter", "Laura", "David", "Sarah", "Michael", "Emma",
         "Robert", "Alice"]


def main() -> None:
    a = parse(__doc__.splitlines()[0], layer=3, head=0)
    p = Probe(a)

    # A. duplicate token
    tally = Tally(p)
    for seed in range(a.prompts):
        g = torch.Generator().manual_seed(seed)
        r = torch.randint(1000, 49000, (a.n,), generator=g)
        perm = torch.randperm(a.n, generator=g)
        ids = torch.cat([torch.tensor([EOS]), r, r[perm]]).unsqueeze(0)
        tr = p.trace(ids)
        tally.add(tr, list(range(1 + a.n, 1 + 2 * a.n)), list(range(1, 1 + a.n)))
    tally.show("A. duplicate TOKEN")

    # B. duplicate name
    names = [x for x in NAMES if len(p.tok(" " + x)["input_ids"]) == 1]
    pool = [p.tok(" " + x)["input_ids"][0] for x in names]
    tally = Tally(p)
    for seed in range(a.prompts):
        g = torch.Generator().manual_seed(100 + seed)
        seq = pool + [pool[int(i)] for i in torch.randint(len(pool), (2 * a.n,), generator=g)]
        ids = torch.tensor([EOS] + seq).unsqueeze(0)
        seen: set[int] = set()
        hit, ctl = [], []
        for pos, t in enumerate(ids[0].tolist()):
            if pos:
                (hit if t in seen else ctl).append(pos)
                seen.add(t)
        tally.add(p.trace(ids), hit, ctl)
    tally.show(f"B. duplicate NAME ({len(names)} names)")
    p.say("done")


if __name__ == "__main__":
    main()
