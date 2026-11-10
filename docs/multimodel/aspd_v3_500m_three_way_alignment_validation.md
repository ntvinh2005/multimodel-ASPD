# ASPD v3.1 real three-way alignment validation

Status: **PASS — bounded real-tokenizer validation**

This validation used the exact approved tokenizer revisions for Gemma, SmolLM2, and Granite against
the pinned `monology/pile-uncopyrighted` stream. It did not approximate token positions or compare
token IDs across models. The shared object remained the exact canonical source-text prefix.

## Contract

For canonical text \(x\), model \(m\), and tokenizer token end \(r\), a boundary is eligible only
when the fast-tokenizer offset ends at \(r\) and decoding the whole native prefix returns exactly
\(x[:r]\). The accepted set is

\[
B(x)=B_{\mathrm{Gemma}}(x)\cap B_{\mathrm{SmolLM2}}(x)
\cap B_{\mathrm{Granite}}(x).
\]

Every accepted anchor records the same character boundary, UTF-8 byte boundary, and canonical
prefix SHA-256 across all three model positions.

## Execution

- Dataset: `monology/pile-uncopyrighted@3be90335b66f24456a5d6659d9c8d208c0357119`.
- Scan: first 2,000 deterministic stream rows.
- Sample: 512 unique train-split documents with the lowest content SHA-256 values in that prefix.
- Maximum audit document size: 100,000 characters; seven larger documents were classified and
  excluded before sampling.
- Native sequence limit: 512 tokens/model.
- Each sampled document was aligned twice independently.
- A separately instantiated dataset stream replayed the first 64 valid document hashes.

The raw immutable artifact is
`analysis/aspd_v3_500m_blocker_resolution/three_way_alignment_audit.json`, file SHA-256
`6dec94cf8d0f9302874b77ae06f2e7aceefe113c6e2d028f3936483a20044298`.

## Results

| Metric | Result |
|---|---:|
| Sampled documents | 512 |
| Accepted aligned anchors | 154,868 |
| Union candidate boundaries | 214,808 |
| Boundary-weighted acceptance | 0.720960 |
| Zero-anchor documents | 0 |
| Boundary/UTF-8/prefix-hash invariant failures | 0 |
| Within-process repeatability | PASS |
| Dataset prefix replay | PASS |
| Alignment-filter source-distribution TV | 0.028386 |
| Document-to-accepted source-distribution TV | 0.051385 |
| Source groups in scan/sample | 14 / 13 |

All 1,536 tokenizer/document passes used the exact linear streaming prefix decoder. The 59,940
boundaries not shared by all tokenizers were rejected rather than resampled. Granite and SmolLM2
had 52 and 66 prefix-roundtrip mismatches respectively; those boundaries were also rejected. No
document lost every anchor.

The alignment filter changed the source-group distribution by total variation 0.0284 relative to
candidate boundaries. This is below the versioned engineering limit 0.10 and is small relative to
the much larger allowed rejection margin. It does not prove absence of every possible semantic
bias; the full run must continue logging accepted documents and source groups.

## Exhaustion risk

The measured mean was 302.48 accepted anchors/document. The normal lower-95% estimate was 291.35,
requiring approximately 1,716,158 train documents for 500M non-repeated accepted anchors.

The pinned dataset API does not expose an exact row count. The pinned dataset README states that it
removes Books3, BookCorpus2, OpenSubtitles, YoutubeSubtitles, and OpenWebText2, while retaining
Pile-CC. The [official Pile composition table](https://github.com/EleutherAI/the-pile) reports
Pile-CC at 227.12 GiB with mean document size 4.33 KiB, implying approximately 55.0M Pile-CC
documents alone—about 32 times the lower-95% requirement. This is a capacity estimate, not a
full-corpus row census, but the margin is sufficient to reject dataset exhaustion as a current
500M blocker.

## Decision and limitations

The machine-readable approval artifact is
`analysis/aspd_v3_500m_blocker_resolution/three_way_alignment_approval.json`. It is pinned by the
main configuration, and preflight/main authorization revalidates its file hash, tokenizer roster,
execution-contract fingerprint, boundary tests, replay status, capacity status, and bias status.

This PASS validates the exact-prefix alignment method on a bounded real sample. It does not validate
physical model hooks, GPU memory, source-weight immutability, optimizer resume, or 500M training.
Those remain separate gates.

