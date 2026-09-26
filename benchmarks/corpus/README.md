# Natural-text corpus for `bench-natural.py`

`corpus.txt` is a concatenation of public-domain English novels downloaded
from Project Gutenberg. It supplies the natural-English filler for the
natural-text long-context lane (`benchmarks/bench-natural.py`), which sits
beside the synthetic pseudo-word lanes that defeat vLLM automatic prefix
caching — this lane probes whether the model's PLE n-gram embedding system
behaves differently on real prose.

## Reproducibility

**`corpus/` is gitignored — `corpus.txt` is NOT committed.** The reproducibility
path is the fetch script:

```bash
./fetch_corpus.sh   # rebuilds corpus.txt in place; idempotent
```

It downloads each book, verifies the bytes look like real English text of a
plausible size, strips the Project Gutenberg license header/footer, and
concatenates into `corpus.txt` (guarded to stay under 20 MB).

## Contents (as fetched 2026-09-25)

| Title | Author | Gutenberg ID / URL | Raw bytes | Words (after boilerplate strip) |
|---|---|---|---:|---:|
| War and Peace | Leo Tolstoy | [2600](https://www.gutenberg.org/ebooks/2600) | 3,359,610 | 563,286 |
| Moby-Dick; or, The Whale | Herman Melville | [2701](https://www.gutenberg.org/ebooks/2701) | 1,276,267 | 212,796 |
| Frankenstein | Mary Wollstonecraft Shelley | [84](https://www.gutenberg.org/ebooks/84) | 448,885 | 75,042 |

Totals: **851,124 words ≈ 1.12M tokens** (measured 1.317 tokens/word against
this server — see the calibration comment in `bench-natural.py`),
corpus.txt = 5,025,154 bytes (~4.8 MB, well under the 20 MB cap and far
above the ~460K-word / 600K-token minimum).

## Public-domain status

All three works were published well before 1900 (1869, 1851, 1818) and are in
the public domain in the United States. They are distributed by Project
Gutenberg, whose plain-text editions are freely redistributable; the script
removes the Gutenberg license boilerplate, keeping only the book text.
