# Third-party notices

This repository contains code and documentation only. Model weights, the
training corpus, and local caches are not committed.

## Open-Jev metrics

`src/mailroom_triage/_vendor_metrics.py` is vendored from the Open-Jev project:

- Source: https://github.com/Zefan-Cai/Open-Jev
- Path: `reproduce/source-code/jev/metrics.py`
- License: MIT, Copyright (c) 2026 Open-Jev contributors
- Full license text: `LICENSES/Open-Jev-MIT.txt`

It is included so that calibration, evaluation and tests work from a clean
clone without requiring the Open-Jev repository as a sibling directory.

## Base encoder

The model uses `sentence-transformers/all-MiniLM-L6-v2` as its base encoder.
The upstream model card is the authoritative source for its license and terms.
Verify the upstream license before redistributing model weights.

## Dataset

The training and evaluation corpus is `mailroom-control-v1` from Open-Jev.
Original generated records are dedicated under CC0-1.0. The dataset is not
included in this repository.
