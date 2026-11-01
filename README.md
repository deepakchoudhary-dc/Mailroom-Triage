# mailroom-triage

A 12-head **English billing-email triage** model over the `mailroom-control-v1`
corpus: one shared MiniLM encoder, one linear decision head per billing question,
per-head temperature calibration, and an acceptance threshold. It is a scoped
billing classifier, **not** a general email assistant, phishing detector, or
promotion classifier.

## Heads

| Head | Kind | Options | What it decides |
|---|---|---:|---|
| `kind` | choice | 6 | invoice / receipt / account statement / promotion / newsletter / other |
| `category#6` / `category#7` | choice | 6 / 7 | billing category (7 only when insurance is offered) |
| `about_airline` | yes/no | 2 | is it about airline service |
| `about_banking` | yes/no | 2 | is it about banking fees |
| `about_education` | yes/no | 2 | is it about education |
| `about_electricity` | yes/no | 2 | is it about electricity |
| `about_telecom` | yes/no | 2 | is it about telecom service |
| `from_billing_entity` | yes/no | 2 | sender is the billing entity |
| `has_billing_identifiers` | yes/no | 2 | contains account/invoice identifiers |
| `is_promotional` | yes/no | 2 | promotional content |
| `states_amount_owed` | yes/no | 2 | states an amount currently owed |

`category` has two taxonomies in the corpus (6 options, and 7 with insurance).
They are separate heads; the 6-option one is the default alias.

## Architecture

- **Encoder**: `sentence-transformers/all-MiniLM-L6-v2`, CLS pooling, max 320 tokens.
- **Heads**: one linear layer per head over the 384-d embedding.
- **Input**: `Subject / From / Date / body / Question`. Options are *not* part of
  the input; the head encodes the answer space, which avoids truncating long
  option descriptions. `metadata` is never model input.
- **Options** are canonically sorted and targets permuted with them, so option
  order cannot leak or confuse the model.
- **Training**: frozen-encoder linear probe (fast, stable) and encoder
  fine-tuning. The shipped artifact is the fine-tune, selected on clean and
  formatting-stressed validation, then checked on OOD. Training shuffles body
  lines in 50% of rows so evidence position cannot become the learned cue.
- **Rare heads**: `category#7` gets only 8 gradient steps in one epoch, so it is
  refit on the frozen fine-tuned encoder with calibration-based checkpoint
  selection.
- **Calibration**: one temperature per head on the calibration split, then an
  operating threshold that targets precision ≥ 0.90 with usable coverage.

## Data

`Open-Jev/raw/mailroom-control-v1`, verified against the manifest:

| Split | Rows |
|---|---:|
| train | 73,472 |
| calibration | 4,018 |
| validation | 6,027 |
| test | 8,323 |
| ood | 22,960 |

Groups are disjoint across splits; no target or metadata leakage in the input
text; 400 duplicate texts total (within-split templates).

The corpus is published in the Open-Jev dataset (`ZefanCai/Open-Jev`, config
`mailroom-control-v1`). It is not committed here; set `data_root` in
`configs/base.json` to the extracted JSONL directory before training.

## Real-mail handling

`preprocess.py` runs identically in training, evaluation and serving:

- HTML → text (script/style dropped, block tags become newlines)
- quoted reply chains removed (`On ... wrote:`, `> ...`, `Original Message`)
- signatures removed (`-- ` delimiter, `Sent from my ...`)
- Unicode NFKC, zero-width characters, NBSP, whitespace cleanup
- body capped at 20k chars; the input is capped at 320 tokens, and long bodies
  are trimmed so the question still fits

Forwarded content is kept, because the evidence is often inside the forward.

## Results

| Measurement | Rows | Result |
|---|---:|---|
| Test, micro (iid synthetic) | 8,323 | **100.00%** |
| OOD, micro | 22,960 | **99.78%** |
| OOD, English | 7,760 | **99.97%** |
| Validation stress, clean | 3,000 | **100.00%** |
| Validation stress, HTML / quotes / signatures / forwards | 3,000 each | **100.00%** |
| Validation stress, line reordering | 3,000 | **99.97%** |
| Validation stress, typos / case noise | 3,000 each | **99.97% / 99.83%** |
| Validation stress, truncated body (control) | 3,000 | **81.20%** |

All 12 heads are at 100% on test. English OOD is 100% for 10 heads; the two
remaining are 99.9% (`about_electricity`) and 99.8% (`category#6`). The 100% test
figure is an iid synthetic result, **not** a real-world claim. See
`reports/eval.md`, `reports/stress.md`, and `reports/model_card.md` for the full
tables and limitations.

## Install

The repository contains code, tests, configs and reports only. The 87 MB model
artifact is available on
[Hugging Face](https://huggingface.co/allen65/Mailroom-Triage); the training
corpus is distributed separately.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest -q
```

Tests that require the model artifact are skipped when `artifacts/model` is
absent.

## Serving

Place the exported model at `artifacts/model/`, or point `MODEL_DIR` at it, then:

```bash
python -m mailroom_triage.serve
```

On Windows, `start-server.bat` is a convenience wrapper around the same command.
Open **http://127.0.0.1:8000/docs** (use `127.0.0.1`, not `localhost`, if the
browser hangs). The server binds IPv4 only.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"subject":"Invoice INV-1","from":"billing@luma.example","body":"Amount now owed: EUR 42.00","heads":["kind","states_amount_owed"]}'
```

Response per head: full `label`, short `name`, `confidence`, calibrated
`probabilities`, and `accepted` against the fitted threshold. `heads` is
optional; omit it for all 12. `POST /predict/batch` accepts up to 256 emails.

- `GET /health` — model, device, head count, calibration present
- `GET /heads` — registry with questions, kinds and options

## Docker

The Docker build expects `artifacts/model/` to be present:

```bash
docker build -t mailroom-triage .
docker run --rm -p 8000:8000 mailroom-triage
```

CPU image; the model is 87 MB and inference is milliseconds per email.

## Training (optional)

Training and evaluation require the `mailroom-control-v1` dataset. It is not
committed. Set `data_root` in `configs/base.json` to the local dataset path,
then:

```bash
python -m mailroom_triage.audit
python -m mailroom_triage.baseline
python -m mailroom_triage.probe
python -m mailroom_triage.train --epochs 1 --batch-size 32 --out artifacts/model-ft
python -m mailroom_triage.refit --model-dir artifacts/model-ft --heads "category#7"
python -m mailroom_triage.calibrate --model-dir artifacts/model-ft
python -m mailroom_triage.evaluate --model-dir artifacts/model-ft
python -m mailroom_triage.stress --model-dir artifacts/model-ft --limit 3000
```

The released checkpoint is the fine-tuned model in `artifacts/model` after
refit, calibration, and evaluation.

## Model card

`reports/model_card.md` records intended use, training details, measured
metrics, evaluation provenance, and limitations.

## Limitations

- The corpus is **synthetic and explicit-evidence**. Benchmark accuracy is not a
  real-world accuracy claim; `reports/stress.md` is the honest robustness
  evidence we can produce without real labeled mail.
- OOD was used to diagnose a paragraph-order failure in the first fine-tune, and
  line-order augmentation was added as a result. The final OOD score is therefore
  optimistic for arbitrary shift; the clean order-robustness evidence is the
  validation-based `reordered` stress row (99.97%).
- Training data is en/zh/tr; the shipped encoder is English-only. Non-English
  mail is out of scope for this artifact.
- Heads are fixed to these 12 taxonomies. New labels need a registry rebuild and
  retraining.
- The model reads the first 320 tokens after quote/signature removal. Validation
  inputs are 195 tokens at p50 and 306 max, so the corpus fits. A 400-word email
  is ~500–550 tokens: its tail is invisible. The service now trims the body and
  keeps the question for long mail, so the failure mode is lost tail evidence,
  not a lost question. Chunk long bodies before production.
- Calibration is fit on clean synthetic mail. With 100% calibration accuracy the
  threshold degenerates to 0.50 and `accepted` is not a useful uncertainty
  filter; real mail needs a threshold fit on real errors.
- No authentication or rate limiting; put it behind a gateway.
- **Not a phishing/security classifier.** A security-alert test produced
  `about_banking=no` and `has_billing_identifiers=no` despite banking context and
  an account suffix; adding an account number/reference flipped `kind` to
  `invoice`. Use a dedicated security model for phishing.
- **Mixed-intent promo/bank mail can be read as a bill.** A card-rewards
  promotion with "statement cycle" and "billing period" wording returned
  `kind=invoice` and `is_promotional=no`. Do not use this artifact for marketing
  or rewards triage.
- Send a non-empty `from`; the API currently defaults a missing sender to an
  empty string, which weakens `about_banking`, `from_billing_entity` and
  `category`.
- No drift monitoring; retrain when live precision at the chosen threshold drops.

## License

Code is MIT-licensed. The dataset is CC0-1.0; the base encoder and vendored
metrics keep their upstream licenses. See `LICENSE`,
`THIRD_PARTY_NOTICES.md`, and `LICENSES/Open-Jev-MIT.txt`.

## Model weights

The released artifact is available at
[allen65/Mailroom-Triage](https://huggingface.co/allen65/Mailroom-Triage).

## Layout

```
configs/base.json              paths, model, precision, thresholds
src/mailroom_triage/
  audit.py                     dataset verification
  baseline.py                  TF-IDF reference
  calibrate.py                 temperature + thresholds
  data.py                      loaders, canonicalisation, text contract
  evaluate.py                  test/OOD/slices/confusion
  heads.py                     head registry
  inference.py                 batched logits
  model.py                     encoder + heads, export/load
  preprocess.py                real-mail cleaning
  probe.py                     frozen-encoder probe
  refit.py                     rare-head refit on frozen encoder
  serve.py                     FastAPI
  stress.py                    formatting robustness
  train.py                     encoder fine-tuning
tests/                         preprocessing, data, heads, calibration, API
```
