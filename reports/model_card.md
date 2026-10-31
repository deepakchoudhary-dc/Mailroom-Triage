# Model card — mailroom-triage v1

## What this is

A 12-head **English billing-email triage** classifier over the
`mailroom-control-v1` corpus. One shared encoder, one linear head per
`(question, option set)`, per-head temperature calibration, and an acceptance
threshold. It is not a general email classifier, phishing detector, or
promotion classifier.

Base encoder: `sentence-transformers/all-MiniLM-L6-v2` (22.7M parameters),
fine-tuned end-to-end. CLS pooling, 384-d, 320 tokens, 87.4 MB artifact.

| Head | Kind | Options | Decision |
|---|---|---:|---|
| `kind` | choice | 6 | invoice / receipt / account statement / promotion / newsletter / other |
| `category#6` | choice | 6 | billing category (6-option taxonomy) |
| `category#7` | choice | 7 | billing category when insurance is offered |
| `about_airline` | yes/no | 2 | about airline service |
| `about_banking` | yes/no | 2 | about banking fees |
| `about_education` | yes/no | 2 | about education |
| `about_electricity` | yes/no | 2 | about electricity supply |
| `about_telecom` | yes/no | 2 | about telecom service |
| `from_billing_entity` | yes/no | 2 | sender is the provider issuing its own charges |
| `has_billing_identifiers` | yes/no | 2 | contains account/invoice identifiers |
| `is_promotional` | yes/no | 2 | promotional content |
| `states_amount_owed` | yes/no | 2 | states an amount currently owed |

## Intended use

English business-email triage: routing, tagging, prioritisation, and review
queues. Input is `subject`, `from`, `date`, `body`; the service returns a label,
calibrated probabilities, and an `accepted` flag.

Not intended for:

- non-English mail (the shipped artifact is English-focused);
- automatic payment, credit, employment, insurance, or legal decisions;
- identity verification or phishing verdicts;
- attachments, MIME-encoded headers, or base64/quoted-printable bodies;
- any workflow where a wrong label has material consequences without human review.

## Training data

`Open-Jev/raw/mailroom-control-v1`, verified against its manifest:

| Split | Rows | English rows |
|---|---:|---:|
| train | 73,472 | — |
| calibration | 4,018 | — |
| validation | 6,027 | — |
| test | 8,323 | 2,813 |
| ood | 22,960 | 7,760 |

The corpus is synthetic, English/Chinese/Turkish, and **explicit-evidence**: the
answer is usually stated in the body. Groups are disjoint across splits; the
input text contains no target or metadata fields. `state.email.from` is a dict
in the corpus and is normalised to `"Name <email>"` by the shared preprocessing,
matching what serving receives.

## Training procedure

- 1 epoch, batch 32, fp32, seed 42, max length 320.
- Encoder LR `3e-5`, head LR `1e-3`, weight decay `0.01`, 10% warmup, cosine decay.
- 2,296 steps, 1,443 s on a GTX 1650 Ti 4 GB (~2.4 GB VRAM).
- **Line-order augmentation**: 50% of training bodies have their lines shuffled.
  Real threads and the corpus OOD split move paragraphs around; without this the
  model learned where the evidence sits rather than what it says. This was the
  single largest robustness fix (see Evaluation provenance).
- Exported weights: `artifacts/model`.
- Rare head `category#7` (256 train rows, 8 steps in the epoch) is refit on the
  frozen fine-tuned encoder for 1,000 steps, with checkpoint selection on the
  calibration split. This took calibration accuracy from 76.2% to 100% in the
  first fine-tune and restored confidence in the augmented one.

## Calibration

Per-head temperature scaling on the calibration split, then an operating
threshold targeting precision ≥ 0.90 with coverage ≥ 0.20:

- temperatures: 0.081–0.206; `category#7` 1.0;
- all 12 thresholds: 0.50;
- calibration precision / coverage: 1.0 / 1.0 for every head.

Because calibration accuracy is 100%, the threshold fit degenerates to 0.50 and
`accepted` is not a useful uncertainty filter on this corpus. It becomes useful
only with real mail containing genuine errors.

## Evaluation

### Overall

| Split | Rows | Micro accuracy | en | tr | zh |
|---|---:|---:|---:|---:|---:|
| calibration | 4,018 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | 6,027 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | 8,323 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ood | 22,960 | 0.9978 | 0.9997 | 0.9999 | 0.9937 |

Test is in-distribution synthetic data. **100% here is not a real-world accuracy
claim.** The OOD split is the only distribution-shift measurement available.

### English OOD by head

| Head | Accuracy |
|---|---:|
| about_airline | 1.000 |
| about_banking | 1.000 |
| about_education | 1.000 |
| about_electricity | 0.999 |
| about_telecom | 1.000 |
| category#6 | 0.998 |
| category#7 | 1.000 |
| from_billing_entity | 1.000 |
| has_billing_identifiers | 1.000 |
| is_promotional | 1.000 |
| kind | 1.000 |
| states_amount_owed | 1.000 |

English OOD micro: **0.9997** over 7,760 rows. OOD misses are concentrated in
Chinese (`kind` 0.982 overall), which is outside the shipped scope.

### Formatting robustness

3,000 validation rows, perturbations that do not change the answer, plus one
control that removes evidence:

| Perturbation | Accuracy |
|---|---:|
| clean | 1.0000 |
| HTML | 1.0000 |
| quoted reply chain | 1.0000 |
| signature | 1.0000 |
| typos | 0.9997 |
| case noise | 0.9983 |
| line reordering | 0.9997 |
| forwarded header | 1.0000 |
| **truncated body (control)** | **0.8120** |

The truncation control quantifies the cost of missing evidence; it is not a
formatting perturbation.

### Evaluation provenance (important)

The first fine-tune scored 100% on test but only **70.6%** on English OOD
`from_billing_entity`. Inspection showed it had learned the *position* of the
provider sentence: reordering the same body lines raised that model to 98.5–100%.
Line-order augmentation was added on the train split, and the retrained model
scored 99.97% English OOD. The OOD split informed the diagnosis, so the final
OOD number is optimistic for arbitrary distribution shift. The clean evidence
for order robustness is the validation-based `reordered` perturbation
(**99.97%**), which was not used for training or tuning.

## Known limitations

- **Synthetic, explicit-evidence data.** Benchmark accuracy does not transfer
  automatically to natural mail with implicit or conflicting evidence.
- **320-token limit.** Validation inputs are 195 tokens at p50, 306 max, so the
  corpus fits. A 400-word email is ~500–550 tokens and its tail is invisible.
  The service trims the body and keeps the question, so the failure mode is lost
  tail evidence, not a lost question. Chunk long bodies before production.
- **English only.** The encoder and evaluation focus are English. Turkish and
  Chinese rows were still trained on for coverage, but are not a supported SLA.
- **Confidence is in-distribution.** Calibration is fit on clean synthetic rows.
  Treat probabilities on unusual real mail with caution.
- **Fixed taxonomy.** New questions or options require a registry rebuild and
  retraining; the 12 heads are not extensible at runtime.
- **Preprocessing scope.** HTML, quoted threads, signatures, forwarded headers,
  Unicode and whitespace are handled. Attachments, MIME-encoded headers, and
  transfer-encoded bodies are not.
- **Security alerts and phishing are out of distribution.** There is no
  phishing/security head. The training split contains **zero** login,
  unauthorized-access, or suspicious-activity rows (searched the full 73,472
  rows). A test email about an unauthorized banking login was correctly
  `kind=other` and `from_billing_entity=no`, but returned `about_banking=no` and
  `has_billing_identifiers=no` despite banking context and an account suffix.
  Adding a full account number and reference flipped `kind` to `invoice` at 1.0
  confidence. Do not use this artifact as a phishing detector; use a dedicated
  security classifier.
- **Promotional bank/card mail can be read as a bill.** A rewards email with
  "monthly statement cycle", "billing period" and a card suffix returned
  `kind=invoice` (0.944), `is_promotional=no` (1.0), `about_banking=no` (1.0) and
  `has_billing_identifiers=no` (1.0), despite "exclusive promotional tier",
  "promotional reward offer" and "5x bonus points". Billing vocabulary overrides
  promotional cues, and the `is_promotional` training positives are explicit
  price/savings offers. Do not use this artifact for marketing or rewards triage.
- **Missing `from` is treated as empty.** `about_banking`, `from_billing_entity`
  and `category` depend on the sender. Send a real `from` value; the current API
  does not reject an empty one.
- **No auth, rate limit, TLS, or drift monitoring.** Put it behind a gateway and
  log predictions if exposed.
- **Threshold is not an uncertainty filter** on this corpus (see Calibration).

## Serving

`POST /predict` (one email, optional `heads`), `POST /predict/batch` (≤256),
`GET /heads`, `GET /health`. Calibrated probabilities, a threshold decision, and
the registry question are returned per head. Docker image is CPU-only, runs as a
non-root user, and has a healthcheck. Long bodies are trimmed to fit the token
budget while reserving room for the question.

## Data and license notes

Corpus: CC0-1.0 for the original English/Chinese/Turkish rows; source code MIT.
Base encoder: check the upstream `sentence-transformers/all-MiniLM-L6-v2`
license before redistribution. This model card describes the artifact in this
repository, not a certification for production use.
