# Robustness under realistic mail formatting

Split: `validation` (3000 rows), model: `model`.
Formatting perturbations do not change the answer; `truncate` removes evidence and is a control.

| Perturbation | Accuracy | about_airline | about_banking | about_education | about_electricity | about_telecom | category#6 | category#7 | from_billing_entity | has_billing_identifiers | is_promotional | kind | states_amount_owed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| html | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| quoted | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| signature | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| typos | 0.9997 | 1.000 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| case | 0.9983 | 1.000 | 1.000 | 0.986 | 1.000 | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| reordered | 0.9997 | 1.000 | 1.000 | 1.000 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| forwarded | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| truncate | 0.8120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.979 | 1.000 | 1.000 | 0.669 | 0.502 | 0.908 | 0.278 | 0.669 |
