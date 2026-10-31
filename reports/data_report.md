# Data report

Rows per split: {'train': 73472, 'calibration': 4018, 'validation': 6027, 'test': 8323, 'ood': 22960}

| Split | Rows | Languages | Duplicate texts |
|---|---:|---|---:|
| train | 73472 | {'en': 24832, 'tr': 24320, 'zh': 24320} | 256 |
| calibration | 4018 | {'en': 1358, 'tr': 1330, 'zh': 1330} | 14 |
| validation | 6027 | {'en': 2037, 'tr': 1995, 'zh': 1995} | 21 |
| test | 8323 | {'en': 2813, 'tr': 2755, 'zh': 2755} | 29 |
| ood | 22960 | {'en': 7760, 'tr': 7600, 'zh': 7600} | 80 |

| Head | Kind | Options | Train rows | Positive rate |
|---|---|---:|---:|---:|
| about_airline | noul | 2 | 6912 | 0.105 |
| about_banking | noul | 2 | 6912 | 0.197 |
| about_education | noul | 2 | 6912 | 0.099 |
| about_electricity | noul | 2 | 6912 | 0.101 |
| about_telecom | noul | 2 | 6912 | 0.070 |
| category#6 | choice | 6 | 4096 |  |
| category#7 | choice | 7 | 256 |  |
| from_billing_entity | noul | 2 | 6912 | 0.333 |
| has_billing_identifiers | noul | 2 | 6912 | 0.556 |
| is_promotional | noul | 2 | 6912 | 0.111 |
| kind | choice | 6 | 6912 |  |
| states_amount_owed | noul | 2 | 6912 | 0.333 |

Groups disjoint across splits: True
Privileged string in model text: False
