`manuscript_tables.json` records the printed mean and sample SD in Tables I/II.
Percentage metrics are stored in percentage points; AUC and MRR are fractions.
Use `python scripts/summarize_paper.py --check-manuscript` to compare all 140
printed values against the saved result JSON files. This check does not rerun training.
