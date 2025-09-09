# Data Card

Sources: FER-2013, RAF-DB, AffectNet (public). Images are not stored in this repo.

- Label set: angry, happy, sad, neutral (optionally: fear, surprise, disgust)
- Splits: stratified 80/10/10 train/val/test; optional cross-dataset test
- Cleaning: remove corrupt, deduplicate via perceptual hash, standardize naming
- Face processing: optional detection and alignment
- Known biases: demographic imbalance, lighting conditions, posed vs spontaneous

Please expand with counts and stats in `data_summary.json`.
