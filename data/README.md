# Data Directory

- Place or symlink datasets under `data/raw/` (do not commit images)
- Clean/interim outputs go to `data/interim/`
- Final splits should be materialized under `data/processed/{train,val,test}`

Recommended: store CSV manifests of image paths and labels under `data/processed/` and keep images outside the repo.
