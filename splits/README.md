# Cross-validation folds

The folds are determined by the files on disk and seed `2026`. This repo ships
the checksums, not the 340 MB of per-image JSON.

```bash
export MLLV1_ROOT=/path/to/Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer_v1
PYTHONPATH=src python scripts/make_cv_splits.py mllv1
```

The script writes the folds and then compares them with `CHECKSUMS.json`. Each
digest hashes the image paths relative to the dataset root, not the absolute
path stored in the fold file. It must end with *Splits match the reference.*
If it does not, this copy does not line up with the folds used in the paper.

After generation:

```
splits/mllv1/
  CHECKSUMS.json   tracked
  config.json      tracked
  summary.json     tracked
  fold_0.json ...  generated locally (gitignored)
  manifest.json    generated locally (gitignored)
```
