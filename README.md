# Mom or Dad? Family Resemblance with PCA

Predicting family resemblance: PCA of the parents' average face vs siblings, plus a small Streamlit demo that tells which parent a child looks more like.

**Live demo:** https://mom-or-dad.streamlit.app/

## What it does

Upload a father, mother and child photo. The app:

1. Detects and crops each face, converts to grayscale, resizes to 64x64 and equalizes the histogram.
2. Projects all three faces into a PCA eigenface space (omega).
3. Compares the child's distance to each parent and shows the closer parent with a percentage split.
4. Shows the parents' average face next to the child.
5. Colors six face regions (forehead, both eyes, nose, mouth, chin) blue (father) or pink (mother), using a separate small PCA for each region.

## Method

- **Data:** TSKinFace, FMSD folder only (228 families, each with father, mother, son and daughter, 64x64 cropped faces).
- **Split:** by family, 5-fold, so siblings never leak between train and test. PCA is fit on training faces only.
- **PCA:** `PCA(0.95)` on flattened pixels (about 127 components).
- **Parents' average:** `(father + mother) / 2`, projected into the same space.
- **Evaluation:** verification test. A child counts as correct if they are closer to their own parents' average (or own sibling) than to a random other family's.

## Results (5-fold, all 228 families)

| Test | Accuracy (chance = 50%) |
|---|---|
| Parents' average vs child | about 59% |
| Sibling vs sibling | about 73% |

Per-region PCA (child vs own parent, vs random parent): forehead 66%, left eye 69%, right eye 70%, nose 71%, mouth 71%, chin 62%.

## Limitations

- PCA on raw pixels mostly captures lighting, pose, age and expression, not inherited features, so accuracy is modest.
- Siblings are matched better than the parents' average, partly because siblings are of similar age and share the photo conditions. Averaging two faces also blurs toward the mean face.
- The region colors compare appearance, so a closed mouth vs a grin can mislead. Treat them as a visual aid, not proof.
- The overall percentage split is usually close to 50/50.
- Works best with front-facing, evenly lit photos.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `pca.pkl` | PCA fit on all 228 families |
| `region_pcas.pkl` | One PCA per face region |
| `requirements.txt` | Dependencies (OpenCV pinned below version 5) |

## Dataset

The TSKinFace images are **not included**, because the dataset terms forbid redistribution. Get the data from the authors and cite:

Qin, X., Tan, X., Chen, S. "Tri-Subject Kinship Verification: Understanding the Core of A Family."
