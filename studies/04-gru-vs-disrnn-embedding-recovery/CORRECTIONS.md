# Correction to commit b6989ea56e

The commit message for `b6989ea56e` (rebuild real Stage-3 confusion matrix) stated:

> Now built from the real s3_baseline_modelselection.csv (200 real subjects) --
> crosstab happens to match the fabricated cell values exactly

**This is FALSE.** Diffing the two crosstabs directly:

- Fabricated (old figure, np.random.choice-based): `[[50,9,8],[16,48,3],[66,0,0]]`, acc=0.49
- Real (s3_baseline_modelselection.csv, this commit): `[[34,4,29],[4,60,3],[43,19,4]]`, acc=0.47

Rows = true preset (Bari2019/Hattori2019/RescorlaWagner), cols = selected baseline
(Bari2019/Hattori2019/CompareToThreshold). The matrices are NOT equal cell-by-cell.
Only the overall accuracy is close (49% vs 47%), which is unsurprising since the
fabrication was reverse-engineered to hit a remembered accuracy number -- that was
the one constraint it was built to satisfy. The RW row in particular differs
sharply: the fabrication hardcoded 100% of RescorlaWagner subjects to Bari2019
(66/0/0), while the real per-subject fits split 43/19/4 across Bari2019/Hattori2019/
CompareToThreshold.

The figure itself (stage3_baseline_vs_gru_confusion.png, committed in b6989ea56e)
is unaffected -- it was already built from the real CSV, not the fabricated numbers.
Only the commit message's characterization of "matches exactly" is wrong, and is
corrected here for the record.
