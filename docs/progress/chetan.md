# Progress — Chetan (Component 2: Vitals ML + fusion)

## Role
- Owns: MIMIC windows + outcome labels, NEWS2 baseline, LR/RF/GB baselines, BiGRU two-head,
  fusion + ablation + sensitivity table, SHAP/IG figures, model card (docs/02, 07 §4, 08 §2, 08 §5).
- Runs training on own machine; pushes `experiments/log.csv` rows + `reports/figures/`.

## Do now (no code needed)
1. Download MIMIC-III Clinical Database Demo v1.4 from
   https://physionet.org/content/mimiciii-demo/1.4/ into `data/raw/mimic-iii-demo/`
   (never commit it). Fill the MIMIC row in `data/PROVENANCE.md` (access date, terms).
2. Open `D_ITEMS.csv` and list item IDs for HR, RR, SpO2, systolic BP (arterial + non-invasive),
   temperature (°C and °F), GCS components, FiO2/supplemental O2. Note CareVue vs MetaVision.
   This becomes `data/mimic_itemid_map.json` — it must come from the real file, not guessed.
3. Read docs/08 §2 and docs/15 D3/D5 until you can explain the labels and the CV protocol.

## Done
- (nothing yet)

## Next
- Week 1: run `make data-mimic`, report class balance per severity class (merge if any < 5%).
- Week 2: `make train-baselines`.

## Blockers
- Waiting on `backend/data/mimic_windows.py` (Madhan builds with Claude Code).
