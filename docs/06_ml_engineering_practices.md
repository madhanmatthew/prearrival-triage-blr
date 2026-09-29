# ML Engineering Practices — Training Methodology, Tuning, Reproducibility

**Applies to:** every trained model in the project (Component 2's vitals/fusion
models, Component 3's RL agent, and the existing traffic/hospital RandomForest
models being retrained on real data).

**Reads first:** `00_MASTER_OVERVIEW.md`, and the relevant component file for
model-specific detail — this file covers the *methodology*, not the
model-specific architecture (that's in `02_ambulance_vitals_agent.md` and
`03_traffic_police_agent.md`).

**Why this file exists:** "we trained a RandomForest and got 91% accuracy" is not
evaluable — it doesn't say whether that 91% is real or a lucky split, whether
better hyperparameters were tried, or whether the number would hold up on new
data. This document makes the training process itself rigorous and reportable.

---

## 1. Core Principle: No Model Is Trained Once and Left Alone

Every model in this project goes through the same disciplined loop, not a single
`.fit()` call:

```
Data → Split (train/val/test) → [Train → Validate → Tune] × N → Final test eval (once)
```

The `[Train → Validate → Tune]` loop runs many times with different
hyperparameters/architectures — only the **final test evaluation** happens once,
at the end, after tuning is finished.

## 2. Train / Validation / Test Protocol

- **Split ratio:** 70/15/15 or 80/10/10, stratified by the target label (severity
  class) so class imbalance doesn't distort any split.
- **Test set is sacred:** never used for hyperparameter decisions. If you check
  test performance mid-tuning and then change something based on it, that number
  is no longer a valid final result — this is a common and easy-to-miss mistake.
- **Validation set** (or cross-validation folds, see below) is what tuning
  decisions are based on.

## 3. Cross-Validation (required for Component 2's models)

Single train/test splits can overstate or understate performance depending on
which patients happen to land in which split — especially with a dataset as
small as MIMIC-III Demo (100 patients). Use **k-fold cross-validation** (k=5
is standard) on the training portion:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
print(f"CV F1-macro: {scores.mean():.3f} ± {scores.std():.3f}")
```

Report the mean **and** standard deviation across folds — the spread tells you
how stable the model is, which matters more than a single point estimate on a
small clinical dataset.

## 4. Hyperparameter Tuning — Actually Search, Don't Guess

For the tree-based models (RandomForest/GradientBoosting in Component 2 and the
retrained traffic/hospital models):

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_split": [2, 5, 10],
}

search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print("Best CV score:", search.best_score_)
best_model = search.best_estimator_
```

For a small grid like this, `GridSearchCV` (exhaustive) is fine. If the search
space grows, switch to `RandomizedSearchCV` rather than exhaustively searching
everything — that's a legitimate, reportable engineering decision to mention
("used randomized search over N candidate configurations due to search-space
size") rather than a shortcut to hide.

## 5. Architecture Comparison — Not Just One Model Type

Component 2 should actually train and fairly compare multiple candidate model
types before picking one, not assume RandomForest is best:

| Candidate | Why it's in the comparison |
|---|---|
| Logistic Regression | Simple, interpretable baseline — if a complex model can't beat this, that's an important finding to report honestly |
| RandomForest | Current default choice, handles non-linear feature interactions well |
| Gradient Boosting | Often outperforms RF on tabular clinical data in literature |
| BiLSTM/GRU sequence model | The core deliverable — required for deterioration-trend prediction, and also produces the severity classification via its second output head. Not interchangeable with the tabular baselines above, which exist as comparison points it must beat, not alternatives to it |

Report a comparison table (CV F1-macro ± std for each), and justify the final
choice with the numbers, not just "we used RandomForest because it's common."

## 6. RL Training Specifics (Component 3)

RL is inherently iterative — training means running the agent through many
episodes, not a single fit call. Specific practices required:

- **Training curve, not just a final number:** log episode reward (or ambulance
  transit time) over training episodes and plot it. A model that's still
  improving when training stopped is a different (and honestly-reportable)
  result than one that's converged.
- **Multiple random seeds:** RL training can be unstable — train with at least
  2-3 different random seeds and report whether results are consistent. If
  results vary wildly across seeds, that's a real finding to discuss, not
  something to hide by only reporting the best seed.
- **Checkpointing:** save the policy periodically during training (e.g., every
  N episodes) so you can compare an early vs. late checkpoint, and so training
  can resume if interrupted.
- **Evaluation is separate from training:** periodically pause training and run
  the current policy in a pure evaluation mode (no exploration/randomness) to
  get a clean transit-time measurement — don't report numbers from episodes
  where the agent was still exploring randomly.

```python
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./checkpoints/",
    log_path="./logs/",
    eval_freq=1000,
    n_eval_episodes=5,
    deterministic=True,
)

model = DQN("MlpPolicy", train_env, seed=42, verbose=1)
model.learn(total_timesteps=100_000, callback=eval_callback)
```

## 7. Reproducibility

- **Fix random seeds everywhere** — `numpy`, `sklearn`, `torch`, and the RL
  library all need seeds set explicitly, or results won't be reproducible run to
  run (a reviewer re-running your code and getting different numbers is a real
  credibility problem).
- **Record library versions** — pin `scikit-learn`, `torch`, `stable-baselines3`,
  `sumo-rl` versions in `requirements.txt`. RL results in particular can shift
  meaningfully across library versions.

## 8. Experiment Tracking

You don't need a heavyweight MLOps platform for a student project, but you do
need a record of what was tried. Minimum viable version:

- A simple table/CSV logging: model type, hyperparameters, CV score, test score,
  date. Even a spreadsheet is fine — the point is having a record, not the tool.
- If you want a proper tool (optional, nice-to-have for the report's rigor):
  `mlflow` can track this automatically with minimal code changes
  (`mlflow.log_params()`, `mlflow.log_metrics()`), and produces comparison
  plots for free.

## 9. Model Versioning / Checkpointing

- Save trained models with `joblib.dump()` (tree-based models) or `torch.save()`
  (sequential/fusion models) — do not leave models as in-memory objects that
  vanish when the server restarts, which is the current MVP's pattern for the 4
  existing models. Persisting real trained models is itself an upgrade from the
  MVP.
- Name saved models with a version/date tag so you can compare "vitals_model_v1"
  vs. "vitals_model_v2" after a retraining pass, rather than overwriting silently.

## 10. Fairness/Bias Consideration (specific to Component 2 — healthcare data)

This is a genuinely important AIML engineering practice for clinical ML, and
worth a paragraph in the report even at student-project scale:

- Check whether the vitals dataset has skewed representation (e.g., by age
  group, if that data is available) and note this as a limitation if so — do
  not claim the model generalizes to populations underrepresented in MIMIC-III
  Demo's 100 patients.
- This is a "known limitations" section item for the model card
  (`02_ambulance_vitals_agent.md` Section 7), not something that needs to be
  solved — just needs to be acknowledged honestly.

## 11. Retraining Cadence (future-work framing, not required for this
submission)

Worth one line in the report as forward-looking context: in a real deployment,
this system would need a periodic retraining pipeline as new real ambulance
vitals/outcome data accumulates, with monitoring for model drift over time. This
is standard production ML practice — mention it as future work, do not attempt
to build an actual production retraining pipeline within the 2-month academic
timeline.

## 12. Checklist Summary

- [ ] Train/val/test split is stratified, test set touched only once
- [ ] Cross-validation used for Component 2's tabular models
- [ ] Hyperparameter search performed (grid or randomized), not guessed
- [ ] At least 3 model architectures compared for Component 2, with a
  justified final choice
- [ ] RL training curve logged and plotted, multiple seeds tried
- [ ] RL checkpointing implemented, evaluation separated from exploration
- [ ] Random seeds fixed everywhere, library versions pinned
- [ ] Simple experiment log maintained (spreadsheet or MLflow)
- [ ] Trained models persisted to disk with version tags, not left in-memory
- [ ] Known-limitations/bias paragraph included in the vitals model card
