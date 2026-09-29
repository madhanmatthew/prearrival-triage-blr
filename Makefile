# Command targets from docs/09 §3. Stubs print "not implemented" until the module exists.
# Windows: run from Git Bash (make ships with Git for Windows via `choco install make`, or use WSL).

PY ?= python
SEED ?= 1

.PHONY: help test run-api data-mimic train-baselines train-seq train-fusion train-text \
        build-rag eval-rag eval-asr sumo-build rl-baseline rl-train rl-eval benchmark demo

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1

test:
	$(PY) -m pytest -q

run-api:
	uvicorn backend.main:app --reload

# ---- Component 2: vitals (Chetan) ----
data-mimic:
	@echo "[TODO] build hourly windows + labels from MIMIC demo (docs/08 §2) -> backend/data/mimic_windows.py"
train-baselines:
	@echo "[TODO] majority, NEWS2, LR, RF, GB with nested grouped CV (docs/15 D5)"
train-seq:
	@echo "[TODO] BiGRU two-head (docs/07 §4) -> backend/ml/seq_model.py"
train-fusion:
	@echo "[TODO] fusion + 3-way ablation + sensitivity table -> backend/ml/fusion_model.py"

# ---- Component 1: reporting (Sankalp) ----
train-text:
	@echo "[TODO] TF-IDF char n-gram + LR text severity (docs/15 D2)"
build-rag:
	@echo "[TODO] corpus -> chunk -> embed -> FAISS (backend/rag/)"
eval-rag:
	@echo "[TODO] retrieval hit@k + groundedness %"
eval-asr:
	@echo "[TODO] Whisper WER per language (backend/asr/)"

# ---- Component 3: traffic RL (Madhan) ----
sumo-build:
	netconvert --node-files sumo/corridor.nod.xml --edge-files sumo/corridor.edg.xml -o sumo/corridor.net.xml
rl-baseline:
	@echo "[TODO] fixed-time + always-green runs, log transit + general delay (docs/09 §4)"
rl-train:
	@echo "[TODO] DQN seed=$(SEED) with checkpoints + eval callback (backend/rl/train.py)"
rl-eval:
	@echo "[TODO] deterministic eval of best checkpoint per seed"

# ---- Component 4 + integration (Ragavendra) ----
benchmark:
	$(PY) -m backend.benchmarks.benchmark
demo:
	@echo "[TODO] start API + frontend + demo scenario (docs/09 §6)"
