# Command targets from docs/09 §3. Stubs print "not implemented" until the module exists.
# Windows: run from Git Bash (make ships with Git for Windows via `choco install make`, or use WSL).

PY ?= python
SEED ?= 1

.PHONY: help test run-api data-mimic train-baselines train-seq train-fusion train-text \
        build-rag eval-rag eval-asr sumo-build sumo-routes sumo-run rl-baseline rl-train rl-eval benchmark demo

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1

test:
	$(PY) -m pytest -q

run-api:
	uvicorn backend.main:app --reload

# ---- Component 2: vitals (Chetan) ----
data-mimic:
	$(PY) -m backend.data.mimic_windows
train-baselines:
	$(PY) -m backend.ml.baselines
train-seq:
	$(PY) -m backend.ml.seq_model
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
DEMAND ?= medium
sumo-build:
	netconvert --node-files sumo/corridor.nod.xml --edge-files sumo/corridor.edg.xml --type-files sumo/corridor.typ.xml -o sumo/corridor.net.xml
sumo-routes:
	for d in low medium peak; do $(PY) sumo/gen_routes.py --demand $$d; done
sumo-run:
	sumo -c sumo/corridor.sumocfg -r sumo/routes_$(DEMAND).rou.xml --seed $(SEED) --tripinfo-output sumo/tripinfo_$(DEMAND)_seed$(SEED).xml
collect-typical:
	$(PY) -m scripts.collect_tomtom_typical $(ARGS)
collect-traffic:
	$(PY) -m scripts.collect_tomtom $(ARGS)
count-vehicles:
	@echo "[TODO] YOLOv8 + ByteTrack counts from VIDEO=$(VIDEO) (docs/19 section 4)"
sumo-osm:
	@echo "[TODO] OSM -> netconvert -> sumo/osm/corridor.net.xml + tls_map.json (docs/19 section 1)"
sumo-demand:
	@echo "[TODO] routeSampler fit per slot -> sumo/demand/<slot>.rou.xml (docs/19 section 5)"
sumo-validate:
	@echo "[TODO] sim vs TomTom typical/live travel time per slot -> reports/rl_calibration.md (docs/19 section 6)"
rl-baseline:
	$(PY) -m backend.rl.baselines --episodes $(or $(EPISODES),20) --seed $(or $(SEED),42)
rl-train:
	@echo "[TODO] DQN seed=$(SEED) with checkpoints + eval callback (backend/rl/train.py)"
rl-eval:
	@echo "[TODO] deterministic eval of best checkpoint per seed"

# ---- Component 4 + integration (Ragavendra) ----
benchmark:
	@echo "[TODO] 8 + 2 scenarios with a fair naive baseline (docs/15 D15)"
demo:
	@echo "[TODO] start API + frontend + demo scenario (docs/09 §6)"
