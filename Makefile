# Command targets from docs/09 §3. Stubs print "not implemented" until the module exists.
# Windows: run from Git Bash (make ships with Git for Windows via `choco install make`, or use WSL).

PY ?= python
SEED ?= 1

.PHONY: help test run-api data-mimic train-baselines train-seq train-fusion eval-ed train-text \
        explain build-rag eval-rag eval-asr eval-asr-noise asr-latency asr-gpu-day sumo-build sumo-routes sumo-run rl-baseline rl-train rl-eval demo-routes benchmark demo

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
	$(PY) -m backend.ml.fusion_model
eval-ed:
	$(PY) -m backend.ml.ed_external_eval $(ARGS)
explain:
	$(PY) -m backend.ml.explain $(ARGS)

# ---- Component 1: reporting (Sankalp) ----
train-text:
	$(PY) -m backend.ml.text_severity --save-model $(ARGS)
build-rag:
	$(PY) -m backend.rag.pipeline build $(ARGS)
eval-rag:
	$(PY) -m backend.rag.eval_rag $(ARGS)
eval-asr:
	$(PY) -m backend.asr.eval_wer $(ARGS)
eval-asr-noise:
	$(PY) -m backend.asr.eval_noise $(ARGS)
asr-latency:
	$(PY) -m backend.asr.latency $(ARGS)
asr-gpu-day:
	$(PY) scripts/asr_gpu_day.py $(ARGS)

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
	$(PY) -m backend.rl.train --seed $(or $(SEED),1) --timesteps $(or $(STEPS),100000)
rl-eval:
	$(PY) -m backend.rl.train --eval-only --seed $(or $(SEED),1) --model $(MODEL)

# ---- Component 4 + integration (Ragavendra) ----
demo-routes:
	$(PY) scripts/build_demo_routes.py $(ARGS)
benchmark:
	@echo "[TODO] 8 + 2 scenarios with a fair naive baseline (docs/15 D15)"
demo:
	@echo "[TODO] start API + frontend + demo scenario (docs/09 §6)"

frontend-install:  ## React UI deps
	cd web && npm install

frontend-dev:  ## React UI dev server (proxies to :8000)
	cd web && npm run dev

frontend-build:  ## build React UI -> web/dist (served at /app/)
	cd web && npm run build

frontend-test:
	cd web && npm test
