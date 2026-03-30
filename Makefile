IMAGE=maritime-tracker:local
COMPOSE=docker compose

build:
	$(COMPOSE) build

shell:
	$(COMPOSE) run --rm --entrypoint bash tracker

smoke-detect:
	$(COMPOSE) run --rm tracker detect \
		--model yolo11n.pt \
		--source /workspace/data/videos \
		--project /workspace/results/detect \
		--name smoke

track-botsort:
	$(COMPOSE) run --rm tracker track \
		--model /workspace/models/vessel.pt \
		--source /workspace/data/videos \
		--tracker /workspace/trackers/botsort_maritime.yaml \
		--project /workspace/results/track \
		--name botsort

track-bytetrack:
	$(COMPOSE) run --rm tracker track \
		--model /workspace/models/vessel.pt \
		--source /workspace/data/videos \
		--tracker /workspace/trackers/bytetrack_maritime.yaml \
		--project /workspace/results/track \
		--name bytetrack

train:
	$(COMPOSE) run --rm tracker train \
		--data /workspace/configs/vessel_data.yaml \
		--model yolo11n.pt \
		--project /workspace/results/train \
		--name vessel_yolo11n

evaluate-botsort:
	$(COMPOSE) run --rm tracker evaluate \
		--pred-dir /workspace/results/track/botsort/mot \
		--gt-dir /workspace/data/gt \
		--output /workspace/results/eval/botsort_metrics.csv

evaluate-bytetrack:
	$(COMPOSE) run --rm tracker evaluate \
		--pred-dir /workspace/results/track/bytetrack/mot \
		--gt-dir /workspace/data/gt \
		--output /workspace/results/eval/bytetrack_metrics.csv
