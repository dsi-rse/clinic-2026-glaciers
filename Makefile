
# general
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
current_abs_path := $(subst Makefile,,$(mkfile_path))

# pipeline constants
# PROJECT_NAME
project_name := "clinic-2026-glaciers"
project_dir := "$(current_abs_path)"

# environment variables
include .env

# Check required environment variables
ifeq ($(DATA_DIR),)
	$(error DATA_DIR must be set in .env file)
endif


# Build Docker image
.PHONY: build-only run-interactive run-notebook labelstudio labelstudio-export labelstudio-stop enhance-contrast enhance-contrast-local split-tiles split-tiles-local split-tiles-custom split-tiles-custom-local split-tiles-pine-island split-tiles-pine-island-local plot-split-steps plot-split-steps-local plot-ice-movement plot-ice-movement-local

# Build Docker image 
build-only: 
	docker compose build

run-interactive: build-only	
	docker compose run -it --rm $(project_name) /bin/bash

run-notebooks: build-only
	docker compose run --rm -p 8888:8888 -t $(project_name) \
	jupyter lab --port=8888 --ip='*' --NotebookApp.token='' --NotebookApp.password='' \
	--no-browser --allow-root

# Hand-labeling with Label Studio. See labelstudio/README.md.
labelstudio:
	docker compose up -d --build tileserver labelstudio
	./labelstudio/setup.sh

# Export the annotations to $(DATA_DIR)/labels/. The filename includes your username and a
# timestamp so that team members never overwrite each other's exports.
labelstudio-export:
	./labelstudio/export.sh

# Stop and remove the Label Studio and tile server containers. Annotations and the tile
# server's cached display ranges stay in their named volumes, so `make labelstudio` picks up
# where you left off. (`docker compose down -v` would delete them.)
labelstudio-stop:
	docker compose down labelstudio tileserver
	@echo "Label Studio stopped. Your annotations are kept; export them with 'make labelstudio-export'."

# Test contrast enhancement (CLAHE, high-pass) on NISAR tiles. See scripts/README.md.
# Docker (official setup): add script options with ARGS, for example
#   make enhance-contrast ARGS="--n-tiles 5 --sigma 30"
enhance-contrast: build-only
	docker compose run --rm $(project_name) python scripts/enhance_contrast.py $(ARGS)

# Same script with uv, for a local environment (no Docker).
enhance-contrast-local:
	uv run python scripts/enhance_contrast.py $(ARGS)

# Split the tiles into train and test with a checkerboard of square blocks, and draw a map
# for each block size: 10 km tiles (1024 px), then 41 km tiles (4096 px). See scripts/README.md.
# Docker (official setup)
split-tiles: build-only
	docker compose run --rm $(project_name) python scripts/split_tiles.py --block-km 30 60 120 --tile-size 1024 --plot
	docker compose run --rm $(project_name) python scripts/split_tiles.py --block-km 100 150 200 --tile-size 4096 --plot

# Same with uv, for a local environment (no Docker).
split-tiles-local:
	uv run python scripts/split_tiles.py --block-km 30 60 120 --tile-size 1024 --plot
	uv run python scripts/split_tiles.py --block-km 100 150 200 --tile-size 4096 --plot

# Split with your own options, for example
#   make split-tiles-custom ARGS="--region -1686.068 -362.187 -1561.229 -255.680 --tile-size 1024 --block-km 40.96 --plot"
split-tiles-custom: build-only
	docker compose run --rm $(project_name) python scripts/split_tiles.py $(ARGS)

split-tiles-custom-local:
	uv run python scripts/split_tiles.py $(ARGS)

# Pine Island comparison: the original bounding box, 10 and 20 km tiles (1024 and 2048 px),
# four block sizes, no gap, with maps. See scripts/README.md.
PINE_ISLAND_ARGS := --region -1686.068 -362.187 -1561.229 -255.680 --tile-size 1024 2048 --block-km 30.72 40.96 53.33 61.44 --gap-km 0 --plot

split-tiles-pine-island: build-only
	docker compose run --rm $(project_name) python scripts/split_tiles.py $(PINE_ISLAND_ARGS)

split-tiles-pine-island-local:
	uv run python scripts/split_tiles.py $(PINE_ISLAND_ARGS)

# Four pictures that show the split step by step (study area, blocks, sample tiles, sample
# tiles after the split). One tile size and one block size per run. See scripts/README.md.
#   make plot-split-steps ARGS="--region -1686.068 -362.187 -1561.229 -255.680 --tile-size 1024 --block-km 40.96"
plot-split-steps: build-only
	docker compose run --rm $(project_name) python scripts/plot_split_steps.py $(ARGS)

plot-split-steps-local:
	uv run python scripts/plot_split_steps.py $(ARGS)

# Plot all the dates of one stack to see how the ice moves. See scripts/README.md.
# Docker (official setup): give the stack id with ARGS, for example
#   make plot-ice-movement ARGS="--stack-id 077_A_140"
plot-ice-movement: build-only
	docker compose run --rm $(project_name) python scripts/plot_ice_movement.py $(ARGS)

# Same with uv, for a local environment (no Docker).
plot-ice-movement-local:
	uv run python scripts/plot_ice_movement.py $(ARGS)
