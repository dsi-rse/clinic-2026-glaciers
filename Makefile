
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
.PHONY: build-only run-interactive run-notebook labelstudio labelstudio-export

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
	docker compose up -d labelstudio
	./labelstudio/setup.sh

# Export the annotations to $(DATA_DIR)/labels/. The filename includes your username and a
# timestamp so that team members never overwrite each other's exports.
labelstudio-export:
	./labelstudio/export.sh


