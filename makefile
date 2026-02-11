# Makefile for llava.c project

# Compiler
CC = gcc

# Directories
SRC_DIR = src
INC_DIR = include
EXAMPLES_DIR = examples
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Detect compiler and adjust source files
ifeq ($(CC),cl)
    # Using MSVC cl compiler - include all files including win.c
    SOURCES = $(wildcard $(SRC_DIR)/*.c)
else
    # Using gcc or other compilers - exclude win.c and run.c (has main)
    SOURCES = $(filter-out $(SRC_DIR)/win.c $(SRC_DIR)/run.c,$(wildcard $(SRC_DIR)/*.c))
endif

OBJECTS_RELEASE = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/release/%.o,$(SOURCES))

# Target executables
TARGET_RUN_CLIP = $(BUILD_DIR)/run_clip
TARGET_RUN_PHI3 = $(BUILD_DIR)/run_phi3

# Compiler flags
CFLAGS = -std=c11 -fopenmp -I$(INC_DIR) -Wall
CFLAGS_RELEASE = $(CFLAGS) -O3 -DNDEBUG

# Linker flags
LDFLAGS = -fopenmp -lm

# Default target
.PHONY: all
all: release

# Release build
.PHONY: release
release: $(TARGET_RUN_CLIP) $(TARGET_RUN_PHI3)

# Build run_clip executable
$(TARGET_RUN_CLIP): $(OBJECTS_RELEASE) $(OBJ_DIR)/release/run_clip.o
	@mkdir -p $(BUILD_DIR)
	$(CC) $(OBJECTS_RELEASE) $(OBJ_DIR)/release/run_clip.o -o $@ $(LDFLAGS)
	@echo "Built release executable: $@"

$(OBJ_DIR)/release/run_clip.o: $(EXAMPLES_DIR)/run_clip.c
	@mkdir -p $(OBJ_DIR)/release
	$(CC) $(CFLAGS_RELEASE) -c $< -o $@

# Build run_phi3 executable
$(TARGET_RUN_PHI3): $(OBJECTS_RELEASE) $(OBJ_DIR)/release/run_phi3.o
	@mkdir -p $(BUILD_DIR)
	$(CC) $(OBJECTS_RELEASE) $(OBJ_DIR)/release/run_phi3.o -o $@ $(LDFLAGS)
	@echo "Built release executable: $@"

$(OBJ_DIR)/release/run_phi3.o: $(EXAMPLES_DIR)/run_phi3.c
	@mkdir -p $(OBJ_DIR)/release
	$(CC) $(CFLAGS_RELEASE) -c $< -o $@

# Compile shared source files
$(OBJ_DIR)/release/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)/release
	$(CC) $(CFLAGS_RELEASE) -c $< -o $@

# Individual targets
.PHONY: run_clip
run_clip: $(TARGET_RUN_CLIP)

.PHONY: run_phi3
run_phi3: $(TARGET_RUN_PHI3)

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all         - Build all targets (default)"
	@echo "  run_clip    - Build run_clip executable"
	@echo "  run_phi3    - Build run_phi3 executable"
	@echo "  release     - Build both executables"
	@echo "  clean       - Clean build directory"
	@echo "  help        - Show this help message"
