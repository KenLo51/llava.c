# Makefile for llava.c project

# Compiler
CC = gcc

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Detect compiler and adjust source files
ifeq ($(CC),cl)
    # Using MSVC cl compiler - include all files including win.c
    SOURCES = $(wildcard $(SRC_DIR)/*.c)
else
    # Using gcc or other compilers - exclude win.c
    SOURCES = $(filter-out $(SRC_DIR)/win.c,$(wildcard $(SRC_DIR)/*.c))
endif

OBJECTS_DEBUG = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/debug/%.o,$(SOURCES))
OBJECTS_RELEASE = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/release/%.o,$(SOURCES))

# Target executables
TARGET_DEBUG = $(BUILD_DIR)/run_debug
TARGET_RELEASE = $(BUILD_DIR)/run

# Compiler flags
CFLAGS = -std=c11 -fopenmp -I$(INC_DIR) -Wall
CFLAGS_DEBUG = $(CFLAGS) -g -DDEBUG
CFLAGS_RELEASE = $(CFLAGS) -O2 -DNDEBUG

# Linker flags
LDFLAGS = -fopenmp -lm

# Default target
.PHONY: all
all: release

# Release build
.PHONY: release
release: $(TARGET_RELEASE)

$(TARGET_RELEASE): $(OBJECTS_RELEASE)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(OBJECTS_RELEASE) -o $@ $(LDFLAGS)
	@echo "Built release executable: $@"

$(OBJ_DIR)/release/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)/release
	$(CC) $(CFLAGS_RELEASE) -c $< -o $@

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all         - Build debug version (default)"
	@echo "  release     - Build optimized release version"
	@echo "  clean       - Remove all build artifacts"
	@echo "  help        - Show this help message"
