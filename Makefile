# General definitions
CC = nvcc # CUDA compiler
PYTHON3 = python3 # Python interpreter

# Compiler flags
CFLAGS = -g -G
INTERNAL_CFLAGS = \
	-std=c++14 \
	-Iinclude \
	$(CFLAGS)

# Default paths
SRC_DIR = src
TARGET_DIR = bin
INPUT_DIR = input
OUTPUT_DIR = output

# Sources
SRC_CUDA = $(SRC_DIR)/cuda_convolution.cu
SRC_LODEPNG = $(SRC_DIR)/lodepng.cpp

# Targets
TARGET_CUDA = $(TARGET_DIR)/cuda_convolution


.PHONY: setup all clean test help


setup:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully."


# Compile all targets
all: $(TARGET_CUDA)

# Compile CUDA code
$(TARGET_CUDA): $(SRC_CUDA) $(SRC_LODEPNG)
	$(CC) $(INTERNAL_CFLAGS) -o $(TARGET_CUDA) $(SRC_CUDA) $(SRC_LODEPNG)


# Run tests using Python wrapper
test: all
	@echo "Running tests using Python wrapper..."
	$(PYTHON3) run_tests.py


# Clean up generated files
clean:
	rm -f $(TARGET_CUDA) $(OUTPUT_DIR)/*.png
	rm -f ./graphs/*.png
	rm -f ./tables/*.csv


# Help command to list available targets
help:
	@echo "Available targets:"
	@echo "  setup         Install required Python dependencies."
	@echo "  all           Compile the CUDA program."
	@echo "  test          Run tests using Python wrapper."
	@echo "  clean         Remove compiled files and generated artifacts."
	@echo "  help          Display this help message."
