# PyMetal Makefile
# Build system for Python Metal bindings

.PHONY: all build clean install test benchmark help

# Default target
all: build

# Variables
SWIFT_FILE = SwiftMetalBridge.swift
LIB_NAME = libpymetallic.dylib
PACKAGE_PREFIX ?= pymetallic
LIB_PATH=$(PACKAGE_PREFIX)/$(LIB_NAME)
PYTHON ?= python

# Check prerequisites
check:
	@echo "Checking prerequisites..."
	@command -v swift >/dev/null 2>&1 || { echo "Swift compiler not found. Please install Xcode or Swift toolchain."; exit 1; }
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "Python not found at $(PYTHON)"; exit 1; }
	@$(PYTHON) -c "import numpy" 2>/dev/null || { echo "NumPy not installed. Run: pip install numpy"; exit 1; }
	@echo "All prerequisites satisfied!"

# Format Python code
format:
	@echo "Formatting Python code..."
	$(PYTHON) -m black $(PACKAGE_PREFIX)/*.py examples/*.py  tests/*.py --line-length 88
	@echo "Formatting complete!"

# Type checking
typecheck:
	@echo "Running type checks..."
	$(PYTHON) -m mypy pymetallic --ignore-missing-imports
	@echo "Type checking complete!"

# Create distribution package
dist: clean build
	@echo "Creating distribution package..."
	$(PYTHON) -m build
	@echo "Distribution package created in dist/"

# Help target
help:
	@echo "PyMetal Build System"
	@echo "==================="
	@echo ""
	@echo "Available targets:"
	@echo "  build        - Build the Swift bridge library"
	@echo "  install      - Install library and Python package system-wide"
	@echo "  install-dev  - Install for development (user-only)"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Run basic functionality test"
	@echo "  examples     - Run comprehensive examples"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  check        - Check prerequisites"
	@echo "  format       - Format Python code with black"
	@echo "  typecheck    - Run type checking with mypy"
	@echo "  dist         - Create distribution package"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  PACKAGE_PREFIX - Installation prefix (default: /usr/local)"
	@echo "  PYTHON         - Python executable (default: python3) if we're on macOS"
UNAME_S := $(shell uname -s)
ifneq ($(UNAME_S),Darwin)
$(error PyMetal requires macOS with Metal support)
endif

# Build the Swift bridge library
build: $(LIB_PATH)
	@echo "Build complete!"

$(LIB_PATH): $(SWIFT_FILE)
	@echo "Compiling Swift Metal bridge..."
	swiftc -emit-library -o $(LIB_PATH) \
		-Xlinker -install_name -Xlinker @rpath/$(LIB_PATH) \
		$(SWIFT_FILE)
	@echo "Successfully built $(LIB_PATH)"

# Install the library
install: $(LIB_PATH)
	@echo "Installing Python package..."
	$(PYTHON) -m pip install -e .
	@echo "Installation complete!"

# Install for development (current user only)
install-dev: install
	@echo "Installing for development..."
	$(PYTHON) -m pip install -e .
	@echo "Development installation complete!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(LIB_PATH)
	rm -f *.dylib
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "Clean complete!"

# Run tests
test: $(LIB_PATH)
	@echo "Running tests..."
	$(PYTHON) -c "import pymetal; pymetal.run_simple_compute_example()"
	@echo "Basic test passed!"

# Run comprehensive examples
examples: $(LIB_PATH)
	@echo "Running comprehensive examples..."
	$(PYTHON) pymetal_examples.py

# Run performance benchmarks
benchmark: $(LIB_PATH)
	@echo "Running performance benchmarks..."
	$(PYTHON) -c "from pymetal_examples import PerformanceBenchmark; b = PerformanceBenchmark(); b.benchmark_matrix_multiply([128, 256, 512]); b.benchmark_vector_operations()"

# Check
