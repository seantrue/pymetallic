# PyMetallic Makefile
# Build system for Python Metal bindings

.PHONY: all build clean install test benchmark help

# Default target
all: check format typecheck install-dev test-cov

# Variables
SWIFT_FILE = SwiftMetalBridge.swift
LIB_NAME = libpymetallic.dylib
SRC_PATH=src
PACKAGE_PREFIX ?= $(SRC_PATH)/pymetallic
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
	$(PYTHON) -m black $(PACKAGE_PREFIX)/*.py $(PACKAGE_PREFIX)/scripts/*.py  $(SRC_PATH)/tests/*.py --line-length 88
	@echo "Formatting complete!"

# Make sure required packages are license compatible
licensecheck:
	@echo "Checking packages for compatibility..."
	licensecheck
	@echo "License check complete"

# Type checking
typecheck:
	@echo "Running type checks..."
	$(PYTHON) -m mypy src/pymetallic src/tests --ignore-missing-imports
	@echo "Type checking complete!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(LIB_PATH)
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete -print
	find . -name "*.pyo" -delete -print
	find src -name "*.dylib" -delete -print
	find src -name "*.gif" -delete -print
	@echo "Clean complete!"

$(LIB_PATH): $(SRC_PATH)/$(SWIFT_FILE)
	@echo "Compiling Swift Metal bridge..."
	swiftc -emit-library -o $(LIB_PATH) \
		-Xlinker -install_name -Xlinker @rpath/$(LIB_NAME) \
		$(SRC_PATH)/$(SWIFT_FILE)
	@echo "Successfully built $(LIB_PATH)"

# Build the Swift bridge library
build: $(LIB_PATH)
	@echo "Build complete!"

# Create distribution package
dist: clean build
	@echo "Creating distribution package..."
	$(PYTHON) -m build
	@echo "Distribution package created in dist/"

# Install the library
install: dist
	@echo "Installing Python package..."
	$(PYTHON) -m pip install dist/pymetallic*.whl
	@echo "Installation complete!"

# Install for development (current user only)
install-dev: uninstall
	@echo "Installing for development..."
	$(PYTHON) -m pip install -e .[dev,docs,test]
	@echo "Development installation complete!"

# Run smoke test
smoke: $(LIB_PATH)
	@echo "Running smoke tests..."
	$(PYTHON) -c "import pymetallic; pymetallic.run_simple_compute_example()"
	$(PYTHON) src/pymetallic/scripts/smoke.py
	@echo "Smoke tests passed!"

# Run test suite
test: $(LIB_PATH)
	@echo "Running tests..."
	py.test $(SRC_PATH)/tests
	@echo "Tests complete!"

# Run test suite with coverage
test-cov: $(LIB_PATH)
	@echo "Running tests..."
	py.test --cov --cov-report html:coverage src
	open coverage/index.html
	@echo "Tests complete!"


# Run performance benchmarks
benchmark: $(LIB_PATH)
	@echo "Running performance benchmarks..."
	$(PYTHON) -c "from pymetallic.helpers import PerformanceBenchmark; b = PerformanceBenchmark(); b.benchmark_matrix_multiply([128, 256, 512]); b.benchmark_vector_operations()"

# Run comprehensive examples
demos: $(LIB_PATH)
	@echo "Running all demos..."
	$(PYTHON) $(SRC_PATH)/scripts/demos.py --demos all

uninstall:
	@echo "Uninstalling Python package..."
	$(PYTHON) -m pip uninstall --yes pymetallic
	@echo "Uninstallation complete..."


# Help target
help:
	@echo "PyMetallic Build System"
	@echo "==================="
	@echo ""
	@echo "Available targets:"
	@echo "  check        - Check prerequisites"
	@echo "  format       - Format Python code with black"
	@echo "  typecheck    - Run type checking with mypy"
	@echo "  licensecheck - Check required package licenses"
	@echo "  clean        - Remove build artifacts"
	@echo "  build        - Build the Swift bridge library"
	@echo "  dist         - Create distribution package"
	@echo "  uninstall    - Uninstall package from site_packages"
	@echo "  install      - Install Python package from wheel to site_packages"
	@echo "  install-dev  - Install for development (user-only)"
	@echo "  smoke        - Run basic functional test"
	@echo "  test         - Run all tests"
	@echo "  test-cov     - Run all tests with coverage"
	@echo "  examples     - Run comprehensive examples"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  PACKAGE_PREFIX - Path to src directory (default: $(PACKAGE_PREFIX))"
	@echo "  PYTHON         - Python executable (default: python3) if we're on macOS"
UNAME_S := $(shell uname -s)
ifneq ($(UNAME_S),Darwin)
$(error PyMetallic requires macOS with Metal support)
endif