# Clone and build
git clone <repository>
cd pymetallic

# Build Swift bridge and install
make build
make install-dev

# Test installation
python -c "import pymetal; pymetal.run_simple_compute_example()"
