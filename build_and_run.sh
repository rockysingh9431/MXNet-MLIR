#!/bin/bash
# This script helps set up MLIR and build mlir-hello, including dependencies and tests.

# Set the LLVM repository path (relative or absolute).
LLVM_REPO=./home/mcw/Projects/llvm-project
# Define the build directory for LLVM.
BUILD_DIR=$LLVM_REPO/build
# Define the installation directory for LLVM.
INSTALL_DIR=$LLVM_REPO/install

# Remove any existing build directory to start fresh.
rm -r $BUILD_DIR
# Create a new build directory for LLVM.
mkdir $BUILD_DIR
# Remove any existing install directory.
rm -r $INSTALL_DIR
# Create a new install directory for LLVM.
mkdir $INSTALL_DIR

# Stop the script on any error.
set -e

# Run cmake to configure the LLVM project with MLIR and other components.
cmake "-H$LLVM_REPO/llvm" \              # Specify the source directory of LLVM.
     "-B$BUILD_DIR" \                    # Specify the build directory.
     -DLLVM_INSTALL_UTILS=ON \           # Enable installation of utility tools like lli.
     -DLLVM_ENABLE_PROJECTS="mlir;clang" \  # Enable MLIR and Clang projects.
     -DLLVM_INCLUDE_TOOLS=ON \           # Include LLVM tools in the build.
     -DLLVM_BUILD_EXAMPLES=ON \          # Enable building examples.
     -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \  # Specify target architectures (X86, NVPTX, AMDGPU).
     -DCMAKE_BUILD_TYPE=Release \        # Set the build type to Release for optimized builds.
     -DLLVM_ENABLE_ASSERTIONS=ON \       # Enable assertions in the build for debugging.
     -DLLVM_ENABLE_RTTI=ON \             # Enable Run-Time Type Information (RTTI).
     -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR \  # Specify installation directory for LLVM.
                 # -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON # (Optional) Use Clang as the compiler and enable lld linker.

# Build LLVM and run the "check-mlir" target to verify MLIR functionality.
cmake --build $BUILD_DIR --target check-mlir -j 10 # Use up to 10 threads for building.

# Uncomment the following line to install LLVM if needed.
# cmake --build $BUILD_DIR --target install -j 10

# Build `lli` (LLVM interpreter), which is not included by default in the standard build.
pushd $BUILD_DIR # Navigate to the build directory.
make lli         # Build the lli tool for testing MLIR-generated IR.
popd             # Return to the previous directory.

# Set up the mlir-hello project.
mkdir build && cd build # Create and navigate to the build directory for mlir-hello.
cmake -G Ninja .. \     # Configure the mlir-hello project with Ninja as the generator.
  -DLLVM_DIR=$LLVM_REPO/build/lib/cmake/llvm \ # Point to the LLVM CMake configuration.
  -DMLIR_DIR=$LLVM_REPO/build/lib/cmake/mlir \ # Point to the MLIR CMake configuration.

# Build the `hello-opt` tool from the mlir-hello project.
cmake --build . --target mxNet-opt

# Run the `hello-opt` tool on the provided MLIR input file and save the output.
./build/bin/mxNet-opt ./test/Hello/print.mlir > print.ll

# Execute the generated LLVM IR (print.ll) using the `lli` tool.
$BUILD_DIR/bin/lli print.ll
