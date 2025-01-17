#!/bin/bash
# Exit the script immediately if a command fails.
set -e

# Pull the latest changes from the main branch of the LLVM repository and integrate them as a subtree.
# The `--squash` option combines all commits into a single one, and the result is placed in the specified prefix directory.
git subtree pull --squash --prefix thirdparty/llvm-project git@github.com:llvm/llvm-project.git main

# Navigate to the build directory of the LLVM project under thirdparty.
pushd $PWD/thirdparty/llvm-project/build

# Run cmake to configure the LLVM project with MLIR and other settings.
cmake -G Ninja ../llvm \                     # Use Ninja as the build system generator and point to the LLVM source directory.
   -DLLVM_ENABLE_PROJECTS=mlir \             # Enable the MLIR project within LLVM.
   -DLLVM_BUILD_EXAMPLES=ON \                # Build example programs in the LLVM project.
   -DLLVM_TARGETS_TO_BUILD="host" \          # Build only the target corresponding to the current host system.
   -DCMAKE_BUILD_TYPE=Release \              # Set the build type to Release for optimized builds.
   -DLLVM_ENABLE_ASSERTIONS=ON -Wno-dev      # Enable runtime assertions for debugging and suppress CMake developer warnings.

# Build and run the `check-mlir` target to verify MLIR functionality.
cmake --build . --target check-mlir

# Return to the previous directory before `pushd`.
popd

# Navigate to the build directory for the main project.
pushd $PWD/build

# Run cmake to configure the main project with the required paths for LLVM and MLIR.
cmake -G Ninja .. \                                        # Use Ninja as the build system generator.
    -DLLVM_DIR=$PWD/../thirdparty/llvm-project/build/lib/cmake/llvm \ # Specify the LLVM configuration directory.
    -DMLIR_DIR=$PWD/../thirdparty/llvm-project/build/lib/cmake/mlir \ # Specify the MLIR configuration directory.
    -Wno-dev                                              # Suppress CMake developer warnings.

# Build and run the `check-hello` target to verify functionality of the "hello" project.
cmake --build . --target check-hello
