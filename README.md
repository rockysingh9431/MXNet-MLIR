# MLIR MXNet Dialect

This is the minimal example to look into the way to implement the MxNet toolkit kind of program with MLIR. The basic code structure is borrowed from [standalone](https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone) and [Toy language](https://github.com/llvm/llvm-project/tree/main/mlir/examples/toy) in LLVM project.

We constantly check the compatibility with the latest LLVM/MLIR in [the nightly build](https://github.com/Lewuathe/mlir-MxNet/actions/workflows/nightly-build.yml). The status of the build is shown in the badge above.

## Prerequisites

* [LLVM](https://llvm.org/)
* [MLIR](https://mlir.llvm.org/)
* [CMake](https://cmake.org/)
* [Ninja](https://ninja-build.org/)

We need to build our own MLIR in the local machine in advance. Please follow the build instruction for MLIR [here](https://mlir.llvm.org/getting_started/). 

## Building

Please make sure to build LLVM project first according to [the instruction](https://mlir.llvm.org/getting_started/).

```sh
mkdir build && cd build
cmake -G Ninja .. -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir

cmake --build . --target MxNet-opt
```

To run the test, `check-mxnet` target will be usable.

To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
