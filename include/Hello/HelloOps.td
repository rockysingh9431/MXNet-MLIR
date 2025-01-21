#ifndef HELLO_OPS
#define HELLO_OPS

include "HelloDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : Hello_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "hello.constant"()
      { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
      : () -> tensor<2x3xf64>
    ```
  }];

  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

//  let parser = [{ return ::parseConstantOp(parser, result); }];
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
}

def PrintOp : Hello_Op<"print", [Pure]> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];

    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def WorldOp : Hello_Op<"world", [Pure]> {
    let summary = "print Hello, World";
    let description = [{
        The "world" operation prints "Hello, World", and produces
        no results.
    }];
}

def AddOp : Hello_Op<"add",[Pure]>{
  let summary="add two tensors element-wise";
  let description=[{
    add two tensors element wise.
  }];
  let arguments = (ins AnyTensor:$lhs,AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}

def AbsOp : Hello_Op<"abs",[Pure]>{
  let summary="perform absolute operation on single input value";
  let description=[{
    perform absolute operation on single tensor input
  }];

  let arguments=(ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}

def ReciprocalSqrtOp : Hello_Op<"rsqrt",[Pure]>{
  let summary="perform operation to find reciprocal of square root";
  let description=[{
    perform reciprocal of a square root on single tensor input
  }];

  let arguments=(ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}

def MulOp : Hello_Op<"mul",[Pure]>{
  let summary="Perform multiply operation";
  let description=[{perform multiply of two tensors element-wise and return a reuslting tensor}];

  let arguments=(ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results=(outs AnyTensor:$result);
}
#endif // HELLO_OPS
