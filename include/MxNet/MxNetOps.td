#ifndef MXNET_OPS
#define MXNET_OPS

include "MxNetDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AddOp : MxNet_Op<"add",[Pure]>{
  let summary="add two tensors element-wise";
  let description=[{
    add two tensors element wise.
  }];
  let arguments = (ins AnyTensor:$lhs,AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}

def AbsOp : MxNet_Op<"abs",[Pure]>{
  let summary="perform absolute operation on single input value";
  let description=[{
    perform absolute operation on single tensor input
  }];

  let arguments=(ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}

def ReciprocalSqrtOp : MxNet_Op<"rsqrt",[Pure]>{
  let summary="perform operation to find reciprocal of square root";
  let description=[{
    perform reciprocal of a square root on single tensor input
  }];

  let arguments=(ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}
#endif // MXNET_OPS
