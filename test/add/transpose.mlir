// RUN: hello-opt 
func.func @transpose(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "hello.transposeOp"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>

  return %t_tensor : tensor<3x2xf64>
}
