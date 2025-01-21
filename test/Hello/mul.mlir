// RUN: mlir-opt
func.func @add(%tensorA:tensor<2x3xf64>,%tensorB:tensor<2x3xf64>)-> tensor<2x3xf64>{
  
  %result = "hello.mul"( %tensorA , %tensorB ) : ( tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  return %result : tensor<2x3xf64>
}