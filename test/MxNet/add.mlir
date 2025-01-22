// RUN: MxNet-opt
func.func @add(%tensorA:tensor<2x3xf64>,%tensorB:tensor<2x3xf64>)-> tensor<2x3xf64>{
  %result = "MxNet.add"( %tensorA , %tensorB ) : ( tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  return %result : tensor<2x3xf64>
}