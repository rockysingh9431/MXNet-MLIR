// RUN: MxNet-opt 
func.func @abs(%tensorA:tensor<2x3xf64>)-> tensor<2x3xf64>{

  //CHECK: %0 = tosa.rsqrt %arg0 : (tensor<2x3xf64>) -> tensor<2x3xf64>
  %result = "MxNet.rsqrt"( %tensorA ) : ( tensor<2x3xf64>) -> tensor<2x3xf64>
  
  // CHECK: %0 : tensor<2x3xf64>
  return %result : tensor<2x3xf64>
}