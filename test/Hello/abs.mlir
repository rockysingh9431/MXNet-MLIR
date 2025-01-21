// RUN: MxNet-opt 
func.func @abs(%tensorA:tensor<2x3xf64>)-> tensor<2x3xf64>{
  
  %result = "hello.abs"( %tensorA ) : ( tensor<2x3xf64>) -> tensor<2x3xf64>
  
  return %result : tensor<2x3xf64>
}