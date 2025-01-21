// RUN: MxNet-opt %s | FileCheck
func.func @abs(%tensorA:tensor<2x3xf64>)-> tensor<2x3xf64>{
  
  %result = "hello.abs"( %tensorA ) : ( tensor<2x3xf64>) -> tensor<2x3xf64>
  // CHECK 
  return %result : tensor<2x3xf64>
}