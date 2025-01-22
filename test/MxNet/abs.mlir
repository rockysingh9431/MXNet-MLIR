// RUN: MxNet-opt %s | FileCheck %s

func.func @abs(%tensorA:tensor<2x3xf64>)-> tensor<2x3xf64>{
  //CHECK: %0 = tosa.abs %arg0 : (tensor<2x3xf64>) -> tensor<2x3xf64>
  %result = "MxNet.abs"( %tensorA ) : ( tensor<2x3xf64>) -> tensor<2x3xf64>

  // CHECK: %0 : tensor<2x3xf64>
  return %result : tensor<2x3xf64>
}
