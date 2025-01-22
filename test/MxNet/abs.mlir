// RUN: MxNet-opt %s | FileCheck %s

func.func @abs(%tensorA:tensor<2x3xf64>)-> tensor<2x3xf64>{
  //CHECK: tosa.abs
  %result = "MxNet.abs"( %tensorA ) : ( tensor<2x3xf64>) -> tensor<2x3xf64>

  return %result : tensor<2x3xf64>
}
