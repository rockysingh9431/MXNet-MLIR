// RUN: MxNet-opt %s | FileCheck %s

func.func @add(%tensorA:tensor<2x3xf64>,%tensorB:tensor<2x3xf64>)-> tensor<2x3xf64>{

  // CHECK: %0 = tosa.add %arg0, %arg1 : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  %result = "MxNet.add"( %tensorA , %tensorB ) : ( tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  // CHECK: %0 : tensor<2x3xf64>
  return %result : tensor<2x3xf64>
}