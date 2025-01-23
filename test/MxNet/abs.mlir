// RUN: MxNet-opt %s | FileCheck %s

//CHECK-LABEL: func.func @abs_f16(%arg0: tensor<2x3xf16>) -> tensor<2x3xf16> {
//CHECK-NEXT: %0 = tosa.cast %arg0 : (tensor<2x3xf16) -> tensor<2x3xf32>
//CHECK-NEXT: %1 = tosa.abs %0 : (tensor<2x3xf32>) -> tensor<2x3xf32>
//CHECK-NEXT: %2 = tosa.cast %1 : (tensor<2x3xf32>) -> tensor<2x3xf16>
//CHECK-NEXT: %2 : tensor<2x3xf16>
//CHECK-NEXT: }
func.func @abs_f16(%tensorA:tensor<2x3xf16>)-> tensor<2x3xf16>{
  %result = "MxNet.abs"( %tensorA ) : ( tensor<2x3xf16>) -> tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}
