// RUN: MxNet-opt 
func.func @add(%intA:tensor<*xf64>,%intB:tensor<*xf64>)-> tensor<*xf64>{
  
  %result = "hello.add"( %intA , %intB ) : ( tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
  
  return %result : tensor<*xf64>
}