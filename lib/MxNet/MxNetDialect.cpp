// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "MxNet/MxNetDialect.h"
#include "MxNet/MxNetOps.h"

using namespace mlir;
using namespace MxNet;

//===----------------------------------------------------------------------===//
// MxNet dialect.
//===----------------------------------------------------------------------===//

#include "MxNet/MxNetOpsDialect.cpp.inc"

void MxNetDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MxNet/MxNetOps.cpp.inc"
      >();
}

void MxNet::ConstantOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  MxNet::ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::Operation *MxNetDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  return builder.create<MxNet::ConstantOp>(
      loc, type, mlir::cast<mlir::DenseElementsAttr>(value));
}