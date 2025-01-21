// #include "Hello/HelloDialect.h"
// #include "Hello/HelloOps.h"
// #include "Hello/HelloPasses.h"

// #include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Transforms/DialectConversion.h"

// #include "mlir/Dialect/Tensor/IR/Tensor.h"
// #include "mlir/Dialect/Tosa/IR/TosaOps.h"
// #include "mlir/Dialect/Traits.h"
// #include "mlir/IR/Matchers.h"

// #include "mlir/Dialect/Tosa/IR/TosaOps.h"
// #include "mlir/IR/PatternMatch.h"
// using namespace mlir;

// class ConvertAtenUnaryFPOnlyOp : public OpConversionPattern<AtenOpT> {
// public:
//   using OpConversionPattern<AtenOpT>::OpConversionPattern;
//   using OpAdaptor = typename AtenOpT::Adaptor;
//   LogicalResult
//   matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     Value self = adaptor.getSelf();
//     auto selfTy = self.getType().cast<TensorType>();

//     if (!selfTy)
//       return rewriter.notifyMatchFailure(op,
//                                          "Only Tensor types supported in
//                                          TOSA");

//     if (selfTy.getElementType().isa<mlir::FloatType>()) {
//       rewriter.replaceOpWithNewOp<TosaOpT>(
//           op,
//           OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
//               op.getType()),
//           self);
//       return success();
//     } else {
//       return rewriter.notifyMatchFailure(
//           op, "Only floating-point datatype legalization supported");
//     }
//   }
// };
// template <typename UnaryOpT, typename TosaOp>
// class LowerUnaryOp : public OpConversonPattern<UnaryOpT>{
//   public:
//   using OpConversionPattern<UnaryOpT>::OpConversionPattern;
//   using OpAdaptor = typename UnaryOpT::Adaptor;
//   LogicalResult matchAndRewrite(UnaryOpT op, OpAdaptor
//   adaptor,ConversionPatternRewriter &rewrite) override{
//     Value input = adaptor.getInput();
//   }
// };