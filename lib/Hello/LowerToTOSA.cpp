#include "Hello/HelloDialect.h"
#include "Hello/HelloOps.h"
#include "Hello/HelloPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"

// Lowering of reciprocal of squareroot to tosa dialect

class ReciprocalSqrtOpLowering
    : public OpRewritePattern<hello::ReciprocalSqrtOp> {
public:
  using OpRewritePattern<hello::ReciprocalSqrtOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hello::ReciprocalSqrtOp op,
                                PatternRewriter &rewriter) const override {
    // Fetch operand
    Value input = op.getInput();

    // Ensure operand has compatible types
    auto resultType = op.getType();
    if (!resultType.isa<TensorType>()) {
      return rewriter.notifyMatchFailure(op, "Expected TensorType");
    }

    // Create a TOSA rsqrt operation
    auto tosaRsqrt =
        rewriter.create<tosa::RsqrtOp>(op.getLoc(), resultType, input);

    // Replace the original operation with the new TOSA operation
    rewriter.replaceOp(op, tosaRsqrt);

    return success();
  }
};
// Absolution operation lowering to tosa dialect
class AbsOpLowering : public OpRewritePattern<hello::AbsOp> {
public:
  using OpRewritePattern<hello::AbsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hello::AbsOp op,
                                PatternRewriter &rewriter) const override {
    // Fetch operand
    Value input = op.getInput();

    // Ensure operand has compatible types
    auto resultType = op.getType();
    if (!resultType.isa<TensorType>()) {
      return rewriter.notifyMatchFailure(op, "Expected TensorType");
    }

    // Create a TOSA abs operation
    auto tosaAbs = rewriter.create<tosa::AbsOp>(op.getLoc(), resultType, input);

    // Replace the original operation with the new TOSA operation
    rewriter.replaceOp(op, tosaAbs);

    return success();
  }
};

// Pattern to lower custom::AddOp to tosa.add
class AddOpLowering : public OpRewritePattern<hello::AddOp> {
public:
  using OpRewritePattern<hello::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hello::AddOp op,
                                PatternRewriter &rewriter) const override {
    // Fetch operands
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Ensure operands have compatible types
    auto resultType = op.getType();
    if (!resultType.isa<TensorType>()) {
      return rewriter.notifyMatchFailure(op, "Expected TensorType");
    }

    // Create a TOSA add operation
    auto tosaAdd =
        rewriter.create<tosa::AddOp>(op.getLoc(), resultType, lhs, rhs);

    // Replace the original operation with the new TOSA operation
    rewriter.replaceOp(op, tosaAdd);

    return success();
  }
};

namespace {
class LowerToTosaPass
    : public mlir::PassWrapper<LowerToTosaPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToTosaPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect, mlir::tosa::TosaDialect,
                    mlir::tensor::TensorDialect>();
  }

  void runOnOperation() final;
};
} // namespace

void LowerToTosaPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addIllegalDialect<hello::HelloDialect>();
  target.addLegalDialect<mlir::affine::AffineDialect, mlir::BuiltinDialect,
                         mlir::func::FuncDialect, mlir::arith::ArithDialect,
                         mlir::memref::MemRefDialect, mlir::tosa::TosaDialect,
                         mlir::tensor::TensorDialect>();
  target.addDynamicallyLegalOp<hello::PrintOp>([](hello::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(), [](mlir::Type type) {
      return mlir::isa<mlir::TensorType>(type);
    });
  });
  target.addLegalOp<hello::WorldOp>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, AbsOpLowering, ReciprocalSqrtOpLowering>(
      &getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> hello::createLowerToTosaPass() {
  return std::make_unique<LowerToTosaPass>();
}
