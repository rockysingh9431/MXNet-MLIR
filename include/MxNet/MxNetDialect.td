#ifndef MXNET_DIALECT
#define MXNET_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MxNet dialect definition.
//===----------------------------------------------------------------------===//

def MxNet_Dialect : Dialect {
    let name = "MxNet";
    let summary = "A MxNet out-of-tree MLIR dialect.";
    let description = [{
        This dialect is minimal example to implement MxNet kind of sample code
        for MLIR.
    }];
    let cppNamespace = "::MxNet";
}

//===----------------------------------------------------------------------===//
// Base MxNet operation definition.
//===----------------------------------------------------------------------===//

class MxNet_Op<string mnemonic, list<Trait> traits = []> :
        Op<MxNet_Dialect, mnemonic, traits>;

#endif // MXNET_DIALECT
