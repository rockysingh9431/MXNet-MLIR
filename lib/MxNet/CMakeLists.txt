add_mlir_dialect_library(MLIRMxNet
		MxNetDialect.cpp
		MxNetOps.cpp
		LowerToAffine.cpp
		LowerToLLVM.cpp
		LowerToTOSA.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/MxNet

        DEPENDS
        MLIRMxNetOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
