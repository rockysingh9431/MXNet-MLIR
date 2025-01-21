configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MXNET_TEST_DEPENDS
        FileCheck count not
        mlir-opt
        MxNet-opt
        )

add_lit_testsuite(check-mxnet "Running the mxnet regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${MXNET_TEST_DEPENDS}
        )
set_target_properties(check-mxnet PROPERTIES FOLDER "Tests")

add_lit_testsuites(MXNET ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MXNET_TEST_DEPENDS})
