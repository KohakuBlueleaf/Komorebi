# Komorebi's native host module, _km_host, built inside a configured tt-metal tree (the
# one komorebi-runtime is built from) with tt-metal's own targets:
#   cmake -B <tree>/build -DCMAKE_PROJECT_Metalium_INCLUDE=<this file> [-DKM_HOST_OUTPUT_DIR=<dir>]
#   ninja -C <tree>/build _km_host
# The module links TT::Metalium, the static X280 host runtime and nanobind; it names no
# build directory, so it runs against any copy of the same libraries.

set(KM_NATIVE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "komorebi_native directory")
set(KM_HOST_OUTPUT_DIR "${PROJECT_BINARY_DIR}/komorebi" CACHE PATH "_km_host output directory")

# The targets exist only once the whole tree is configured: add the module at its end.
function(_km_add_host)
    foreach(_target TT::Metalium TT::Nanobind x280_device)
        if(NOT TARGET ${_target})
            message(FATAL_ERROR "_km_host: ${_target} is not configured (X280 runtime and Python bindings on?)")
        endif()
    endforeach()
    add_library(_km_host MODULE "${KM_NATIVE_DIR}/csrc/host/km_host.cpp")
    target_include_directories(
        _km_host
        PRIVATE
            "${KM_NATIVE_DIR}/include"
            "${KM_NATIVE_DIR}/x280/kernel"
            "${KM_NATIVE_DIR}/x280/engines"
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
    )
    target_link_libraries(_km_host PRIVATE TT::Nanobind x280_device TT::Metalium)
    # The module's sources by their path inside the package, not where they were built.
    target_compile_options(_km_host PRIVATE "-ffile-prefix-map=${KM_NATIVE_DIR}=komorebi_native")
    nanobind_opt_size(_km_host)
    nanobind_compile_options(_km_host)
    nanobind_link_options(_km_host)
    nanobind_extension_abi3(_km_host)
    set_target_properties(
        _km_host
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY
                "${KM_HOST_OUTPUT_DIR}"
            SKIP_BUILD_RPATH
                TRUE
    )
endfunction()

cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR} CALL _km_add_host)
