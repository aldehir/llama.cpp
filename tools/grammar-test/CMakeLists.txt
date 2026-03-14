set(TARGET llama-grammar-test)
add_executable(${TARGET} grammar-test.cpp)
target_include_directories(${TARGET} PRIVATE ../../src)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# This tool uses internal llama symbols (grammar internals) that are not
# exported from the DLL on Windows. Link the object files directly.
# The common library transitively links llama.lib (import lib), so we use
# /FORCE:MULTIPLE to let the linker resolve the duplicates.
if (BUILD_SHARED_LIBS AND WIN32)
    target_link_libraries(${TARGET} PRIVATE $<TARGET_OBJECTS:llama>)
    target_link_options(${TARGET} PRIVATE /FORCE:MULTIPLE)
endif ()
