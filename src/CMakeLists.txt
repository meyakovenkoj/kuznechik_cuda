include_directories(${PROJECT_SOURCE_DIR}/include/)

if(CUDAToolkit_FOUND)
    set(CUDA_SOURCES cuda_helper.cu cuda_kuznechik.cu)
else()
    set(CPU_SOURCES cpu_kuznechik.c)
endif()

add_library(gostcrypt STATIC ${CUDA_SOURCES} ${CPU_SOURCES} kuznechik.c ctr_encrypt.c)
target_compile_features(gostcrypt PUBLIC cxx_std_11)
set_target_properties(gostcrypt PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gostcrypt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(gostcrypt
    PROPERTIES
    PUBLIC_HEADER "${PROJECT_SOURCE_DIR}/include/kuznechik.h;${PROJECT_SOURCE_DIR}/include/common.h"
)
install(TARGETS gostcrypt
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/gostcrypt
    )

add_executable(cpu_main.exe main.c input.c)
target_link_libraries(cpu_main.exe gostcrypt)

# test will not compile with cuda
if(NOT CUDAToolkit_FOUND)
    add_executable(cpu_test.exe cpu_test.c)
    target_link_libraries(cpu_test.exe gostcrypt)
endif()

add_executable(gen.exe generator.c input.c)
