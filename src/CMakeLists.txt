include_directories(${PROJECT_SOURCE_DIR}/include/)

add_library(gostcrypt cpu_kuznechik.c kuznechik.c ctr_encrypt.c)
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

