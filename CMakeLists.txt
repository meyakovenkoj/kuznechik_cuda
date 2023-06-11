cmake_minimum_required (VERSION 3.9)

project ("gostcrypt")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_compile_definitions(CPU_PROG)

include_directories(include)

add_subdirectory(src)
