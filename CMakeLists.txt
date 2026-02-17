cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tinylm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)

find_library(SENTENCEPIECE_LIBRARY 
    NAMES sentencepiece
    PATHS /opt/homebrew/lib /usr/local/lib /usr/lib
    REQUIRED
)

find_path(SENTENCEPIECE_INCLUDE_DIR
    NAMES sentencepiece_processor.h
    PATHS /opt/homebrew/include /usr/local/include /usr/include
    REQUIRED
)

if (APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.h" "src/*.hpp")

add_executable(sandbox ${SOURCES} ${HEADERS})

target_include_directories(sandbox PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${SENTENCEPIECE_INCLUDE_DIR}
)

target_link_libraries(sandbox "${TORCH_LIBRARIES}" ${SENTENCEPIECE_LIBRARY})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET sandbox
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:sandbox>)
endif (MSVC)