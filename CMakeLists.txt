cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tinylm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

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

set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/inference.proto")
set(PROTO_SRC "${CMAKE_CURRENT_BINARY_DIR}/inference.pb.cc")
set(PROTO_HDR "${CMAKE_CURRENT_BINARY_DIR}/inference.pb.h")
set(GRPC_SRC "${CMAKE_CURRENT_BINARY_DIR}/inference.grpc.pb.cc")
set(GRPC_HDR "${CMAKE_CURRENT_BINARY_DIR}/inference.grpc.pb.h")

add_custom_command(
    OUTPUT ${PROTO_SRC} ${PROTO_HDR} ${GRPC_SRC} ${GRPC_HDR}
    COMMAND protobuf::protoc
    ARGS --grpc_out=${CMAKE_CURRENT_BINARY_DIR}
         --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         -I ${CMAKE_CURRENT_SOURCE_DIR}
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.h" "src/*.hpp")

add_executable(sandbox ${SOURCES} ${HEADERS} ${PROTO_SRC} ${PROTO_HDR} ${GRPC_SRC} ${GRPC_HDR})

target_include_directories(sandbox PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
    ${SENTENCEPIECE_INCLUDE_DIR}
)

message(STATUS "Forcing Link Order: Protobuf -> gRPC -> Torch -> SentencePiece")

target_link_libraries(sandbox 
    ${Protobuf_LIBRARIES}

    gRPC::grpc++
    gRPC::grpc++_reflection

    "${TORCH_LIBRARIES}"

    ${SENTENCEPIECE_LIBRARY}
)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET sandbox POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:sandbox>)
endif (MSVC)