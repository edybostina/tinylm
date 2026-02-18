LIBTORCH_PATH := "/opt/homebrew/Cellar/"

BUILD_DIR := "build"

default: build

configure:
    mkdir -p {{BUILD_DIR}}
    cd {{BUILD_DIR}} && cmake -DCMAKE_PREFIX_PATH={{LIBTORCH_PATH}} ..

build: configure
    cd {{BUILD_DIR}} && cmake --build . --config Release --parallel

clean:
    rm -rf {{BUILD_DIR}}

rebuild: clean build

run: build
    ./{{BUILD_DIR}}/sandbox

debug-config:
    mkdir -p {{BUILD_DIR}}
    cd {{BUILD_DIR}} && cmake -DCMAKE_PREFIX_PATH={{LIBTORCH_PATH}} -DCMAKE_BUILD_TYPE=Debug ..

debug: debug-config
    cd {{BUILD_DIR}} && cmake --build . --config Debug

info:
    @echo "Build directory: {{BUILD_DIR}}"
    @echo "LibTorch path: {{LIBTORCH_PATH}}"
    @echo "Binary output: {{BUILD_DIR}}/sandbox"
