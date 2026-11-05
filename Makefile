# Convenience wrapper around the authoritative CMake build.  Do not compile
# individual source files here: mlx-c/Metal and all shared sources are required.
CMAKE ?= cmake
BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= Release
CMAKE_CONFIGURE = $(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DMLX_C_BUILD_EXAMPLES=OFF

.PHONY: all configure qwen3-asr resident-bench clean

all: qwen3-asr resident-bench

configure:
	$(CMAKE_CONFIGURE)

qwen3-asr: configure
	$(CMAKE) --build $(BUILD_DIR) --target qwen3-asr

resident-bench: configure
	$(CMAKE) --build $(BUILD_DIR) --target qwen3-asr-resident-bench

clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean
