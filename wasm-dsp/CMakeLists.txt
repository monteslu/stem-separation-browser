cmake_minimum_required(VERSION 3.13)
project(demucs_dsp)

set(CMAKE_CXX_STANDARD 17)

add_executable(demucs_dsp
    src/dsp.cpp
    src/api.cpp
)

target_include_directories(demucs_dsp PRIVATE src)

# Emscripten settings
if(EMSCRIPTEN)
    set_target_properties(demucs_dsp PROPERTIES
        SUFFIX ".js"
    )
    target_link_options(demucs_dsp PRIVATE
        "SHELL:-s EXPORTED_FUNCTIONS=[_compute_stft,_compute_istft_merge,_get_stft_frames,_get_stft_buffer_size,_malloc,_free]"
        "SHELL:-s EXPORTED_RUNTIME_METHODS=[ccall,cwrap]"
        "SHELL:-s MODULARIZE=1"
        "SHELL:-s EXPORT_NAME=DemucsModule"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
        "SHELL:-s INITIAL_MEMORY=134217728"
        "SHELL:-s WASM=1"
        -O3
    )
endif()
