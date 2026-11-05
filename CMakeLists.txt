cmake_minimum_required(VERSION 3.9)
project(cgif LANGUAGES CXX)

# ---------------------------------------------------------------------------
# CGIF: Combining Proximity Graphs and Inverted Files for Efficient Filtered
# Vector Search over Arbitrary Predicates.
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)

# --- Compiler flags -------------------------------------------------------
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-Ofast -std=c++20 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "-Ofast -std=c++20 -DHAVE_CXX0X -fpic -ftree-vectorize")
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
    if (COMPILER_SUPPORT_NATIVE_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc")
endif()
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# --- Build options --------------------------------------------------------
# NOSIMD: disable manual SIMD kernels in the HNSW distance functions.
option(NOSIMD "Disable manual SIMD vectorization in HNSW" OFF)
if (NOSIMD)
    add_definitions(-DNO_MANUAL_VECTORIZATION)
    add_definitions(-DKGRAPH_NO_VECTORIZE)
endif()

# FAST_MODE: skip the internal per-query profiling/logging counters. Turn ON
# for benchmarking (higher QPS); keep OFF to collect the detailed statistics
# printed by cgif_search.
option(FAST_MODE "Skip internal profiling counters (faster)" OFF)
if (FAST_MODE)
    add_definitions(-DFAST_MODE)
endif()

# --- Dependencies ---------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Faiss is vendored as a git submodule under third_party/faiss and must be
# built first (see README). We link the resulting static library.
set(FAISS_ROOT "${PROJECT_SOURCE_DIR}/third_party/faiss")
add_library(libfaiss STATIC IMPORTED)
set_target_properties(libfaiss PROPERTIES
    IMPORTED_LOCATION "${FAISS_ROOT}/build/faiss/libfaiss.a")

# --- Include paths --------------------------------------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/common
    ${PROJECT_SOURCE_DIR}/include/base_hnsw
    ${PROJECT_SOURCE_DIR}/src
    ${FAISS_ROOT}
    ${Boost_INCLUDE_DIRS})

# --- UTIL: shared data-loading / utility library --------------------------
file(GLOB UTIL_SRCS ${PROJECT_SOURCE_DIR}/src/common/*.cc)
add_library(UTIL STATIC ${UTIL_SRCS})
target_link_libraries(UTIL PUBLIC OpenMP::OpenMP_CXX)

# --- Executables ----------------------------------------------------------
# build_ivf_index: build the IVF (coarse-quantizer) index consumed by CGIF.
add_executable(build_ivf_index apps/build_ivf_index.cc)
target_link_libraries(build_ivf_index
    UTIL OpenMP::OpenMP_CXX libfaiss ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# cgif_search: build the CGIF graph (-build_only) and run filtered queries.
add_executable(cgif_search apps/cgif_search.cc)
target_link_libraries(cgif_search
    UTIL OpenMP::OpenMP_CXX libfaiss ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# postfilter_hnsw: PostFilter-HNSW baseline (unfiltered top-ef, then filter);
# used to reproduce Fig. 8 (see reproduce/fig8/).
add_executable(postfilter_hnsw apps/postfilter_hnsw.cc)
target_link_libraries(postfilter_hnsw UTIL OpenMP::OpenMP_CXX)
