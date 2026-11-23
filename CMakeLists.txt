cmake_minimum_required(VERSION 3.20)
project(zcvid VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" _zcvid_is_top_level)
option(ZCVID_BUILD_TESTS "Build zcvid CTest suite" ${_zcvid_is_top_level})
option(ZCVID_BUILD_BENCH "Build zcvid latent hybrid benchmark" ${_zcvid_is_top_level})

add_library(zcvid INTERFACE)
add_library(zcvid::zcvid ALIAS zcvid)
target_include_directories(zcvid INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

set(ZCVID_SIMD_FLAGS "")
if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(ZCVID_SIMD_FLAGS "-mavx2")
endif()

if(NOT WIN32)
  find_package(Threads REQUIRED)
endif()

function(zcvid_configure_target target)
  target_link_libraries(${target} PRIVATE zcvid)
  if(NOT WIN32)
    target_link_libraries(${target} PRIVATE m Threads::Threads)
  endif()
  if(MSVC)
    target_compile_options(${target} PRIVATE /O2 /W3 /UNDEBUG /experimental:c11atomics)
  else()
    target_compile_options(${target} PRIVATE -O3 -UNDEBUG -Wall -Wextra ${ZCVID_SIMD_FLAGS})
  endif()
endfunction()

if(ZCVID_BUILD_TESTS)
  enable_testing()
  foreach(t IN ITEMS test_zcvid_packing test_zcvid_codec test_zcvid_proofs)
    add_executable(${t} tests/${t}.c)
    zcvid_configure_target(${t})
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()

if(ZCVID_BUILD_BENCH)
  add_executable(bench_latent_hybrid tests/bench_latent_hybrid.c)
  zcvid_configure_target(bench_latent_hybrid)
endif()
