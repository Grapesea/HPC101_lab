cmake_minimum_required(VERSION 3.20)
project(amss_ncku_lab LANGUAGES CXX Fortran)

# --- Options ----------------------------------------------------------------
option(AMSS_ENABLE_GPU "Enable CUDA GPU build (ABEGPU target)" OFF)
# Default CPU build for the ARM TaiShan-v120 evaluation node.  Keep CUDA
# architecture selection separate below; AMSS_OPT applies to C++/Fortran.
set(AMSS_OPT "-Ofast -ffast-math" CACHE STRING "C/C++/Fortran optimization level")
set(AMSS_ARCH_FLAGS "-mcpu=native" CACHE STRING "Architecture-specific compiler flags")
# Default ON: the lab4 CPU sweep (results/4-96927.log) relies on OpenMP for
# TwoPunctureABE's 24-thread relaxation, so the submitted build must enable
# it even when compile.sh is invoked with no extra AMSS_* environment set.
option(AMSS_ENABLE_OPENMP "Enable OpenMP in CPU solvers" ON)
option(AMSS_ENABLE_SANITY_CHECK "Keep per-RHS full-array NaN scans" OFF)
set(AMSS_MPI_CUDA_AWARE 0 CACHE STRING "Set MPI_CUDA_AWARE macro for ABEGPU (0/1)")
set_property(CACHE AMSS_MPI_CUDA_AWARE PROPERTY STRINGS 0 1)
# Extra runtime library search paths baked into ABE/TwoPunctureABE's RPATH.
# Needed for the hybrid-toolchain CPU build: compile.sh stages a
# GNU-compiled TwoPunctureABE (faster TwoPuncture relaxation) inside the
# Arm-compiled build tree (faster Fortran-heavy ABE evolution). That GNU
# binary carries no RPATH for Arm's OpenMP runtime, and evaluation nodes may
# not have an unversioned libomp.so on the default linker search path, so
# without this it fails with "libomp.so: cannot open shared object file"
# once relocated. Colon-separated, e.g. "/opt/arm/.../lib:/opt/arm/.../lib64".
set(AMSS_EXTRA_RPATH "" CACHE STRING "Extra colon-separated runtime library search paths for ABE/TwoPunctureABE RPATH")

# --- Compile flags ----------------------------------------------------------
# Legacy AMSS-NCKU has strict-aliasing UB; -fno-strict-aliasing is kept
# regardless of AMSS_OPT so changing -O does not lose the safety flag.
# `-Dfortran3 -Dnewc` select the modern C/Fortran source branches (without
# them the C++ side falls back to pre-ANSI <iostream.h> includes and the
# Fortran side skips the f90 modules). Apply to every supported compiler.
separate_arguments(AMSS_OPT_LIST UNIX_COMMAND "${AMSS_OPT}")
separate_arguments(AMSS_ARCH_FLAGS_LIST UNIX_COMMAND "${AMSS_ARCH_FLAGS}")
if(AMSS_EXTRA_RPATH)
  string(REPLACE ":" ";" AMSS_EXTRA_RPATH_LIST "${AMSS_EXTRA_RPATH}")
endif()

# Clang/IntelLLVM/ARMClang all accept the GCC-style flags used here.
# COMPILE_LANG_AND_ID takes comma-separated compiler IDs (not CMake list).
add_compile_options(
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,IntelLLVM,Intel,ARMClang>:${AMSS_OPT_LIST};${AMSS_ARCH_FLAGS_LIST};-fno-strict-aliasing;-Wno-deprecated;-Dfortran3;-Dnewc>"
  "$<$<COMPILE_LANG_AND_ID:Fortran,GNU,Flang,IntelLLVM,Intel,ARMClang>:${AMSS_OPT_LIST};${AMSS_ARCH_FLAGS_LIST};-fno-strict-aliasing;-cpp>"
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

if(AMSS_ENABLE_SANITY_CHECK)
  add_compile_definitions(AMSS_SANITY_CHECK)
endif()

# --- MPI --------------------------------------------------------------------
find_package(MPI REQUIRED COMPONENTS CXX)

# --- Shared Fortran sources -------------------------------------------------
set(AMSS_F90SOURCES
  src/bssn_rhs.f90
  src/diff_new.f90 src/enforce_algebra.f90
  src/fadmquantites_bssn.f90 src/fmisc.f90
  src/getnp4.f90
  src/initial_puncture.f90 src/kodiss.f90
  src/lopsidediff.f90
  src/prolongrestrict_cell.f90 src/rungekutta4_rout.f90
  src/sommerfeld_rout.f90
)

# --- ABE: CPU main evolution executable -------------------------------------
set(ABE_SOURCES
  src/ABE.C src/Ansorg.C src/Block.C src/MPatch.C src/Parallel.C
  src/bssn_class.C src/cgh.C src/checkpoint.C
  src/misc.C src/monitor.C src/perf.C
  src/surface_integral.C src/transpbh.C src/var.C
  ${AMSS_F90SOURCES}
)
add_executable(ABE ${ABE_SOURCES})
target_link_libraries(ABE PRIVATE MPI::MPI_CXX)
# The Fortran stencil kernels below are linked with OpenMP when requested.

# --- TwoPunctureABE: initial-data solver -----------------------------------
add_executable(TwoPunctureABE src/TwoPunctureABE.C src/TwoPunctures.C)
target_link_libraries(TwoPunctureABE PRIVATE MPI::MPI_CXX)

# --- Extra RPATH (hybrid-toolchain runtime library fix, see AMSS_EXTRA_RPATH
# above) -----------------------------------------------------------------
if(AMSS_EXTRA_RPATH_LIST)
  set_target_properties(ABE TwoPunctureABE PROPERTIES
    BUILD_RPATH "${AMSS_EXTRA_RPATH_LIST}"
    INSTALL_RPATH "${AMSS_EXTRA_RPATH_LIST}")
endif()

# --- Optional OpenMP --------------------------------------------------------
if(AMSS_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX Fortran)
  target_link_libraries(ABE PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_Fortran)
  target_link_libraries(TwoPunctureABE PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- ABEGPU: GPU evolution executable (device-resident baseline) -----------
# Same C++/Fortran sources as ABE except bssn_class.C → bssn_gpu_class.C,
# plus GPU kernel/agent files.
set(ABEGPU_CXX_SOURCES
  src/ABE.C src/Ansorg.C src/Block.C src/MPatch.C src/Parallel.C
  src/bssn_gpu_class.C src/bssn_step_gpu.C
  src/cgh.C src/checkpoint.C src/misc.C
  src/monitor.C src/perf.C src/surface_integral.C
  src/transpbh.C src/var.C
  src/helper.cpp src/Parallel_GPU.cpp
)
set(ABEGPU_CUDA_SOURCES
  src/fmisc_gpu.cu src/lopsidediff_gpu.cu src/kodiss_gpu.cu src/diff_new_gpu.cu
  src/bssn_rhs_gpu.cu src/prolongrestrict_cell_gpu.cu src/gpu_manager.cu
  src/enforce_algebra_gpu.cu src/sommerfeld_rout_gpu.cu src/rungekutta4_rout_gpu.cu
  src/fadmquantites_bssn_gpu.cu src/misc_gpu.cu src/MPatch_gpu.cu
  src/surface_integral_gpu.cu src/getnp4_gpu.cu
)

if(AMSS_ENABLE_GPU)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures" FORCE)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  find_package(CUDAToolkit REQUIRED)

  add_executable(ABEGPU ${ABEGPU_CXX_SOURCES} ${ABEGPU_CUDA_SOURCES} ${AMSS_F90SOURCES})
  set_source_files_properties(${ABEGPU_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
  set_target_properties(ABEGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(ABEGPU PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-rdc=true;-lineinfo>")
  target_compile_definitions(ABEGPU PRIVATE USE_GPU fortran3 newc MPI_CUDA_AWARE=${AMSS_MPI_CUDA_AWARE})
  target_include_directories(ABEGPU PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(ABEGPU PRIVATE CUDA::cudart MPI::MPI_CXX)
  if(AMSS_ENABLE_OPENMP)
    target_link_libraries(ABEGPU PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_Fortran)
  endif()
endif()

# --- Configure summary ------------------------------------------------------
message(STATUS "==================== AMSS-NCKU Lab 4 ====================")
message(STATUS "CXX      : ${CMAKE_CXX_COMPILER}")
message(STATUS "Fortran  : ${CMAKE_Fortran_COMPILER}")
if(AMSS_ENABLE_GPU)
  message(STATUS "GPU      : ON, CUDA=${CMAKE_CUDA_COMPILER}, arch=${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "GPU      : OFF")
endif()
message(STATUS "AMSS_OPT=${AMSS_OPT} ARCH_FLAGS=${AMSS_ARCH_FLAGS} OPENMP=${AMSS_ENABLE_OPENMP} SANITY_CHECK=${AMSS_ENABLE_SANITY_CHECK} MPI_CUDA_AWARE=${AMSS_MPI_CUDA_AWARE}")
message(STATUS "AMSS_EXTRA_RPATH=${AMSS_EXTRA_RPATH}")
message(STATUS "==========================================================")
