
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "Fortran"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_Fortran
  "/home/h3240104505/HPC101_lab/lab4/src/bssn_rhs.f90" "/home/h3240104505/HPC101_lab/lab4/build_gpu/CMakeFiles/ABEGPU.dir/src/bssn_rhs.f90.o"
  "/home/h3240104505/HPC101_lab/lab4/src/diff_new.f90" "/home/h3240104505/HPC101_lab/lab4/build_gpu/CMakeFiles/ABEGPU.dir/src/diff_new.f90.o"
  "/home/h3240104505/HPC101_lab/lab4/src/enforce_algebra.f90" "/home/h3240104505/HPC101_lab/lab4/build_gpu/CMakeFiles/ABEGPU.dir/src/enforce_algebra.f90.o"
  "/home/h3240104505/HPC101_lab/lab4/src/fadmquantites_bssn.f90" "/home/h3240104505/HPC101_lab/lab4/build_gpu/CMakeFiles/ABEGPU.dir/src/fadmquantites_bssn.f90.o"
  "/home/h3240104505/HPC101_lab/lab4/src/fmisc.f90" "/home/h3240104505/HPC101_lab/lab4/build_gpu/CMakeFiles/ABEGPU.dir/src/fmisc.f90.o"
  "/home/h3240104505/HPC101_lab/lab4/src/getnp4.f90" "/home/h3240104505/HPC101_lab/lab4/build_gpu/CMakeFiles/ABEGPU.dir/src/getnp4.f90.o"
  "/home/h3240104505/HPC101_lab/lab4/src/initial_puncture.f90" "/home/h3240104505/HPC101_lab/lab4/build_gpu/CMakeFiles/ABEGPU.dir/src/initial_puncture.f90.o"
  "/home/h3240104505/HPC101_lab/lab4/src/kodiss.f90" "/home/h3240104505/HPC101_lab/lab4/build_gpu/CMakeFiles/ABEGPU.dir/src/kodiss.f90.o"
  "/home/h3240104505/HPC101_lab/lab4/src/lopsidediff.f90" "/home/h3240104505/HPC101_lab/lab4/build_gpu/CMakeFiles/ABEGPU.dir/src/lopsidediff.f90.o"
  "/home/h3240104505/HPC101_lab/lab4/src/prolongrestrict_cell.f90" "/home/h3240104505/HPC101_lab/lab4/build_gpu/CMakeFiles/ABEGPU.dir/src/prolongrestrict_cell.f90.o"
  "/home/h3240104505/HPC101_lab/lab4/src/rungekutta4_rout.f90" "/home/h3240104505/HPC101_lab/lab4/build_gpu/CMakeFiles/ABEGPU.dir/src/rungekutta4_rout.f90.o"
  "/home/h3240104505/HPC101_lab/lab4/src/sommerfeld_rout.f90" "/home/h3240104505/HPC101_lab/lab4/build_gpu/CMakeFiles/ABEGPU.dir/src/sommerfeld_rout.f90.o"
  )
set(CMAKE_Fortran_COMPILER_ID "GNU")
set(CMAKE_Fortran_SUBMODULE_SEP "@")
set(CMAKE_Fortran_SUBMODULE_EXT ".smod")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_Fortran
  "MPI_CUDA_AWARE=0"
  "USE_GPU"
  "fortran3"
  "newc"
  )

# The include file search paths:
set(CMAKE_Fortran_TARGET_INCLUDE_PATH
  "/home/h3240104505/HPC101_lab/lab4/src"
  "/usr/include"
  "/usr/lib/x86_64-linux-gnu/openmpi/include"
  "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi"
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  "/home/h3240104505/HPC101_lab/lab4/src/MPatch_gpu.cu" "CMakeFiles/ABEGPU.dir/src/MPatch_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/MPatch_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/bssn_rhs_gpu.cu" "CMakeFiles/ABEGPU.dir/src/bssn_rhs_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/bssn_rhs_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/diff_new_gpu.cu" "CMakeFiles/ABEGPU.dir/src/diff_new_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/diff_new_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/enforce_algebra_gpu.cu" "CMakeFiles/ABEGPU.dir/src/enforce_algebra_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/enforce_algebra_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/fadmquantites_bssn_gpu.cu" "CMakeFiles/ABEGPU.dir/src/fadmquantites_bssn_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/fadmquantites_bssn_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/fmisc_gpu.cu" "CMakeFiles/ABEGPU.dir/src/fmisc_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/fmisc_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/getnp4_gpu.cu" "CMakeFiles/ABEGPU.dir/src/getnp4_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/getnp4_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/gpu_manager.cu" "CMakeFiles/ABEGPU.dir/src/gpu_manager.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/gpu_manager.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/kodiss_gpu.cu" "CMakeFiles/ABEGPU.dir/src/kodiss_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/kodiss_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/lopsidediff_gpu.cu" "CMakeFiles/ABEGPU.dir/src/lopsidediff_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/lopsidediff_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/misc_gpu.cu" "CMakeFiles/ABEGPU.dir/src/misc_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/misc_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/prolongrestrict_cell_gpu.cu" "CMakeFiles/ABEGPU.dir/src/prolongrestrict_cell_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/prolongrestrict_cell_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/rungekutta4_rout_gpu.cu" "CMakeFiles/ABEGPU.dir/src/rungekutta4_rout_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/rungekutta4_rout_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/sommerfeld_rout_gpu.cu" "CMakeFiles/ABEGPU.dir/src/sommerfeld_rout_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/sommerfeld_rout_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/surface_integral_gpu.cu" "CMakeFiles/ABEGPU.dir/src/surface_integral_gpu.cu.o" "gcc" "CMakeFiles/ABEGPU.dir/src/surface_integral_gpu.cu.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/ABE.C" "CMakeFiles/ABEGPU.dir/src/ABE.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/ABE.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/Ansorg.C" "CMakeFiles/ABEGPU.dir/src/Ansorg.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/Ansorg.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/Block.C" "CMakeFiles/ABEGPU.dir/src/Block.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/Block.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/MPatch.C" "CMakeFiles/ABEGPU.dir/src/MPatch.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/MPatch.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/Parallel.C" "CMakeFiles/ABEGPU.dir/src/Parallel.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/Parallel.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/Parallel_GPU.cpp" "CMakeFiles/ABEGPU.dir/src/Parallel_GPU.cpp.o" "gcc" "CMakeFiles/ABEGPU.dir/src/Parallel_GPU.cpp.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/bssn_gpu_class.C" "CMakeFiles/ABEGPU.dir/src/bssn_gpu_class.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/bssn_gpu_class.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/bssn_step_gpu.C" "CMakeFiles/ABEGPU.dir/src/bssn_step_gpu.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/bssn_step_gpu.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/cgh.C" "CMakeFiles/ABEGPU.dir/src/cgh.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/cgh.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/checkpoint.C" "CMakeFiles/ABEGPU.dir/src/checkpoint.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/checkpoint.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/helper.cpp" "CMakeFiles/ABEGPU.dir/src/helper.cpp.o" "gcc" "CMakeFiles/ABEGPU.dir/src/helper.cpp.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/misc.C" "CMakeFiles/ABEGPU.dir/src/misc.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/misc.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/monitor.C" "CMakeFiles/ABEGPU.dir/src/monitor.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/monitor.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/perf.C" "CMakeFiles/ABEGPU.dir/src/perf.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/perf.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/surface_integral.C" "CMakeFiles/ABEGPU.dir/src/surface_integral.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/surface_integral.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/transpbh.C" "CMakeFiles/ABEGPU.dir/src/transpbh.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/transpbh.C.o.d"
  "/home/h3240104505/HPC101_lab/lab4/src/var.C" "CMakeFiles/ABEGPU.dir/src/var.C.o" "gcc" "CMakeFiles/ABEGPU.dir/src/var.C.o.d"
  "" "ABEGPU" "gcc" "CMakeFiles/ABEGPU.dir/link.d"
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
