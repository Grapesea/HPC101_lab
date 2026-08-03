# Local cross-compilation toolchain for building/testing the RISC-V target
# on an x86_64 host.  Not used on the cluster node itself (there CMake
# detects riscv64 natively).  Run with:
#   cmake -B build-rv -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-toolchain.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-riscv64;-L;/usr/riscv64-linux-gnu")
