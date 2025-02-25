#!/bin/bash -e

kokkos_build_type="${1}"

if [ "$kokkos_build_type" = "none" ]; then
    "Will not install Kokkos"
    return 0;
fi

# If all arguments are valid, you can use them in your script as needed
echo "Kokkos Build Type: $kokkos_build_type"

if [ ! -d "${KOKKOS_SOURCE_DIR}/core" ]
then
  echo "Missing Kokkos submodules, downloading...."
  git submodule update --init --recursive
fi

rm -rf ${KOKKOS_INSTALL_DIR}
mkdir -p ${KOKKOS_BUILD_DIR} 

# Kokkos flags for Cuda
CUDA_ADDITIONS=(
-D Kokkos_ENABLE_CUDA=ON
-D Kokkos_ENABLE_CUDA_CONSTEXPR=ON
-D Kokkos_ENABLE_CUDA_LAMBDA=ON
-D Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON
)

# Kokkos flags for Hip
HIP_ADDITIONS=(
-D Kokkos_ENABLE_HIP=ON
-D CMAKE_CXX_COMPILER=hipcc
-D Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=ON
)

# Kokkos flags for OpenMP
OPENMP_ADDITIONS=(
-D Kokkos_ENABLE_OPENMP=ON
)

# Kokkos flags for PThreads
PTHREADS_ADDITIONS=(
-D Kokkos_ENABLE_THREADS=ON
)

# Configure kokkos using CMake
cmake_options=(
    -D CMAKE_BUILD_TYPE=Debug
    -D CMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL_DIR}"
    -D CMAKE_CXX_STANDARD=17
    -D Kokkos_ENABLE_SERIAL=ON
    -D Kokkos_ARCH_NATIVE=ON
    -D Kokkos_ENABLE_TESTS=OFF
    -D BUILD_TESTING=OFF
)

if [ "$kokkos_build_type" = "openmp" ]; then
    cmake_options+=(
        ${OPENMP_ADDITIONS[@]}
    )
elif [ "$kokkos_build_type" = "openmp_mpi" ]; then
    cmake_options+=(
        ${OPENMP_ADDITIONS[@]}
    )
elif [ "$kokkos_build_type" = "pthreads" ]; then
    cmake_options+=(
        ${PTHREADS_ADDITIONS[@]}
    )
elif [ "$kokkos_build_type" = "cuda" ]; then
    cmake_options+=(
        ${CUDA_ADDITIONS[@]}
    )
elif [ "$kokkos_build_type" = "cuda_mpi" ]; then
    cmake_options+=(
        ${CUDA_ADDITIONS[@]}
    )
elif [ "$kokkos_build_type" = "hip" ]; then
    cmake_options+=(
        ${HIP_ADDITIONS[@]}
    )
elif [ "$kokkos_build_type" = "hip_mpi" ]; then
    cmake_options+=(
        ${HIP_ADDITIONS[@]}
    )
fi

# Print CMake options for reference
echo "CMake Options: ${cmake_options[@]}"

# Configure kokkos
cmake "${cmake_options[@]}" -B "${KOKKOS_BUILD_DIR}" -S "${KOKKOS_SOURCE_DIR}"

# Build kokkos
echo "Building kokkos..."
make -C ${KOKKOS_BUILD_DIR} -j${MATAR_BUILD_CORES}

# Install kokkos
echo "Installing kokkos..."
make -C ${KOKKOS_BUILD_DIR} install

echo "kokkos installation complete."
