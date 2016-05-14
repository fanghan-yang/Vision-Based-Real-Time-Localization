cmake_minimum_required(VERSION 2.8)
project( Registration_gopro3 )

set(CPACK_GENERATOR "ZIP")
include(CPack)
INCLUDE ( /usr/share/cmake-2.8/Modules/FindPkgConfig.cmake )

find_package( OpenCV REQUIRED )
#pkg_check_modules( REQUIRED flann )
find_package( OpenMP REQUIRED)
find_package(CUDA)

#find_package( Boost REQUIRED COMPONENTS filesystem )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2") 
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -u MAIN__")
SET(HAVE_F2C 1 CACHE BOOL "Do we have f2c or F77/I77?" )
SET(LAPACKBLAS_DIR /usr/lib CACHE PATH "the name of the lapack library")
SET(LAPACK_LIB lapack CACHE STRING "the name of the lapack library" )
SET(BLAS_LIB blas CACHE STRING "the name of the blas library" )
IF(HAVE_F2C)
	SET(F2C_LIB f2c CACHE STRING " the name of the f2c library")
ELSE(HAVE_F2C)
	SET(F77_LIB libF77 CACHE STRING "the name of the F77 library" )
	SET(I77_LIB libI77 CACHE STRING "the name of the I77 library" )
ENDIF(HAVE_F2C)

set(cuda_sources
  # dynamic.cu
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  cudaSiftD.h  
  cudaSift.h  
  cudautils.h
)  
set(sources
  AbsoluteRegistration.cpp
  DataUtility.cpp
  epnp.cpp
  MathUtility.cpp
  # EstimateCameraPose.cpp
  MultiviewGeometryUtility.cpp
)  

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(Registration_gopro3 ${cuda_sources} ${sources} OPTIONS -arch=sm_35)

set_target_properties(Registration_gopro3 PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries( Registration_gopro3
	/usr/local/cuda-7.0/lib/libcudadevrt.a
	${OpenCV_LIBS}
	${LAPACK_LIB}
	${BLAS_LIB}
	${F2C_LIB}
)
