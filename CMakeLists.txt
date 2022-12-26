cmake_minimum_required(VERSION 3.18)

project(ChlötzliSpüeler)

if (MSVC)
	set(CMAKE_GENERATOR_PLATFORM x64)
endif()	
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") 
	#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:library")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -mavx -mavx2")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -O0 -g3 -ggdb -fsanitize=address -fsanitize=undefined -D_GLIBCXX_DEBUG" )
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Os -march=native -DNDEBUG -flto -fomit-frame-pointer")
endif()

include(FetchContent)

add_executable(cs "src/main.cpp")

# ------------------ Eigen ------------------
message(STATUS "Fetching Eigen3...")
FetchContent_Declare(
	eigen3
	GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
	GIT_SHALLOW TRUE
	GIT_TAG        3.4
)
FetchContent_GetProperties(eigen3)

if(NOT eigen3_POPULATED)
	FetchContent_Populate(eigen3)
	target_include_directories(cs PUBLIC "${eigen3_SOURCE_DIR}")
endif()

# ------------------ cuNSearch------------------
#message(STATUS "Fetching cuNSearch...")
#FetchContent_Declare(
#	cun
#	GIT_REPOSITORY        https://github.com/InteractiveComputerGraphics/cuNSearch.git
#	GIT_SHALLOW           TRUE
#)
#FetchContent_MakeAvailable(cun)

# ------------------ SPH ------------------
message(STATUS "Fetching SPlisHSPlasH...")
FetchContent_Declare(
	sph
	GIT_REPOSITORY        https://github.com/InteractiveComputerGraphics/SPlisHSPlasH.git
	GIT_TAG               2.12.0
	GIT_SHALLOW           TRUE
)

if(NOT sph_POPULATED)
	set(USE_DOUBLE_PRECISION OFF CACHE BOOL "Use double precision" FORCE)
	set(USE_AVX ON CACHE BOOL "Use AVX" FORCE)
	set(USE_GPU_NEIGHBORHOOD_SEARCH ON CACHE BOOL "Use GPU neighborhood search" FORCE)
	set(CUNSEARCH_USE_DOUBLE_PRECISION OFF CACHE BOOL "" FORCE)
	set(SPH_LIBS_ONLY ON CACHE BOOL "" FORCE)

	FetchContent_Populate(sph)
	set(CMAKE_MODULE_PATH "${sph_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})
	add_subdirectory("${sph_SOURCE_DIR}" "${sph_BINARY_DIR}")
endif()

target_link_libraries(cs SPlisHSPlasH Utilities)