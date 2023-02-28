cmake_minimum_required(VERSION 3.24)

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

add_executable(cs 
    "src/main.cpp"
    "src/imgui_impl_raylib.cpp"
)

add_executable(cs_test
    "src/main_test.cpp"
)

#enable_language(CUDA)
#find_package(CUDA 9.0 REQUIRED)

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
    target_include_directories(cs_test PUBLIC "${eigen3_SOURCE_DIR}")
endif()

# ------------------ cuNSearch------------------
#message(STATUS "Fetching cuNSearch...")
#FetchContent_Declare(
#	cun
#	GIT_REPOSITORY        https://github.com/InteractiveComputerGraphics/cuNSearch.git
#    GIT_TAG               "aba3da18cb4f45cd05d729465d1725891ffc33da"
#)
#option(BUILD_DEMO "Build example of how to use this library." OFF)
#set(CUNSEARCH_USE_DOUBLE_PRECISION FALSE)
#FetchContent_MakeAvailable(cun)
#set_property(TARGET cuNSearch PROPERTY CUDA_ARCHITECTURES native)
#add_compile_options(-DGPU_NEIGHBORHOOD_SEARCH)

# --------------- RAYLIB ---------------
message(STATUS "Fetching Raylib...")
FetchContent_Declare(
	RAYLIB
	GIT_REPOSITORY  https://github.com/raysan5/raylib.git
	GIT_TAG         bf2ad9df5fdcaa385b2a7f66fd85632eeebbadaa
)
option(BUILD_EXAMPLES "Build example of how to use this library." OFF)
option(BUILD_SHARED_LIBS "Build shared libs." OFF)
set(GRAPHICS "GRAPHICS_API_OPENGL_43" FORCE)
FetchContent_MakeAvailable(RAYLIB)

# ------------------ IMGUI ------------------
message(STATUS "Fetching imgui...")
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
)
FetchContent_GetProperties(imgui)

if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)

    set(IMGUI_FILES
        "${imgui_SOURCE_DIR}/imgui_demo.cpp"
        "${imgui_SOURCE_DIR}/imgui_draw.cpp"
        "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
        "${imgui_SOURCE_DIR}/imgui_tables.cpp"
        "${imgui_SOURCE_DIR}/imgui.cpp"        
    )

    add_library(imgui STATIC "${IMGUI_FILES}")

    target_include_directories(cs PUBLIC "${imgui_SOURCE_DIR}")
endif()

# ------------------ LIBMORTON ------------------
message(STATUS "Fetching libmorton...")
FetchContent_Declare(
    libmorton 
    GIT_REPOSITORY      https://github.com/Forceflow/libmorton.git
    GIT_TAG             v0.2.10
    GIT_SHALLOW         TRUE
)
FetchContent_MakeAvailable(libmorton)

# ------------------ ROBIN-MAP ------------------
message(STATUS "Fetching robin map...")
FetchContent_Declare(robin_map
  GIT_REPOSITORY        https://github.com/Tessil/robin-map.git
  GIT_TAG               v1.0.1
  GIT_SHALLOW           TRUE
)
FetchContent_MakeAvailable(robin_map)

# ------------------ KOMPUTE ------------------
message(STATUS "Fetching kompute...")
FetchContent_Declare(kompute
  GIT_REPOSITORY        https://github.com/KomputeProject/kompute.git
  GIT_TAG               v0.8.1
  GIT_SHALLOW           TRUE
)
option(KOMPUTE_OPT_REPO_SUBMODULE_BUILD "Use the submodule repos instead of external package manager" ON)
FetchContent_MakeAvailable(kompute)

# ------------------ CATCH2 ------------------
message(STATUS "Fetching Catch2...")
FetchContent_Declare(
		Catch2 
		GIT_REPOSITORY      https://github.com/catchorg/Catch2.git
		GIT_TAG             v2.13.6
		GIT_SHALLOW	        TRUE
)
FetchContent_MakeAvailable(Catch2)

target_link_libraries(cs 
    #cuNSearch
    raylib 
    imgui
    libmorton
    tsl::robin_map
    kompute
)

target_link_libraries(cs_test
    libmorton
    tsl::robin_map
    kompute
    Catch2
)