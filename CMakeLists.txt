cmake_minimum_required(VERSION 3.13)
project(CPP_CSHC)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(CPP_CSHC
		Cart.cpp
		Cart.hpp
		main.cpp
		Matrix.h
		Utils.cpp
		Utils.h Bagging.cpp Bagging.hpp)
