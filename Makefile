INCLUDES = -I video-processing/ `pkg-config --cflags --libs opencv4` -I /usr/include/libsvm -pthread
CC = g++
COMPILE_FLAGS = -std=c++11 -Wall -g
SRCS = main.cpp video-processing/*.cpp
TARGET = runnable
all: 
	$(CC) ${COMPILE_FLAGS} ${SRCS} ${INCLUDES} -o ${TARGET}