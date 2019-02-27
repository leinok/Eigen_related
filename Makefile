CXX	= g++
CXXFLAGS += -std=c++11 -I /home/saiclei/Research/Softwares/eigen-git-mirror/ -march=native -Wall 

all: test_eigen.o softmax_layer.o fully_connected_layer.o activation_layer.o pooling_layer.o base_functions.o
	${CXX} ${CXXFLAGS} -o test_eigen test_eigen.o softmax_layer.o fully_connected_layer.o activation_layer.o base_functions.o

test_eigen.o: test_eigen.cpp softmax_layer.hpp
	${CXX} ${CXXFLAGS} -c test_eigen.cpp 

softmax_Layer.o: softmax_layer.cpp softmax_layer.hpp
	${CXX} ${CXXFLAGS} -c softmax_layer.cpp 

fully_connected_layer.o: fully_connected_layer.cpp fully_connected_layer.hpp
	${CXX} ${CXXFLAGS} -c fully_connected_layer.cpp

activation_layer.o: activation_layer.cpp activation_layer.hpp
	${CXX} ${CXXFLAGS} -c activation_layer.cpp

pooling_layer.o: pooling_layer.cpp pooling_layer.hpp config.hpp base_functions.hpp
	${CXX} ${CXXFLAGS} -c pooling_layer.cpp

base_functions.o: base_functions.cpp base_functions.hpp
	${CXX} ${CXXFLAGS} -c base_functions.cpp
.PHONY: clean

clean:
	rm -f *.o test_eigen 
