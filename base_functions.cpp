#include "base_functions.hpp"
#include <string>
#include <iostream>



namespace saicdl {

template <typename T>
void BaseFunctions::printShape(const T& tensor) {
    std::string t;
    for (int i = 0; i < tensor.NumDimensions; ++i) {
        t += std::to_string(tensor.dimension(i)) + " ";
    }
    std::cout << "tensor shape: " << '\n';
}

}

