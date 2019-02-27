#ifndef BASE_FUNCTIONS_H_
#define BASE_FUNCTIONS_H_

#include "config.hpp"
#include "Eigen/Dense"


namespace saicdl {

class BaseFunctions {
public:
    template<class T>
    static void printShape(const T& tensor);
};
}
#endif  // end of BASE_FUNCTIONS_H_
