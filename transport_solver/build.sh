g++ -O3 -Wall -shared -std=c++20 -fPIC -I "/usr/local/include/eigen3" $(python3 -m pybind11 --includes) transport_solver.cpp -o transport_solver$(python3-config --extension-suffix)
