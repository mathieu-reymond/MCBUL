#ifndef IO_HEADER_FILE
#define IO_HEADER_FILE

#include <iosfwd>

#include <Types.hpp>

// Weights, mus, sigmas
std::tuple<Vector, std::vector<Vector>, std::vector<Vector>> parseModelParameters(std::istream & input);

#endif
