#ifndef TYPES_HEADER_FILE
#define TYPES_HEADER_FILE

#include <vector>
#include <random>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Dense>

using Float = float;
// This should have decent properties.
using RandomEngine = std::mt19937;

using Vector = Eigen::Matrix<Float, Eigen::Dynamic, 1>;

using Matrix2D       = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;
using SparseMatrix2D = Eigen::SparseMatrix<Float, Eigen::RowMajor>;

using Matrix3D       = std::vector<Matrix2D>;
using SparseMatrix3D = std::vector<SparseMatrix2D>;

// To decide whether to init something as a sample or as maximum likelihood.
enum class InitType {SAMPLE, ML};

#endif
