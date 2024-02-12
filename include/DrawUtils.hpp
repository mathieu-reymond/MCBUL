#ifndef DRAW_UTILS_HEADER
#define DRAW_UTILS_HEADER

#include <Types.hpp>

#include <Eigen/Eigenvalues>
#include <boost/math/constants/constants.hpp>
#include <iostream>

#include <Matplotlib.hpp>
// #include <matplot/matplot.h>

template <typename T>
std::vector<double> toVec(const T & t) {
    std::vector<double> v(t.size());
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = t[i];
    return v;
}

std::vector<double> toVecD(const double d) {
    std::vector<double> v{d};
    return v;
}

inline void plotGP(const Matrix2D & xAsk, const Vector & muAsk, const Matrix2D & cov, const Matrix2D & xData, const Vector & muData) {
    namespace plt = matplotlibcpp;
    if (xAsk.rows() != muAsk.size()) {
        std::cout << xAsk << "\n\n" << muAsk << std::endl;
        throw std::runtime_error("Ask points and mu have different sizes");
    }
    if (xData.rows() != muData.size())
        throw std::runtime_error("Train points and mu have different sizes");
    switch (xAsk.cols()) {
        case 1: {
            plt::plot(toVec(xAsk.col(0)), toVec(muAsk));
            plt::plot(toVec(xData.col(0)), toVec(muData), {{"marker", "x"}, {"linestyle", " "}});

            Vector uncertainty = 1.96 * cov.diagonal().array().sqrt();
            plt::fill_between(toVec(xAsk.col(0)), toVec(muAsk - uncertainty), toVec(muAsk + uncertainty), {{"alpha", "0.1"}, {"color", "blue"}});
            std::cout << uncertainty << '\n';

            plt::show();
            break;
        }
        case 2: {
            std::vector<std::vector<double>> x, y, z;
            const int sqSize = std::sqrt(xAsk.rows());
            int counter = 0;
            for (auto i = 0; i < sqSize; ++i) {
                std::vector<double> x_row, y_row, z_row;
                for (auto j = 0; j < sqSize; ++j) {
                    x_row.push_back(xAsk(counter, 0));
                    y_row.push_back(xAsk(counter, 1));
                    z_row.push_back(muAsk(counter));

                    ++counter;
                }
                x.push_back(x_row);
                y.push_back(y_row);
                z.push_back(z_row);
            }
            auto axis = plt::plot_surface(x, y, z);

            auto xxx = toVec(xData.col(0));
            auto yyy = toVec(xData.col(1));
            auto zzz = toVec(muData);
            plt::plot3(xxx, yyy, zzz, {{"marker", "x"}, {"linestyle", " "}}, axis);

            plt::show();
            break;
        }
        default:
            throw std::runtime_error("Can't plot more than 2d");
    }
}

inline Matrix2D plotEllipse(const Vector & mean, const Matrix2D & cov) {
    constexpr double chiSquare = 2.4477;

    Eigen::EigenSolver<Matrix2D> solver(cov);

    int maxId;
    double maxEigenVal = solver.eigenvalues().real().maxCoeff(&maxId);
    Vector maxEigenVec = solver.eigenvectors().col(maxId).real();
    int minId = !maxId;
    double minEigenVal = solver.eigenvalues()(minId).real();

    double angle = std::atan2(maxEigenVec[1], maxEigenVec[0]);
    if (angle < 0) angle += 2 * boost::math::constants::pi<double>();

    double a = chiSquare * std::sqrt(maxEigenVal);
    double b = chiSquare * std::sqrt(minEigenVal);

    constexpr int steps = 100;
    Matrix2D coords(steps, 2); double theta = 0.0;
    for (auto i = 0; i < steps; ++i) {
        coords(i, 0) = a * std::cos(theta);
        coords(i, 1) = b * std::sin(theta);

        theta += 2.0 * boost::math::constants::pi<double>() / steps;
    }
    Matrix2D R(2, 2);
    R << std::cos(angle), std::sin(angle), -std::sin(angle), std::cos(angle);

    coords *= R;
    coords.rowwise() += mean.transpose();

    return coords;
}

// void plotGP(const Matrix2D & xAsk, const Vector & muAsk, const Matrix2D & cov, const Matrix2D & xData = {}, const Vector & muData = {}) {
//     if (xAsk.rows() != muAsk.size()) {
//         std::cout << xAsk << "\n\n" << muAsk << std::endl;
//         throw std::runtime_error("Ask points and mu have different sizes");
//     }
//     if (xData.rows() != muData.size())
//         throw std::runtime_error("Train points and mu have different sizes");
//
//     auto plot = matplot::plot(
//             toVec(xAsk.col(0)), toVec(muAsk),
//             toVec(xData.col(0)), toVec(muData)
//     );
//     plot[1]->marker("x").line_spec().line_style(matplot::line_spec::line_style::none);
//     matplot::save("test.png");
// }
//
// Matrix2D plotEllipse(const Vector & mean, const Matrix2D & cov) {
//     return Matrix2D();
// }

#endif
