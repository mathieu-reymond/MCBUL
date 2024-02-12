#ifndef HDF5CPP_CONTRIB_EIGEN_H
#define HDF5CPP_CONTRIB_EIGEN_H

#include <Eigen/Core>
#include <h5cpp/hdf5.hpp>
#include <Types.hpp>

namespace hdf5 {
    namespace datatype {
        template <typename Derived>
        class TypeTrait<Eigen::MatrixBase<Derived>> {
            public:
                using TypeClass = typename TypeTrait<::Float>::TypeClass;
                static TypeClass create(const Eigen::MatrixBase<Derived> &) { 
                    return TypeTrait<::Float>::create();
                }
                const static TypeClass & get(const Eigen::MatrixBase<Derived> &) {
                    const static TypeClass & cref_ = TypeTrait<::Float>::create(); 
                    return cref_;
                }
        };
    }

    namespace dataspace {
        template <typename Derived>
        class TypeTrait<Eigen::MatrixBase<Derived>> {
            public:
                using DataspaceType = Simple;
                static DataspaceType create(const Eigen::MatrixBase<Derived>& matrix) {
                    auto rows = static_cast<hsize_t>(matrix.rows());
                    auto cols = static_cast<hsize_t>(matrix.cols());
                    return Simple({rows, cols});
                }
                static const Dataspace & get(const Eigen::MatrixBase<Derived>& matrix, DataspacePool & pool) {
                    return pool.getSimple(matrix.size());
                }
                static void* ptr(Eigen::MatrixBase<Derived>& value) {
                    Eigen::Ref<Matrix2D> ref(value);
                    return static_cast<void*>(ref.data());
                }
                static const void* cptr(const Eigen::MatrixBase<Derived>& value) {
                    const Eigen::Ref<const Matrix2D> ref(value);
                    return static_cast<const void*>(ref.data());
                }
        };
    }
}

#endif
