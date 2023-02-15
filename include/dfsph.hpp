#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include <tsl/robin_map.h>
#include <libmorton/morton.h>

#include <Eigen/Eigen>

namespace SPH {

    using namespace Eigen;

    template<class PROXY>
    class SparseGrid {
        using T = typename PROXY::T;
        constexpr static size_t DIM = PROXY::DIM;

        EIGEN_STRONG_INLINE uint_fast64_t morton_index(int _x, int _y, int _z) {
            const uint_fast64_t out = libmorton::morton3D_64_encode(uint_fast64_t(TMathUtil<T>::Abs(_x)), uint_fast64_t(TMathUtil<T>::Abs(_y)), uint_fast64_t(TMathUtil<T>::Abs(_z)));
            return (out & 0x1FFFFFFFFFFFFFFF) | uint_fast64_t(std::signbit(_x)) << 63 | uint_fast64_t(std::signbit(_y)) << 62 | uint_fast64_t(std::signbit(_z)) << 61;
        }

        EIGEN_STRONG_INLINE uint_fast64_t morton_index(int _x, int _y) {
            const uint_fast64_t out = libmorton::morton2D_64_encode(uint_fast64_t(TMathUtil<T>::Abs(_x)), uint_fast64_t(TMathUtil<T>::Abs(_y)));
            return (out & 0x1FFFFFFFFFFFFFFF) | uint_fast64_t(std::signbit(_x)) << 63 | uint_fast64_t(std::signbit(_y)) << 62;
        }

    public:

        SparseGrid(const typename PROXY& _proxy) {

            static_assert(DIM == 2 || DIM == 3, "only Dim {2, 3} supported");

            tsl::robin_map<uint_fast64_t, std::vector<int>> grid;

            //bucket size
            const T bs = T(1) / 2 * _proxy.d;
            const T h = T(1) / (_proxy.h * T(0.5));

            //insert
            for (int i = 0; i < _proxy.size(); ++i) {
                const uint_fast64_t key = morton_index((_proxy.x[i] * bs));
                grid[key].add(i);
            }

            //reset
            _proxy.n.Reset(_proxy.size());
            for (int i = 0; i < _proxy.size(); ++i)
                _proxy.n[i].Reset(8);

            //nn
            const T r2 = _proxy.h2;
            for (int i = 0; i < _proxy.size(); ++i) {

                const Vector<T, DIM> min_x = (_proxy.x[i] - Vector<T, DIM>(h)) * h;
                const Vector<T, DIM> max_x = (_proxy.x[i] + Vector<T, DIM>(h)) * h;

                const Vector<T, DIM>& pos = _proxy.x[i];

                if constexpr (DIM == 2) {
                    for (int y = int(min_x(1)); y <= int(max_x(1)); ++y) {
                        for (int x = int(min_x(0)); x <= int(max_x(0)); ++x) {
                            const uint_fast64_t idx = morton_index(x, y);
                            const std::vector<int>& e = grid[idx];
                            for (int i : e) {
                                const T dist = (_proxy.x[i] - pos).squaredNorm();
                                if (dist <= _proxy.h2)
                                    _proxy.n[i].Add(i);
                            }
                        }
                    }
                } else if constexpr (DIM == 3) {
                    for (int z = int(min_x(2)); z <= int(max_x(2)); ++z) {
                        for (int y = int(min_x(1)); y <= int(max_x(1)); ++y) {
                            for (int x = int(min_x(0)); x <= int(max_x(0)); ++x) {
                                const uint_fast64_t idx = morton_index(x, y, z);
                                const std::vector<int>& e = grid[idx];
                                for (int i : e) {
                                    const T dist = (_proxy.x[i] - pos).squaredNorm();
                                    if (dist <= _proxy.h2)
                                        _proxy.n[i].Add(i);
                                }
                            }
                        }
                    }
                }

            }

        }//SparseGrid

    };//SparseGrid

    namespace Kernel {

        class CubicSpline {

            template<class PROXY>
            EIGEN_STRONG_INLINE static typename PROXY::T sigma(const typename PROXY& _proxy) noexcept {
                using T = typename PROXY::T;
                constexpr static size_t DIM = PROXY::DIM;
                if constexpr (DIM == 3) {
                    constexpr static T ip = T(8) / T(M_PI);
                    return ip * _proxy.h3i;
                } else if constexpr (DIM == 2) {
                    constexpr static T ip = T(40) / (T(7) * T(M_PI));
                    return ip * _proxy.h2i;
                } else {
                    constexpr static T ip = T(4) / T(3);
                    return ip * _proxy.hi;
                }
            }//sigma

        public:

            template<class PROXY>
            EIGEN_STRONG_INLINE static typename PROXY::T W_ij(const typename PROXY& _proxy, int _i, int _j) noexcept {
                using T = typename PROXY::T;
                constexpr static size_t DIM = PROXY::DIM;
                const T s = PROXY::KERNEL::template sigma<PROXY>(_proxy.h);
                const T q = _proxy.h1 * (_proxy.x[_i] - _proxy.x[_j]).norm();
                if (T(0) <= q && q <= T(0.5)) return (s * (T(6) * (TMathUtil<T>::Pow(q, 3) - TMathUtil<T>::Pow(q, 2) + T(1))));
                else if (T(0.5) < q && q <= T(1)) return (s * (T(2) * TMathUtil<T>::Pow((T(1) - q), 3)));
                else return T(0);
            }//W_ij

            template<class PROXY>
            EIGEN_STRONG_INLINE static Vector<typename PROXY::T, PROXY::DIM> dW_ij(const typename PROXY& _proxy, int _i, int _j) noexcept {
                using T = typename PROXY::T;
                constexpr static size_t DIM = PROXY::DIM;
                const T s = CubicSpline::template sigma<PROXY>(_proxy.h) * _proxy.h4i;
                Vector<T, DIM> r = _proxy.x[_i] - _proxy.x[_j];
                const T q = _proxy.h1 * r.norm();
                r.normalize();
                if (T(0) <= q && q <= T(0.5)) return r * (s * - T(6) * q * (T(2) - T(3) * q));
                else if (T(0.5) < q && q <= T(1)) return r * (s * - T(6) * TMathUtil<T>::Pow(T(1) - q, 2));
                else return r * T(0);
            }//dW_ij

        };//CubicSpline

    }//Kernel

    namespace Operator {

        struct MassDensitiyEstimator {

            template<class PROXY>
            EIGEN_STRONG_INLINE static typename PROXY::T ro_i(const typename PROXY& _proxy, int _i) noexcept {
                using T = typename PROXY::T;
                constexpr static size_t DIM = PROXY::DIM;
                T out = T(0);
                for (int k = 0; k < _proxy.n[_i].Num(); ++k) {
                    const int j = _proxy.n[_i][k];
                    out += _proxy.m[j] * _proxy.W[_i][j]; //PROXY::KERNEL::template W_ij<PROXY>(_proxy, _i, j);
                }//ro_i
                return out;
            }

        };//MassDensitiyEstimator

        struct Differential {

            template<class PROXY, class FUNC>
            EIGEN_STRONG_INLINE static Vector<typename PROXY::T, PROXY::DIM> difference(const typename PROXY& _proxy, int _i, FUNC&& _A) noexcept {
                using T = typename PROXY::T;
                constexpr static size_t DIM = PROXY::DIM;
                Vector<T, DIM> out = Vector<T, DIM>::Zero();
                const auto Ai = _A(_i);
                for (int k = 0; k < _proxy.n[_i].Num(); ++k) {
                    const int j = _proxy.n[_i][k];
                    const T dW = _proxy.dW[_i][j]; //PROXY::KERNEL::template dW_ij<PROXY>(_x[_i] - _x[j], _h);
                    const T ro_j = _proxy.ro[j]; //PROXY::RO::template ro_i<PROXY>(j, _x, _m);
                    out += (_proxy.m[j] / ro_j) * (_A(j) - Ai) * dW;
                }
                return out;
            }//difference

            template<class PROXY, class FUNC>
            EIGEN_STRONG_INLINE static Vector<typename PROXY::T, PROXY::DIM> symmetric(const typename PROXY& _proxy, int _i, FUNC&& _A) noexcept {
                using T = typename PROXY::T;
                constexpr static size_t DIM = PROXY::DIM;
                const T ro_i = _proxy.ro[_i]; //PROXY::RO::template ro_i<PROXY>(_proxy, _i);
                const T ro_i_2 = ro_i * ro_i;
                const auto Ai = _A(_i);
                Vector<T, DIM> out = Vector<T, DIM>::Zero();
                for (int k = 0; k < _proxy.n[_i].Num(); ++k) {
                    const int j = _proxy.n[_i][k];
                    const T dW = _proxy.dW[_i][j]; //PROXY::KERNEL::template dW_ij<PROXY>(_x[i] - _x[j], _h);
                    const T ro_j = _proxy.ro[j]; //PROXY::RO::template ro_i<PROXY>(j, _x, _m);
                    out += _proxy.m[j] * (Ai / ro_i_2) + (_A(j) / (ro_j * ro_j)) * dW;
                }
                return out;
            }//symmetric

            template<class PROXY, class FUNC>
            EIGEN_STRONG_INLINE static Vector<typename PROXY::T, PROXY::DIM> spatial(const typename PROXY& _proxy, int _i, FUNC&& _A) noexcept {
                using T = typename PROXY::T;
                constexpr static size_t DIM = PROXY::DIM;
                Vector<T, DIM> out = Vector<T, DIM>::Zero();
                for (int k = 0; k < _proxy.n[_i].Num(); ++k) {
                    const int j = _proxy.n[_i][k];
                    const T ro_j = _proxy.ro[j]; //PROXY::RO::template ro_i<PROXY>(_proxy, j);
                    out += _proxy.m[j] / ro_j * _A(j) * _proxy.dW[_i][j];
                }
                return out;
            }//spatial

        };//Differential

    }//Operator

    struct DFSPH {

        template<class PROXY>
        EIGEN_STRONG_INLINE static typename PROXY::T CFL(const typename PROXY& _proxy) noexcept {
            using T = typename PROXY::T;
            constexpr static size_t DIM = PROXY::DIM;
            T max = -(1. / 0.);
            int ii = 0;
            for (int i = 0; i < _proxy.size(); ++i) {
                const T m = TMathUtil<T>::Max(max, _proxy.v[i].squaredNorm());
                if (m > max) {
                    max = m;
                    ii = i;
                }
            }
            return T(0.4) * _proxy.h / TMathUtil<T>::Sqrt(max);
        }//CFL

        template<class PROXY>
        EIGEN_STRONG_INLINE static typename PROXY::T alpha_i(const typename PROXY& _proxy, int _i) noexcept {
            using T = typename PROXY::T;
            constexpr static size_t DIM = PROXY::DIM;
            T a1 = T(0);
            T a2 = T(0);
            for (int k = 0; k < _proxy.n[_i].Num(); ++k) {
                const int j = _proxy.n[_i][k];
                const T w = _proxy.W[_i][j]; //KERNEL::W_ij(_x[i] - _x[j], _h);
                a1 += _proxy.m[j] * w;
                a2 += _proxy.m[j] * TMathUtil<T>::Pow(TMathUtil<T>::Abs(w), 2);
            }
            return TMathUtil<T>::Pow(TMathUtil<T>::Abs(a1), 2) + a2;
        }//alpha_i

        template<class PROXY>
        EIGEN_STRONG_INLINE static void correct_divergence_error(typename PROXY& _proxy) noexcept {
            using T = typename PROXY::T;
            constexpr static size_t DIM = PROXY::DIM;

            std::memcpy(_proxy.v_s.data(), _proxy.v.data(), _proxy.size() * sizeof(T));

            T avg = T(0);
            size_t iter = 0;

            std::vector<T> Dp_Dt(_proxy.size());

            while (avg > _proxy.div_error || iter < 1) {

                //compute Dro_Dt
                avg = T(0);
                for (size_t i = 0; i < _proxy.size(); ++i) {
                    Dp_Dt[i] = - _proxy.ro[i] * PROXY::DIFFERENTIAL::template spatial<PROXY>(_proxy, i, [&](int ii) { return _proxy.v_s[ii]; });
                    avg += Dp_Dt[i];
                }
                avg /= T(_proxy.size());

                //adapt velocities
                for (int i = 0; i < _proxy.size(); ++i) {

                    const T k_v_i_ro = (_proxy.dTi * Dp_Dt[i] * _proxy.alpha[i]) / _proxy.ro[i];

                    Vector<T, DIM> v = Vector<T, DIM>::Zeros();
                    for (int k = 0; k < _proxy.n[i].Num(); ++k) {
                        const int j = _proxy.n[i][k];

                        const T k_v_j = _proxy.dTi * Dp_Dt[j] * _proxy.alpha[j];
                        v += _proxy.m[j] * (k_v_i_ro  + k_v_j / _proxy.ro[j]) * _proxy.dW[i][j]; //PROXY::KERNEL::template dW_ij<PROXY>(_x[i] - _x[j], _h);
                    }
                    _proxy.v_s[i] -= _proxy.dT * v;
                }

            }

        }//correct_divergence_error

        template<class PROXY>
        EIGEN_STRONG_INLINE static void correct_density_error(typename PROXY& _proxy) noexcept {
            using T = typename PROXY::T;
            constexpr static size_t DIM = PROXY::DIM;

            int iter = 0;
            T error = T(0);
            while (error > _proxy.dens_error || iter < 2) {

                //predict density
                error = T(0);
                for (int i = 0; i < _proxy.size(); ++i) {
                    T rs = T(0);
                    for (int k = 0; k < _proxy.n[i].Num(); ++k) {
                        const int j = _proxy.n[i][k];
                        rs += _proxy.m[j] * (_proxy.v_s[i] - _proxy.v_s[j]) * _proxy.dW[i][j];
                    }
                    _proxy.ro_s[i] = _proxy.ro[i] + _proxy.dt * rs;
                    error += _proxy.ro_s[i] - _proxy.ro_0;
                }

                //adapt velocities
                for (int i = 0; i < _proxy.size(); ++i) {
                    const T K_i = ((_proxy.ro_s[i] - _proxy.ro[i]) * _proxy.dt2i * _proxy.alpha[i]) / _proxy.ro[i];
                    Vector<T, DIM> er = Vector<T, DIM>::Zeros();
                    for (int k = 0; k < _proxy.n[i].Num(); ++k) {
                        const int j = _proxy.n[i][k];
                        const T K_j = ((_proxy.ro_s[j] - _proxy.ro[j]) * _proxy.dt2i * _proxy.alpha[j]) / _proxy.ro[j];
                        er += _proxy.m[j] * (K_i + K_j) * _proxy.dW[i][j];
                    }
                    _proxy.v_s[i] -= (_proxy.dt * er);
                }

                iter++;

            }

        }//correct_density_error

        template<class PROXY>
        EIGEN_STRONG_INLINE static void correct_strain_rate_error(typename PROXY& _proxy) noexcept {

            for (int i = 0; i < _proxy.size(); ++i) {

            }

            int iter = 0;
            while(true && iter < 1){

                //compute strain rate
                for (int i = 0; i < _proxy.size(); ++i) {

                }

                //adapt velocities
                for (int i = 0; i < _proxy.size(); ++i) {

                }

                iter++;

            }



        }//correct_strain_rate_error

    };//DFSPH

    namespace Detail {

        template<
            class _T,
            size_t _DIM,
            class _KERNEL,
            class _RO,
            class _DIFFERENTIAL,
            class _SPH,
            class _GRID>
        struct SimulationProxy {
            using T = _T;
            constexpr static size_t DIM = _DIM;

            using KERNEL = typename _KERNEL;

            using RO = typename _RO; //MassDensitiyEstimator

            using DIFFERENTIAL = typename _DIFFERENTIAL;

            using SPH = typename _SPH;

            using GRID = typename _GRID;

            //-----------------------------------------------

            bool is_init_step = true;
            
            //-----------------------------------------------

            T dt;
            T dti; //1/dt
            T dt2;
            T dt2i;

            //-----------------------------------------------

            // particle spacing
            T d;

            // kernel radius
            T h; // 2d
            T hi;  // 1/h
            T h2;  // h*h
            T h2i; // 1/(h*h)
            T h3;  // h*h*h
            T h3i; // 1/(h*h*h)
            T h4i; // 1/(h*h*h*h)

            T div_error = T(0.1);
            T dens_error = T(0.1);

            T ro_0 = T(1000);

            //-----------------------------------------------

            std::vector<Vector<T, DIM>> x;
            std::vector<Vector<T, DIM>> v;
            std::vector<std::vector<T>> m;
            std::vector<Vector<T, DIM>> F_ext; //external forces

            std::vector<T> alpha;
            std::vector<T> ro;

            std::vector<std::vector<int>> n; //regions
            std::vector<std::vector<T>> W;
            std::vector<std::vector<Vector<T, DIM>>> dW;
            std::vector<T> ro_s;
            std::vector<Vector<T, DIM>> v_s;

            //-----------------------------------------------

            //std::vector<Vector<T, DIM>> f_P_i;
            //std::vector<std::vector<Vector<T, DIM>>> f_P_i_j;
            //std::vector<T> k_i_v;
            //std::vector<T> ro_s_i_r0;

            template<class S = int>
            S size() const noexcept {
                return S(x.Num());
            }//size

            void resize(int _new_size) {
                is_init_step = true;
                x.resize(_new_size);
                v.resize(_new_size);
                m.resize(_new_size);
                F_ext.resize(_new_size);
                for(size_t i = 0; i < F_ext.size(); ++i)
                    F_ext[i] = Vector<T, DIM>::Zero();
                alpha.resize(_new_size);
                ro.resize(_new_size);
                W.resize(_new_size);
                dW.resize(_new_size);
                ro_s.resize(_new_size);
                v_s.resize(_new_size);
            }//resize

            void set_dt(T _dt) noexcept {
                dt = _dt;
                dti = T(1) / dt;
                dt2 = _dt * _dt;
                dt2i = T(1) / dt2;
            }//set_dt

            void set_kernel_radius(T _h) noexcept {
                is_init_step = true;
                d = _h * T(0.5);
                h = _h; // 2d
                hi = T(1) / h; // 1/h
                h2 = h*h; // h*h
                h2i = T(1) / h2; // 1/(h*h)
                h3 = h * h2;  // h*h*h
                h3i = T(1) / (h3); // 1/(h*h*h)
                h4i = T(1) / (h * h3); // 1/(h*h*h*h)
            }//set_kernel_radius

        };//SimulationProxy

        template<class PROXY>
        class SimulationStep {
            using T = typename PROXY::template T;
            constexpr static size_t DIM = PROXY::DIM;

        public:
            SimulationStep(const typename PROXY& _proxy) {

                if(_proxy.is_init_step) {
                    _proxy.is_init_step = false;

                    typename PROXY::template GRID<PROXY> grid(_proxy);

                    //init ro and alpha
                    for (int i = 0; i < _proxy.size(); ++i) {
                        _proxy.ro[i] = PROXY::RO::template  ro_i<PROXY>(_proxy, i);
                        _proxy.alpha[i] = PROXY::SPH::template alpha_i<PROXY>(_proxy, i);
                    }

                    //precompute kernels
                    for (int i = 0; i < _proxy.size(); ++i){
                        _proxy.W[i].resize(_proxy.n[i].Num());
                        _proxy.dW[i].resize(_proxy.n[i].Num());

                        for (int k = 0; k < _proxy.n[i].Num(); ++k) {
                            const int j = _proxy.n[i][k];
                            _proxy.W[i] = PROXY::KERNEL::template W_ij(_proxy, i, j);
                            _proxy.dW[i] = PROXY::KERNEL::template dW_ij(_proxy, i, j);
                        }

                    }
                }

                //adapt time step
                _proxy.set_dt(PROXY::SPH::template CFL<PROXY>(_proxy));

                //predict velocities
                for (int i = 0; i < _proxy.size(); ++i)
                    _proxy.v_s[i] = _proxy.v[i] + (_proxy.dt / _proxy.m[i]) * _proxy.F_ext[i];

                //correct Density Error
                PROXY::SPH::template correct_density_error<PROXY>(_proxy);

                //update positions
                for (int i = 0; i < _proxy.size(); ++i)
                    _proxy.x[i] = _proxy.x[i] + _proxy.dt * _proxy.v_s[i];

                //update neighbors
                typename PROXY::template GRID<PROXY> grid(_proxy);

                //update kernels
                for (int i = 0; i < _proxy.size(); ++i){
                    _proxy.W[i].resize(_proxy.n[i].Num());
                    _proxy.dW[i].resize(_proxy.n[i].Num());

                    for (int k = 0; k < _proxy.n[i].Num(); ++k) {
                        const int j = _proxy.n[i][k];
                        _proxy.W[i] = PROXY::KERNEL::template W_ij(_proxy, i, j);
                        _proxy.dW[i] = PROXY::KERNEL::template dW_ij(_proxy, i, j);
                    }

                }

                //update ro and alpha
                for (int i = 0; i < _proxy.size(); ++i) {
                    _proxy.ro[i] = PROXY::RO::template ro_i<PROXY>(_proxy, i);
                    _proxy.alpha[i] = PROXY::SPH::template alpha_i<PROXY>(_proxy, i);
                }

                //correct Divergence Error & update velocities
                PROXY::SPH::template correct_divergence_error<PROXY>(_proxy);

                for (int i = 0; i < _proxy.size(); ++i)
                    _proxy.v[i] = _proxy.v_s[i];

            }//SimulationStep

        };//SimulationStep

        //template<typename, size_t>
        //using Kerneld3 = typename Kernel::template CubicSpline<double, 3ull>;

        //template<typename, size_t, template<typename, size_t> typename>
        //using ROd3 = typename Operator::template MassDensitiyEstimator<double, 3ull, Kerneld3>;

        //template<typename, size_t, class FUNC, template<typename, size_t> typename, template<typename, size_t, template<typename, size_t> typename> typename>
        //using Differentiald3 = typename Operator::template Differential<double, 3ull, FUNC, Kerneld3, ROd3>;
    }
    //template<typename,size_t, template<typename, size_t> typename,
    //    template<typename, size_t, template<typename, size_t> typename> typename,
    //    template<typename, size_t, class FUNC, template<typename, size_t> typename, template<typename, size_t, template<typename, size_t> typename> typename> typename>
    //using DFSPH_Proxyd3 = typename Detail::template SimulationProxy<double, 3ull, Detail::Kerneld3, Detail::ROd3, Detail::Differentiald3>;

}//SPH
