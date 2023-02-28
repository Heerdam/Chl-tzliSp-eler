
#include "../include/dfsph.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

using namespace SPH;

TEST_CASE( "Kernel", "CubicSpline" ) {

    constexpr double eps = 1.0e-5;
    constexpr double supportRadius = 4. * 0.025;
    constexpr unsigned int numberOfSteps = 50;
    constexpr double stepSize = 2. * supportRadius / (double)(numberOfSteps - 1);

    DefaultProxy proxy;
    proxy.resize(numberOfSteps);
    proxy.set_kernel_radius(supportRadius);

    {
        proxy.x[0] = Eigen::Vector3d(0., 0., 0.);
        proxy.x[0] = Eigen::Vector3d(1., 1., 1.);
        const double v = DefaultProxy::KERNEL::W_ij(proxy, 0, 1);
    }

    {
        proxy.x[0] = Eigen::Vector3d(0., 0., 0.);
        proxy.x[0] = Eigen::Vector3d(1., 1., 1.);
        const Eigen::Vector3d v = DefaultProxy::KERNEL::dW_ij(proxy, 0, 1);
    }

    Eigen::Vector3d xi = Eigen::Vector3d::Zero();
    Eigen::Vector3d sumV = Eigen::Vector3d::Zero();

    double sum = 0.;
    double V = pow(stepSize, 3);

    bool positive = true;

    for (unsigned int i = 0; i < numberOfSteps; ++i) {
        for (unsigned int j = 0; j < numberOfSteps; ++j) {
            for (unsigned int k = 0; k < numberOfSteps; ++k) {

                const Eigen::Vector3d xj(-supportRadius + i*stepSize, -supportRadius + j*stepSize, -supportRadius + k*stepSize);

                
                const double W = TestType::W(xi - xj);
                sum += W *V;
                sumV += TestType::gradW(xi - xj) * V;
                if (W < -eps) positive = false;
            }
        }
    }
    REQUIRE(std::fabs(sum - 1.) < eps);
    REQUIRE(sumV.norm() < eps);
    REQUIRE(positive);

}