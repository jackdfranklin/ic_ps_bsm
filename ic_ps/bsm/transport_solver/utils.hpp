#ifndef UTILS_H_
#define UTILS_H_

#include <array>
#include <cmath>
#include <complex>

#include <Eigen/Dense>

enum flavour_state {e, mu, tau};
enum mass_state {one = 0, two = 1, three = 2};

Eigen::VectorXd initial_flux(const Eigen::VectorXd &E_GeV, double E0, double gamma);

Eigen::VectorXd solve_iteration(const Eigen::MatrixXd &left_operator, const Eigen::VectorXd &right_side);

void fix_values(const Eigen::VectorXd &deltaE_GeV, Eigen::VectorXd &current_function, double epsilon=1e-30);

Eigen::VectorXd get_diff_flux(const Eigen::VectorXd &int_flux, const Eigen::VectorXd &E_GeV);

Eigen::VectorXd energy_bin_widths(const Eigen::VectorXd &E_GeV);

namespace utils{

    /* Weights and nodes for 3-point Gauss-Legendre quadrature */
    constexpr std::array<double, 3> w_integ = {5./9., 8./9., 5./9.};
    constexpr std::array<double, 3> x_integ = {-std::sqrt(3./5.), 0, std::sqrt(3./5.)};


    double atan_diff(double x, double y);

    double dilog(double x);

    std::complex<double> dilog(std::complex<double> z);

    void handle_gsl_error(int status);

    std::complex<double> dilogdiff(std::complex<double> x, std::complex<double> y);

    constexpr double GeV2_to_cm2 = std::pow(1.97e-14, 2);

    constexpr double cm3_to_GeV3 = std::pow(1.97e-14, -3);
}

#endif // UTILS_H_
