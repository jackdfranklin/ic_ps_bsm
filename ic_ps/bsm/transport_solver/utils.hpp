#ifndef UTILS_H_
#define UTILS_H_

#include <array>
#include <cmath>

#include <Eigen/Dense>

enum flavour {e, mu, tau};
enum mass_state {one = 0, two = 1, three = 2};

Eigen::VectorXd initial_flux(const Eigen::VectorXd &E_GeV, double E0, double gamma);

Eigen::VectorXd solve_iteration(const Eigen::MatrixXd &left_operator, const Eigen::VectorXd &right_side);

void fix_values(const Eigen::VectorXd &deltaE_GeV, Eigen::VectorXd &current_function, double epsilon=1e-30);

Eigen::VectorXd get_diff_flux(const Eigen::VectorXd &int_flux, const Eigen::VectorXd &E_GeV);

Eigen::VectorXd energy_bin_widths(const Eigen::VectorXd &E_GeV);

namespace utils{

    double atan_diff(double x, double y);

}

#endif // UTILS_H_
