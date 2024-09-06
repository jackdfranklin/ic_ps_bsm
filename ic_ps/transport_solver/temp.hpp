#ifndef TEMP_H_
#define TEMP_H_
#endif // TEMP_H_

enum flavour {e, mu, tau};
enum mass_state {one, two, three};

# define PI 3.141592653589793

std::array<std::array<double, 3>,3> PMNS_squared();

Eigen::VectorXd initial_flux(const Eigen::VectorXd &E_GeV, double E0, double gamma);

double K(const mass_state i, const double E_plus, const double E_minus, const std::array<std::array<double, 3>,3> &PMNS_sq, const std::array<double,3> &mass);

Eigen::VectorXd K(const mass_state i, const Eigen::VectorXd E_GeV, const std::array<std::array<double,3>,3> &PMNS_sq, const std::array<double,3> &neutrino_masses_GeV);

double A(mass_state j, mass_state k, mass_state i, mass_state l);

double B(mass_state j, mass_state k, mass_state i, mass_state l);

double J(mass_state j, mass_state i, double En_plus, double En_minus, const std::array<double,3> &mass);

double J(mass_state j, mass_state i, double En_plus, double En_minus, double Em_plus, double Em_minus, const std::array<double,3> &mass);

Eigen::MatrixXd I(mass_state j, mass_state i, const Eigen::VectorXd &E_GeV, const std::array<std::array<double,3>,3> &PMNS_sq, const std::array<double,3> &neutrino_masses_GeV);
	
Eigen::VectorXd solve_iteration(const Eigen::MatrixXd &left_operator, const Eigen::VectorXd &right_side);

void fix_values(const Eigen::VectorXd &deltaE_GeV, Eigen::VectorXd &current_function, double epsilon=1e-30);

Eigen::VectorXd get_diff_flux(const Eigen::VectorXd &int_flux, const Eigen::VectorXd &E_GeV);

Eigen::VectorXd energy_bin_widths(const Eigen::VectorXd &E_GeV);
