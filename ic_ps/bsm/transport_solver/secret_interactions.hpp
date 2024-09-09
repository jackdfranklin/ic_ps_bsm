#ifndef SI_H
#define SI_H

#include <array>

#include <Eigen/Dense>

#include "utils.hpp"

namespace SI{

    double K(const mass_state i, 
             const double E_plus, 
             const double E_minus, 
             const std::array<double,3> &mass);

    Eigen::VectorXd K(const mass_state i, 
                      const Eigen::VectorXd E_GeV, 
                      const std::array<double,3> &neutrino_masses_GeV);

    double A(mass_state j, mass_state k, mass_state i, mass_state l);

    double B(mass_state j, mass_state k, mass_state i, mass_state l);

    double J(mass_state j, mass_state i, 
             double En_plus, double En_minus, 
             const std::array<double,3> &mass);

    double J(mass_state j, mass_state i, 
             double En_plus, double En_minus, 
             double Em_plus, double Em_minus, 
             const std::array<double,3> &mass);

    Eigen::MatrixXd I(mass_state j, mass_state i, 
                      const Eigen::VectorXd &E_GeV, 
                      const std::array<double,3> &neutrino_masses_GeV);

}

#endif
