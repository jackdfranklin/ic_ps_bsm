#ifndef TZ_H
#define TZ_H

#include <functional>

#include <Eigen/Dense>

#include "utils.hpp" 

typedef std::function<Eigen::VectorXd(mass_state, Eigen::VectorXd, std::array<double, 3>)> Loss_term; 

typedef std::function<Eigen::MatrixXd(mass_state, mass_state, Eigen::VectorXd, std::array<double, 3>)> Gain_term ;

Eigen::VectorXd transport_flux_z(Eigen::VectorXd energy_nodes, 
                                 double gamma, double zmax, 
                                 std::array<double,3> neutrino_masses_GeV, 
                                 double relic_density_cm, 
                                 const Loss_term &K, const Gain_term &I, 
                                 int steps); 

#endif
