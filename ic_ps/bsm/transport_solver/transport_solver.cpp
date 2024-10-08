#include <functional>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdio.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "utils.hpp"
#include "constants.hpp"
#include "sm_interactions.hpp"
#include "secret_interactions.hpp"
#include "transport_flux_redshift.hpp"

typedef std::function<Eigen::VectorXd(mass_state, Eigen::VectorXd, std::array<double, 3>)> Loss_term; 

typedef std::function<Eigen::MatrixXd(mass_state, mass_state, Eigen::VectorXd, std::array<double, 3>)> Gain_term ;

Eigen::MatrixXd transport_flux(Eigen::VectorXd energy_nodes, 
                               Eigen::VectorXd gamma_grid, double distance_Mpc, 
                               std::array<double,3> neutrino_masses_GeV, 
                               double relic_density_cm, 
                               const Loss_term &K, const Gain_term &I, 
                               int steps) {
	
    size_t N_nodes = energy_nodes.size();
    double distance_cm = 3.086e+24 * distance_Mpc;
    double delta_r = distance_cm/steps;

    Eigen::VectorXd deltaE_GeV = energy_bin_widths(energy_nodes);
    double deltalog10E = std::log10(energy_nodes.head(2)(1)) 
        - std::log10(energy_nodes.head(1)(0));

    std::array< std::array<Eigen::MatrixXd, 3>, 3> Is;
    Is.at(one).at(one) =  I(one, one, energy_nodes, 
                            neutrino_masses_GeV);
    Is.at(two).at(one) =  I( two, one, energy_nodes, 
                            neutrino_masses_GeV);
    Is.at(three).at(one) =  I(three, one, energy_nodes, 
                            neutrino_masses_GeV);

    Is.at(one).at(two) =  I(one, two, energy_nodes, 
            neutrino_masses_GeV);
    Is.at(two).at(two) =  I(two, two, energy_nodes, 
            neutrino_masses_GeV);
    Is.at(three).at(two) =  I(three, two, energy_nodes, 
            neutrino_masses_GeV);

    Is.at(one).at(three) =  I(one, three, energy_nodes, 
            neutrino_masses_GeV);
    Is.at(two).at(three) =  I(two, three, energy_nodes, 
            neutrino_masses_GeV);
    Is.at(three).at(three) =  I(three, three, energy_nodes, 
            neutrino_masses_GeV);

    std::array<Eigen::VectorXd, 3> Ks;
    Ks.at(one) = K(one,   energy_nodes, neutrino_masses_GeV);
    Ks.at(two) = K(two,   energy_nodes, neutrino_masses_GeV);
    Ks.at(three) = K(three, energy_nodes, neutrino_masses_GeV);

    double norm = 1.0/3.0;
    std::array<Eigen::VectorXd, 3> fluxes;

    Eigen::MatrixXd final_muon_fluxes(energy_nodes.size(), gamma_grid.size());

    for(auto i = 0; i < gamma_grid.size(); ++i){

        double gamma = gamma_grid(i);

        fluxes.at(one) = norm 
            * ( 2.0 * constants::PMNS_sq[mu][one] + constants::PMNS_sq[e][one])
            * initial_flux(energy_nodes, 1000.0, gamma).array() * deltaE_GeV.array();

        fluxes.at(two) = norm 
            * ( 2.0 * constants::PMNS_sq[mu][two] 
                    + constants::PMNS_sq[e][two])
            * initial_flux(energy_nodes, 1000.0, gamma).array() * deltaE_GeV.array();

        fluxes.at(three) = norm 
            * ( 2.0 * constants::PMNS_sq[mu][three] 
                    + constants::PMNS_sq[e][three])
            * initial_flux(energy_nodes, 1000.0, gamma).array() * deltaE_GeV.array();

        for (int step = 0; step < steps; step++){

            for( auto m = N_nodes - 1; m > 0; m-- ) {

                Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
                Eigen::Vector3d rhs = Eigen::Vector3d::Zero();

                double deltaE_m = deltaE_GeV(m);

                double E_m = energy_nodes(m);

                double E_m_minus_half = 
                    std::pow(10, std::log10(E_m) 
                            - 0.5 * deltalog10E );
                double E_m_plus_half = 
                    std::pow(10, std::log10(E_m) + 0.5 * deltalog10E );

                for ( mass_state i: {one, two, three} ) {

                    M(i, i) += 1.0 / delta_r; 
                    // Loss term from cross section
                    M(i, i) += relic_density_cm * Ks.at(i)(m) / deltaE_m;

                    // Gain term from up/down scattering
                    for ( mass_state j: {one, two, three} ) {
                        M(j, i) -= relic_density_cm * Is.at(j).at(i)(m, m) / deltaE_m;
                    }

                    rhs(i) += ( 1.0 / delta_r ) * fluxes.at(i)(m);

                    if ( m != N_nodes - 1 ) {
                        double E_m_plus_1 = energy_nodes(m + 1);

                        for ( mass_state j: {one, two, three} ) {

                            // Gain term from up/down scattering
                            rhs(i) += relic_density_cm 
                                * ( Is.at(j).at(i).row(m).tail(N_nodes - ( m + 1 ) )
                                    .transpose().array() 
                                    * fluxes.at(j).tail(N_nodes - m - 1).array() 
                                    / deltaE_GeV.tail(N_nodes - m - 1).array() )
                                .sum();
                        }

                    }
                }

                Eigen::Vector3d sol = M.fullPivLu().solve(rhs);

                fluxes.at(one)(m) = sol(one);
                fluxes.at(two)(m) = sol(two);
                fluxes.at(three)(m) = sol(three);

            }


            for ( mass_state i: {one, two, three} ) {
                fix_values(deltaE_GeV, fluxes.at(i));
            }

        }

        final_muon_fluxes.col(i) = 
            (   constants::PMNS_sq[mu][one]   * fluxes.at(one)  
              + constants::PMNS_sq[mu][three] * fluxes.at(two) 
              + constants::PMNS_sq[mu][two]   * fluxes.at(three) ).array()
            / deltaE_GeV.array();


    }

    return final_muon_fluxes;
}

Eigen::MatrixXd transport_flux_SM(Eigen::VectorXd energy_nodes,
                                  Eigen::VectorXd gamma_grid, double distance_Mpc, 
                                  std::array<double,3> neutrino_masses_GeV, 
                                  double relic_density_cm, 
                                  int steps) {

    const Loss_term K = SM::K;

    const Gain_term I = SM::I;

    return transport_flux(energy_nodes, gamma_grid, distance_Mpc, 
                          neutrino_masses_GeV, relic_density_cm, 
                          K, I, steps);

}

Eigen::MatrixXd transport_flux_SI(double g, double m_phi,
                                  Eigen::VectorXd energy_nodes,
                                  Eigen::VectorXd gamma_grid, double distance_Mpc, 
                                  std::array<double,3> neutrino_masses_GeV, 
                                  double relic_density_cm, 
                                  int steps) {
    bool majorana = true;

    const Loss_term K = [=](mass_state i, const Eigen::VectorXd E_GeV, 
                        const std::array<double,3> &neutrino_masses_GeV) {
        
        return SI::K(i, E_GeV, neutrino_masses_GeV, g, m_phi, majorana);

    };

    const Gain_term I = [=](mass_state j, mass_state i, 
                        const Eigen::VectorXd E_GeV, 
                        const std::array<double,3> &neutrino_masses_GeV) {
        
        return SI::I(j, i, E_GeV, neutrino_masses_GeV, g, m_phi, majorana);

    };

    return transport_flux(energy_nodes, gamma_grid, distance_Mpc, 
                          neutrino_masses_GeV, relic_density_cm, 
                          K, I, steps);

}

Eigen::MatrixXd transport_flux_SM_redshift(Eigen::VectorXd energy_nodes,
                                  Eigen::VectorXd gamma_grid, double zmax, 
                                  std::array<double,3> neutrino_masses_GeV, 
                                  double relic_density_cm ) {

    const Loss_term K = SM::K;

    const Gain_term I = SM::I;

    return transport_flux_z(energy_nodes, gamma_grid, zmax, 
                          neutrino_masses_GeV, relic_density_cm, 
                          K, I);

}

Eigen::MatrixXd transport_flux_SI_redshift(double g, double m_phi,
                                  Eigen::VectorXd energy_nodes,
                                  Eigen::VectorXd gamma_grid, 
                                  double zmax, 
                                  std::array<double,3> neutrino_masses_GeV, 
                                  double relic_density_cm) {
    bool majorana = true;

    const Loss_term K = [=](mass_state i, const Eigen::VectorXd E_GeV, 
                        const std::array<double,3> &neutrino_masses_GeV) {
        
        return SI::K(i, E_GeV, neutrino_masses_GeV, g, m_phi, majorana);

    };

    const Gain_term I = [=](mass_state j, mass_state i, 
                        const Eigen::VectorXd E_GeV, 
                        const std::array<double,3> &neutrino_masses_GeV) {
        
        return SI::I(j, i, E_GeV, neutrino_masses_GeV, g, m_phi, majorana);

    };

    return transport_flux_z(energy_nodes, gamma_grid, zmax, 
                          neutrino_masses_GeV, relic_density_cm, 
                          K, I);

}

PYBIND11_MODULE(transport_solver, mod) {

	mod.doc() = "Neutrino flux transport equation solver";

	mod.def("transport_flux", &transport_flux_SM, 
                "Module for solving the propagation of a neutrino flux from a  \
                 point source through space");

        mod.def("transport_flux_SI", &transport_flux_SI, 
                "Function for propagating a neutrino flux from a               \ 
                 point source through space with secret interactions");

	mod.def("transport_flux_SM_redshift", &transport_flux_SM_redshift, 
                "Function for finding the final flux at Earth of a neutrino  \
                 flux from a point source at redshift zmax undergoing SM interactions");

        mod.def("transport_flux_SI_redshift", &transport_flux_SI_redshift, 
                "Function for finding the final flux at Earth of a neutrino  \
                 flux from a point source at redshift zmax undergoing SI interactions");
}
