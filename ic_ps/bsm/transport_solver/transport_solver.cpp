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

Eigen::VectorXd transport_flux(Eigen::VectorXd energy_nodes, 
                               double gamma, double distance_Mpc, 
                               std::array<double,3> neutrino_masses_GeV, 
                               double relic_density_cm, 
                               const Loss_term &K, const Gain_term &I, 
                               int steps) {
	
	size_t N_nodes = energy_nodes.size();
	double distance_cm = 3.086e+24 * distance_Mpc;
	double delta_r = distance_cm/steps;

	Eigen::VectorXd deltaE_GeV = energy_bin_widths(energy_nodes);

	Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(N_nodes, N_nodes);

	const Eigen::MatrixXd I_11 =  I(one, one, energy_nodes, 
                                        neutrino_masses_GeV);
        //std::cout<<I_11<<std::endl;
	const Eigen::MatrixXd I_21 =  I( two, one, energy_nodes, 
                                        neutrino_masses_GeV);
	const Eigen::MatrixXd I_31 =  I(three, one, energy_nodes, 
                                        neutrino_masses_GeV);

	const Eigen::MatrixXd I_12 =  I(one, two, energy_nodes, 
                                        neutrino_masses_GeV);
	const Eigen::MatrixXd I_22 =  I(two, two, energy_nodes, 
                                        neutrino_masses_GeV);
	const Eigen::MatrixXd I_32 =  I(three, two, energy_nodes, 
                                        neutrino_masses_GeV);

	const Eigen::MatrixXd I_13 =  I(one, three, energy_nodes, 
                                        neutrino_masses_GeV);
	const Eigen::MatrixXd I_23 =  I(two, three, energy_nodes, 
                                        neutrino_masses_GeV);
	const Eigen::MatrixXd I_33 =  I(three, three, energy_nodes, 
                                        neutrino_masses_GeV);

	const Eigen::VectorXd K_1 = K(one,   energy_nodes, neutrino_masses_GeV);
        //std::cout<<K_1<<std::endl;
	const Eigen::VectorXd K_2 = K(two,   energy_nodes, neutrino_masses_GeV);
	const Eigen::VectorXd K_3 = K(three, energy_nodes, neutrino_masses_GeV);

	Eigen::MatrixXd D_1 = ( 1.0 / ( relic_density_cm * delta_r ) )
                              * deltaE_GeV.asDiagonal();
	D_1 += 0.5 * K_1.asDiagonal();
	D_1 -= I_11;

	Eigen::MatrixXd D_2 = ( 1.0 / ( relic_density_cm * delta_r ) )
                              * deltaE_GeV.asDiagonal();
	D_2 += 0.5 * K_2.asDiagonal();
	D_2 -= I_22;

	Eigen::MatrixXd D_2_inv = D_2.inverse();

	Eigen::MatrixXd D_3 = ( 1.0 / ( relic_density_cm * delta_r ) )
                              * deltaE_GeV.asDiagonal();
	D_3 += 0.5 * K_3.asDiagonal();
	D_3 -= I_33;

	Eigen::MatrixXd sigma_1 = ( 1.0 / ( relic_density_cm * delta_r ) )
                                  * deltaE_GeV.asDiagonal();
	sigma_1 -= 0.5 * K_1.asDiagonal();

	Eigen::MatrixXd sigma_2 = ( 1.0 / ( relic_density_cm * delta_r ) )
                                  * deltaE_GeV.asDiagonal();
	sigma_2 -= 0.5 * K_2.asDiagonal();

	Eigen::MatrixXd sigma_3 = ( 1.0 / ( relic_density_cm * delta_r ) )
                                  * deltaE_GeV.asDiagonal();
	sigma_3 -= 0.5 * K_3.asDiagonal();

	Eigen::MatrixXd G_1 = D_1 - I_21 * D_2_inv * I_12;
	Eigen::MatrixXd G_3 = D_3 - I_23 * D_2_inv * I_32;
	Eigen::MatrixXd G_3_inv = G_3.inverse();

	Eigen::MatrixXd d_3 = ( I_23 * D_2_inv * I_32 + I_31 ) * G_3_inv;
	Eigen::MatrixXd d_2 = I_21 * D_2_inv + d_3 * I_32 * D_2_inv;

	Eigen::MatrixXd c_3 = I_13 + I_23 * D_2_inv * I_12;

	Eigen::MatrixXd M_1 = G_1 
                            - ( I_31 + I_21 * D_2_inv * I_32 ) * G_3_inv * c_3;
	Eigen::MatrixXd M_1_inv = M_1.inverse();


	double norm = 1.0/3.0;
	Eigen::VectorXd current_1 = norm 
                                    * initial_flux(energy_nodes, 1000.0, gamma)
                                    * ( 2.0 * constants::PMNS_sq[mu][one] 
                                        + constants::PMNS_sq[e][one]);

        Eigen::VectorXd current_2 = norm 
                                    * initial_flux(energy_nodes, 1000.0, gamma) 
                                    * ( 2.0 * constants::PMNS_sq[mu][two] 
                                        + constants::PMNS_sq[e][two]) ;

	Eigen::VectorXd current_3 = norm 
                                    * initial_flux(energy_nodes, 1000.0, gamma)
                                    * ( 2.0 * constants::PMNS_sq[mu][three] 
                                        + constants::PMNS_sq[e][three]);


        Eigen::VectorXd del_1, del_2, del_3;
        Eigen::VectorXd b_1, b_3;
        Eigen::VectorXd new_1, new_2, new_3;

	for (int step = 0; step < steps; step++){

		del_1 = sigma_1 * current_1 
                                         + I_11 * current_1 
                                         + I_21 * current_2 
                                         + I_31 * current_3;

		del_2 = sigma_2 * current_2 
                                         + I_12 * current_1 
                                         + I_22 * current_2 
                                         + I_32 * current_3;

		del_3 = sigma_3 * current_3 
                                        + I_13 * current_1 
                                        + I_23 * current_2 
                                        + I_33 * current_3;
												   
		b_1 = del_1 + d_2 * del_2 + d_3 * del_3;
		b_3 = del_3 + I_23 * D_2_inv * del_2;
		
		new_1 = 
                    M_1.triangularView<Eigen::Upper>().solve(b_1);
		new_3 = 
                    G_3.triangularView<Eigen::Upper>().solve(b_3 
                                                             + c_3 * current_1);
		new_2 = 
                    D_2.triangularView<Eigen::Upper>().solve(del_2 
                                                             + I_12 * new_1 
                                                             + I_32 * new_3);

		fix_values(deltaE_GeV, new_1);
		fix_values(deltaE_GeV, new_2);
		fix_values(deltaE_GeV, new_3);

		std::swap(new_1, current_1);
		std::swap(new_2, current_2);
		std::swap(new_3, current_3);

	}

	return ( constants::PMNS_sq[mu][one]   * current_1  
               + constants::PMNS_sq[mu][three] * current_3 
               + constants::PMNS_sq[mu][two]   * current_2);
	
}

Eigen::VectorXd transport_flux_SM(Eigen::VectorXd energy_nodes,
                                  double gamma, double distance_Mpc, 
                                  std::array<double,3> neutrino_masses_GeV, 
                                  double relic_density_cm, 
                                  int steps) {

    const Loss_term K = SM::K;

    const Gain_term I = SM::I;

    return transport_flux(energy_nodes, gamma, distance_Mpc, 
                          neutrino_masses_GeV, relic_density_cm, 
                          K, I, steps);

}

Eigen::VectorXd transport_flux_SI(double g, double m_phi,
                                  Eigen::VectorXd energy_nodes,
                                  double gamma, double distance_Mpc, 
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

    return transport_flux(energy_nodes, gamma, distance_Mpc, 
                          neutrino_masses_GeV, relic_density_cm, 
                          K, I, steps);

}

Eigen::VectorXd transport_flux_SM_redshift(Eigen::VectorXd energy_nodes,
                                  double gamma, double zmax, 
                                  std::array<double,3> neutrino_masses_GeV, 
                                  double relic_density_cm, 
                                  int steps) {

    const Loss_term K = SM::K;

    const Gain_term I = SM::I;

    return transport_flux_z(energy_nodes, gamma, zmax, 
                          neutrino_masses_GeV, relic_density_cm, 
                          K, I, steps);

}

Eigen::VectorXd transport_flux_SI_redshift(double g, double m_phi,
                                  Eigen::VectorXd energy_nodes,
                                  double gamma, double zmax, 
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

    return transport_flux_z(energy_nodes, gamma, zmax, 
                          neutrino_masses_GeV, relic_density_cm, 
                          K, I, steps);

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
