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

Eigen::VectorXd transport_flux(Eigen::VectorXd energy_nodes, 
                               double gamma, 
                               double distance_Mpc, 
                               std::array<double,3> neutrino_masses_GeV, 
                               double relic_density_cm, 
                               int steps){
	
	size_t N_nodes = energy_nodes.size();
	double distance_cm = 3.086e+24 * distance_Mpc;
	double delta_r = distance_cm/steps;

	Eigen::VectorXd deltaE_GeV = energy_bin_widths(energy_nodes);

	Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(N_nodes, N_nodes);

	const Eigen::MatrixXd I_11 =  SM::I(one, one, energy_nodes, 
                                        neutrino_masses_GeV);
	const Eigen::MatrixXd I_21 =  SM::I( two, one, energy_nodes, 
                                        neutrino_masses_GeV);
	const Eigen::MatrixXd I_31 =  SM::I(three, one, energy_nodes, 
                                        neutrino_masses_GeV);

	const Eigen::MatrixXd I_12 =  SM::I(one, two, energy_nodes, 
                                        neutrino_masses_GeV);
	const Eigen::MatrixXd I_22 =  SM::I(two, two, energy_nodes, 
                                        neutrino_masses_GeV);
	const Eigen::MatrixXd I_32 =  SM::I(three, two, energy_nodes, 
                                        neutrino_masses_GeV);

	const Eigen::MatrixXd I_13 =  SM::I(one, three, energy_nodes, 
                                        neutrino_masses_GeV);
	const Eigen::MatrixXd I_23 =  SM::I(two, three, energy_nodes, 
                                        neutrino_masses_GeV);
	const Eigen::MatrixXd I_33 =  SM::I(three, three, energy_nodes, 
                                        neutrino_masses_GeV);

	const Eigen::VectorXd K_1 = SM::K(one,   energy_nodes, neutrino_masses_GeV);
	const Eigen::VectorXd K_2 = SM::K(two,   energy_nodes, neutrino_masses_GeV);
	const Eigen::VectorXd K_3 = SM::K(three, energy_nodes, neutrino_masses_GeV);

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


	for (int step = 0; step < steps; step++){

		Eigen::VectorXd del_1 = sigma_1 * current_1 
                                         + I_11 * current_1 
                                         + I_21 * current_2 
                                         + I_31 * current_3;

		Eigen::VectorXd del_2 = sigma_2 * current_2 
                                         + I_12 * current_1 
                                         + I_22 * current_2 
                                         + I_32 * current_3;

		Eigen::VectorXd del_3 = sigma_3 * current_3 
                                        + I_13 * current_1 
                                        + I_23 * current_2 
                                        + I_33 * current_3;
												   
		Eigen::VectorXd b_1 = del_1 + d_2 * del_2 + d_3 * del_3;
		Eigen::VectorXd b_3 = del_3 + I_23 * D_2_inv * del_2;
		
		Eigen::VectorXd new_1 = 
                    M_1.triangularView<Eigen::Upper>().solve(b_1);
		Eigen::VectorXd new_3 = 
                    G_3.triangularView<Eigen::Upper>().solve(b_3 
                                                             + c_3 * current_1);
		Eigen::VectorXd new_2 = 
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

PYBIND11_MODULE(transport_solver, mod) {

	mod.doc() = "Neutrino flux transport equation solver";

	mod.def("transport_flux", &transport_flux, "Module for solving the propagation of a neutrino flux from a point source through space");
}
