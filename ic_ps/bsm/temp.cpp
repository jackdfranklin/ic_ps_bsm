#include "temp.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <stdio.h>

#include <Eigen/Dense>

std::array<std::array<double, 3>,3> PMNS_squared(){
	
	//Mixing angle values from NuFIT 2022
	double theta_12 = 33.41*PI/180.0;
	double sin_theta_12 = std::sin(theta_12);
	double cos_theta_12 = std::cos(theta_12);

	double theta_23 = 49.1*PI/180.0;
	double sin_theta_23 = std::sin(theta_23);
	double cos_theta_23 = std::cos(theta_23);

	double theta_13 = 8.54*PI/180.0;
	double sin_theta_13 = std::sin(theta_13);
	double cos_theta_13 = std::cos(theta_13);

	double delta_cp = 197*PI/180.0;
	std::complex<double> exp_delta_cp(std::cos(delta_cp), std::sin(delta_cp));

	double Ue1_sq = (cos_theta_12*cos_theta_13)*(cos_theta_12*cos_theta_13);
	double Umu1_sq = std::norm(-sin_theta_12*cos_theta_23 - cos_theta_12*sin_theta_23*sin_theta_13*exp_delta_cp);
	double Utau1_sq = std::norm(sin_theta_12*sin_theta_23 - cos_theta_12*cos_theta_23*sin_theta_13*exp_delta_cp);

	double Ue2_sq = (sin_theta_12*cos_theta_13)*(sin_theta_12*cos_theta_13);
	double Umu2_sq = std::norm(cos_theta_12*cos_theta_23 - sin_theta_12*sin_theta_23*sin_theta_13*exp_delta_cp);
	double Utau2_sq = std::norm(-cos_theta_12*sin_theta_23 - sin_theta_12*cos_theta_23*sin_theta_13*exp_delta_cp);

	double Ue3_sq = sin_theta_13*sin_theta_13;
	double Umu3_sq = (sin_theta_23*cos_theta_13)*(sin_theta_23*cos_theta_13);
	double Utau3_sq = (cos_theta_23*cos_theta_13)*(cos_theta_23*cos_theta_13);

	std::array<std::array<double, 3>,3> PMNS_sq = { std::array<double,3>{Ue1_sq, Ue2_sq, Ue3_sq}, 
													std::array<double,3>{Umu1_sq, Umu2_sq, Umu3_sq}, 
													std::array<double,3>{Utau1_sq, Utau2_sq, Utau3_sq}};

	return PMNS_sq;
}

Eigen::VectorXd initial_flux(const Eigen::VectorXd &E_GeV, double E0, double gamma){
	Eigen::VectorXd result(E_GeV.size());

	double deltalog10E = ( std::log(E_GeV.tail(1)(0)) - std::log(E_GeV.head(1)(0)) )/E_GeV.size();

	for(size_t index = 0; index < E_GeV.size(); index++){
		double log10E = std::log10(E_GeV(index));
		double E_plus = std::pow(10.0, log10E + 0.5*deltalog10E);
		double E_minus = std::pow(10.0, log10E - 0.5*deltalog10E);
		result(index) = ( std::pow(E0,gamma)*( std::pow(E_plus,1.0-gamma) - std::pow(E_minus, 1.0-gamma) )/(1.0-gamma) ) / (E_plus-E_minus);
	}

	return result;
}

double K(const mass_state i, const double E_plus, const double E_minus, const std::array<std::array<double, 3>,3> &PMNS_sq, const std::array<double,3> &mass){
	
	constexpr double GF_sq_GeV = (1.166378e-5)*(1.166378e-5);
	constexpr double me_sq_GeV = (0.51099895e-3)*(0.51099895e-3);
	constexpr double sin_sq_thetaW = 0.23132; //PDG effective angle

	constexpr double g_A_sq = 0.25;
	constexpr double g_V_sq = (-0.5 +2.0*sin_sq_thetaW)*(-0.5 + 2.0*sin_sq_thetaW);

	constexpr double g_A_e_sq = 0.25;
	constexpr double g_V_e_sq = (0.5 +2.0*sin_sq_thetaW)*(0.5 + 2.0*sin_sq_thetaW);

	double E_sq = E_plus*E_plus - E_minus*E_minus;
	double deltaE = E_plus - E_minus;
	double ln_E = std::log(E_plus/E_minus);

	double sigma = 0.0;
	for (mass_state j : {one, two, three}){
		double s = mass[j]*E_sq;

		//Contribution from l=l' vv + vvbar
		sigma += (PMNS_sq[e][i]*PMNS_sq[e][j] + PMNS_sq[mu][i]*PMNS_sq[mu][j] + PMNS_sq[tau][i]*PMNS_sq[tau][j])*3.0*s/(M_PI);

		double mixing_sum = PMNS_sq[e][i]*PMNS_sq[mu][j]+PMNS_sq[mu][i]*PMNS_sq[e][j]+PMNS_sq[e][i]*PMNS_sq[tau][j]+PMNS_sq[tau][i]*PMNS_sq[e][j]+PMNS_sq[mu][i]*PMNS_sq[tau][j]+PMNS_sq[tau][i]*PMNS_sq[mu][j];	

		sigma += mixing_sum * 2.0*s/(3.0*PI);

		if(2.0*mass[j]*E_minus >= 4.0*me_sq_GeV){

			sigma += PMNS_sq[e][i]*PMNS_sq[e][j]*2.0*(s-4.0*me_sq_GeV + 3.0*me_sq_GeV*me_sq_GeV*ln_E)/(3.0*PI);

			sigma += (PMNS_sq[mu][i]*PMNS_sq[mu][j] + PMNS_sq[tau][i]*PMNS_sq[tau][j]) * (4.0*sin_sq_thetaW*(2.*sin_sq_thetaW - 1.)*(2.*me_sq_GeV*deltaE + s) - me_sq_GeV*deltaE + s)/(12.*PI);

		} else if(4.0*me_sq_GeV <= 2.0*mass[j]*E_plus){

			double E_prime = 2.0*me_sq_GeV/mass[j];
			double s_prime = mass[j]*(E_plus*E_plus - E_prime*E_prime);
			double ln_E_prime = std::log(E_plus/E_prime);
			double deltaE_prime = E_plus - E_prime;

			double phase_space_factor = std::sqrt(1 - 4*me_sq_GeV/s);

			sigma += phase_space_factor*PMNS_sq[e][i]*PMNS_sq[e][j]*(2.0*me_sq_GeV*(g_V_e_sq - 2.0*g_A_e_sq) + s*(g_V_e_sq + g_A_e_sq))/(3.0*PI);

			sigma += phase_space_factor*(PMNS_sq[mu][i]*PMNS_sq[mu][j] + PMNS_sq[tau][i]*PMNS_sq[tau][j]) * (2.0*me_sq_GeV*(g_V_sq - 2.0*g_A_sq) + s*(g_V_sq + g_A_sq))/(3.0*PI);


		}
	}

	sigma *= GF_sq_GeV;

	//printf("Energy=%g GeV, sigma=%g cm^-2 \n", E_GeV, sigma*(1.97e-9)*(1.97e-9));
	//printf("%g, %g \n", E_GeV, sigma*(1.97e-9)*(1.97e-9));

	return sigma*(1.97e-14)*(1.97e-14);

}

Eigen::VectorXd K(const mass_state i, const Eigen::VectorXd E_GeV, const std::array<std::array<double,3>,3> &PMNS_sq, const std::array<double,3> &neutrino_masses_GeV){

	Eigen::VectorXd result(E_GeV.size());
	
	double deltalog10E = ( std::log(E_GeV.tail(1)(0)) - std::log(E_GeV.head(1)(0)) )/E_GeV.size();

	for(size_t index = 0; index < E_GeV.size(); index++){
		double log10E = std::log10(E_GeV(index));
		result(index) = K(i, std::pow(10.0, log10E + 0.5*deltalog10E), std::pow(10.0, log10E - 0.5*deltalog10E), PMNS_sq, neutrino_masses_GeV);
	}

	return result;
}

double A(mass_state j, mass_state k, mass_state i, mass_state l){
	
	double a = 0.0;
	if(i==j and k==l){
		a += 1.0;	
	}
	if(i==k and j==l){
		a += 1.0;	
	}

	return a*a;
}

double B(mass_state j, mass_state k, mass_state i, mass_state l){
	
	double b = 0.0;
	if(j==k and i==l){
		b += 1.0;
	}
	if(j==i and k==l){
		b += 1.0;
	}
	return b*b;
}

double J(mass_state j, mass_state i, double En_plus, double En_minus, const std::array<double,3> &mass){ 
	
	constexpr double GF_sq_GeV = (1.166378e-5)*(1.166378e-5);

	double J_ji = 0.0;

	for(mass_state k: {one, two, three}){

		for(mass_state l: {one, two, three}){

			double a = A(j,k,i,l);
			double b = B(j,k,i,l);

			J_ji += mass[k]*( (a - b/3.0 - (a-b)/2.0)*En_plus*En_plus - a*En_plus*En_minus + b/3.0 * En_minus*En_minus*En_minus/En_plus + (a-b)/2.0 * En_minus*En_minus );
		}
	}

	return GF_sq_GeV * J_ji / (2.0*M_PI);
}

double J(mass_state j, mass_state i, double En_plus, double En_minus, double Em_plus, double Em_minus, const std::array<double,3> &mass){ 
	
	constexpr double GF_sq_GeV = (1.166378e-5)*(1.166378e-5);

	double J_ji = 0.0;

	for(mass_state k: {one, two, three}){

		for(mass_state l: {one, two, three}){

			double a = A(j,k,i,l);
			double b = B(j,k,i,l);

			J_ji += mass[k]* (a*(Em_plus - Em_minus)*(En_plus-En_minus) - b/3.0 * (1.0/Em_plus - 1.0/Em_minus)*(En_plus*En_plus*En_plus - En_minus*En_minus*En_minus));
		}
	}

	return GF_sq_GeV * J_ji / (2.0*M_PI);
}

Eigen::MatrixXd I(mass_state j, mass_state i, const Eigen::VectorXd &E_GeV, const std::array<std::array<double,3>,3> &PMNS_sq, const std::array<double,3> &neutrino_masses_GeV){
	
	Eigen::MatrixXd I_ji = Eigen::MatrixXd::Zero(E_GeV.size(), E_GeV.size());

	double deltalog10E = ( std::log(E_GeV.tail(1)(0)) - std::log(E_GeV.head(1)(0)) )/E_GeV.size();

	for(size_t n = 0; n < E_GeV.size(); n++){
		for(size_t m = n; m < E_GeV.size(); m++){
			if(m == n){
				double log10E = std::log10(E_GeV(n));
				I_ji(n,n) = J(j,i,std::pow(10, log10E+0.5*deltalog10E), std::pow(10, log10E-0.5*deltalog10E), neutrino_masses_GeV);
			} else {
				double log10En = std::log10(E_GeV(n));
				double log10Em = std::log10(E_GeV(m));
				I_ji(n,m) = J(j,i,std::pow(10, log10En+0.5*deltalog10E), std::pow(10, log10En-0.5*deltalog10E), std::pow(10, log10Em+0.5*deltalog10E), std::pow(10, log10Em-0.5*deltalog10E), neutrino_masses_GeV);
			}
		}
	}

	return 0.5 *(1.97e-14)*(1.97e-14)* I_ji;
}
	
Eigen::VectorXd solve_iteration(const Eigen::MatrixXd &left_operator, const Eigen::VectorXd &right_side){

	return left_operator.triangularView<Eigen::Upper>().solve(right_side);
}

void fix_values(const Eigen::VectorXd &deltaE_GeV, Eigen::VectorXd &current_function, double epsilon){

	for(size_t index = 0; index < current_function.size(); index++){
		if(current_function(index)*deltaE_GeV(index) < epsilon or current_function(index) < 0.0){
			current_function(index) = 0.0;
		}
	}

}

Eigen::VectorXd get_diff_flux(const Eigen::VectorXd &int_flux, const Eigen::VectorXd &E_GeV){

	double deltalog10E = ( std::log(E_GeV.tail(1)(0)) - std::log(E_GeV.head(1)(0)) )/E_GeV.size();

	Eigen::VectorXd result(int_flux.size());

	for(size_t index = 0; index < int_flux.size(); index++){
		double log10E = std::log10(E_GeV(index));
		double E_plus = std::pow(10.0, log10E + 0.5*deltalog10E);
		double E_minus = std::pow(10.0, log10E - 0.5*deltalog10E);
		double deltaE = E_plus - E_minus;
		result(index) = int_flux(index)/deltaE;
	}

	return result;
}

Eigen::VectorXd energy_bin_widths(const Eigen::VectorXd &E_GeV){

	double deltalog10E = ( std::log(E_GeV.tail(1)(0)) - std::log(E_GeV.head(1)(0)) )/E_GeV.size();

	Eigen::VectorXd result(E_GeV.size());

	for(size_t index = 0; index < E_GeV.size(); index++){
		double log10E = std::log10(E_GeV(index));
		double E_plus = std::pow(10.0, log10E + 0.5*deltalog10E);
		double E_minus = std::pow(10.0, log10E - 0.5*deltalog10E);
		double deltaE = E_plus - E_minus;
		result(index) = deltaE;
	}

	return result;
}

