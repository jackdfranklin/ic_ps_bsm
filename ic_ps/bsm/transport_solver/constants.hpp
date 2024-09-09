#ifndef CONSTS_H
#define CONSTS_H

#include <cmath>
#include <complex>

namespace constants{

	constexpr double theta_12 = 33.41 * M_PI / 180.0;
	constexpr double sin_theta_12 = std::sin(theta_12);
	constexpr double cos_theta_12 = std::cos(theta_12);

	constexpr double theta_23 = 49.1 * M_PI / 180.0;
	constexpr double sin_theta_23 = std::sin(theta_23);
	constexpr double cos_theta_23 = std::cos(theta_23);

	constexpr double theta_13 = 8.54 * M_PI / 180.0;
	constexpr double sin_theta_13 = std::sin(theta_13);
	constexpr double cos_theta_13 = std::cos(theta_13);

	constexpr double delta_cp = 197 * M_PI / 180.0;
	constexpr std::complex<double> exp_delta_cp(std::cos(delta_cp), std::sin(delta_cp));

	constexpr double Ue1_sq = (cos_theta_12 * cos_theta_13) * (cos_theta_12 * cos_theta_13);
	constexpr double Umu1_sq = std::norm(-sin_theta_12 * cos_theta_23 - cos_theta_12 * sin_theta_23 * sin_theta_13 * exp_delta_cp);
	constexpr double Utau1_sq = std::norm(sin_theta_12 * sin_theta_23 - cos_theta_12 * cos_theta_23 * sin_theta_13 * exp_delta_cp);

	constexpr double Ue2_sq = (sin_theta_12 * cos_theta_13) * (sin_theta_12 * cos_theta_13);
	constexpr double Umu2_sq = std::norm(cos_theta_12 * cos_theta_23 - sin_theta_12 * sin_theta_23 * sin_theta_13 * exp_delta_cp);
	constexpr double Utau2_sq = std::norm(-cos_theta_12 * sin_theta_23 - sin_theta_12 * cos_theta_23 * sin_theta_13 * exp_delta_cp);

	constexpr double Ue3_sq = sin_theta_13 * sin_theta_13;
	constexpr double Umu3_sq = (sin_theta_23 * cos_theta_13) * (sin_theta_23 * cos_theta_13);
	constexpr double Utau3_sq = (cos_theta_23 * cos_theta_13) * (cos_theta_23 * cos_theta_13);

	constexpr std::array<std::array<double, 3>,3> PMNS_sq = { std::array<double,3>{Ue1_sq, Ue2_sq, Ue3_sq}, 
								  std::array<double,3>{Umu1_sq, Umu2_sq, Umu3_sq}, 
								  std::array<double,3>{Utau1_sq, Utau2_sq, Utau3_sq}};

	constexpr double GF_sq_GeV = (1.166378e-5)*(1.166378e-5);

	constexpr double me_sq_GeV = (0.51099895e-3)*(0.51099895e-3);

	constexpr double sin_sq_thetaW = 0.23132; //PDG effective angle

}

#endif
