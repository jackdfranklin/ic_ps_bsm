#include "sm_interactions.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <stdio.h>

#include <Eigen/Dense>

#include "constants.hpp"

namespace SM{

    double K( const mass_state i, 
              const double E_plus, 
              const double E_minus, 
              const std::array<double,3> &mass){

        constexpr double g_A_sq = 0.25;
        constexpr double g_V_sq = 
                        std::pow( (-0.5 + 2.0 * constants::sin_sq_thetaW ), 2);

        constexpr double g_A_e_sq = 0.25;
        constexpr double g_V_e_sq = 
                        std::pow( (0.5 + 2.0 * constants::sin_sq_thetaW ), 2);

        double E_sq = E_plus * E_plus - E_minus * E_minus;
        double deltaE = E_plus - E_minus;
        double ln_E = std::log( E_plus / E_minus );

        double sigma = 0.0;

        for (mass_state j : {one, two, three}) {

            double s = mass.at(j) * E_sq;

            //Contribution from l=l' vv + vvbar
            sigma += ( constants::PMNS_sq[e][i]   * constants::PMNS_sq[e][j] 
                      + constants::PMNS_sq[mu][i]  * constants::PMNS_sq[mu][j] 
                      + constants::PMNS_sq[tau][i] * constants::PMNS_sq[tau][j]) 
                     * 3.0 * s / M_PI;

            double mixing_sum = 
                      constants::PMNS_sq[e][i]   * constants::PMNS_sq[mu][j] 
                    + constants::PMNS_sq[mu][i]  * constants::PMNS_sq[e][j] 
                    + constants::PMNS_sq[e][i]   * constants::PMNS_sq[tau][j] 
                    + constants::PMNS_sq[tau][i] * constants::PMNS_sq[e][j] 
                    + constants::PMNS_sq[mu][i]  * constants::PMNS_sq[tau][j] 
                    + constants::PMNS_sq[tau][i] * constants::PMNS_sq[mu][j];	

            sigma += mixing_sum * 2.0 * s / (3.0 * M_PI);

            if( 2.0 * mass.at(j) * E_minus >= 4.0 * constants::me_sq_GeV ){

                sigma += 
                    constants::PMNS_sq[e][i] * constants::PMNS_sq[e][j]
                    * 2.0 * ( s - 4.0 * constants::me_sq_GeV 
                              + 3.0 * std::pow(constants::me_sq_GeV, 2) * ln_E )
                    / (3.0 * M_PI);

                sigma += 
                    (  constants::PMNS_sq[mu][i]  * constants::PMNS_sq[mu][j] 
                     + constants::PMNS_sq[tau][i] * constants::PMNS_sq[tau][j] )
                    * ( 4.0 * constants::sin_sq_thetaW 
                        * ( 2. * constants::sin_sq_thetaW - 1. ) 
                        * ( 2. * constants::me_sq_GeV * deltaE + s ) 
                        - constants::me_sq_GeV * deltaE + s )
                    / (12. * M_PI);

            } 
            else if( 4.0 * constants::me_sq_GeV <= 2.0 * mass.at(j) * E_plus ){

                double E_prime = 2.0 * constants::me_sq_GeV/mass.at(j);
                double s_prime = 
                            mass.at(j) * (E_plus * E_plus - E_prime * E_prime);
                double ln_E_prime = std::log(E_plus/E_prime);
                double deltaE_prime = E_plus - E_prime;

                double phase_space_factor = 
                            std::sqrt( 1.0 - 4.0 * constants::me_sq_GeV / s);

                sigma += phase_space_factor 
                        * constants::PMNS_sq[e][i] * constants::PMNS_sq[e][j] 
                        * ( 2.0 * constants::me_sq_GeV 
                            * (g_V_e_sq - 2.0 * g_A_e_sq) 
                            + s * (g_V_e_sq + g_A_e_sq) )
                        / (3.0 * M_PI);

                sigma += 
                    phase_space_factor 
                    * ( constants::PMNS_sq[mu][i] * constants::PMNS_sq[mu][j] 
                      + constants::PMNS_sq[tau][i] * constants::PMNS_sq[tau][j] ) 
                    * ( 2.0 * constants::me_sq_GeV * ( g_V_sq - 2.0 * g_A_sq ) 
                            + s * ( g_V_sq + g_A_sq ) )
                    / ( 3.0 * M_PI );


            }
        }

        sigma *= constants::GF_sq_GeV;

        return sigma * (1.97e-14) * (1.97e-14);

    }

    Eigen::VectorXd K(const mass_state i, 
                      const Eigen::VectorXd E_GeV, 
                      const std::array<double,3> &neutrino_masses_GeV){

        Eigen::VectorXd result(E_GeV.size());

        double deltalog10E = 
                    ( std::log(E_GeV.tail(1)(0)) - std::log(E_GeV.head(1)(0)) ) 
                    / E_GeV.size();

        for(size_t index = 0; index < E_GeV.size(); index++){

            double log10E = std::log10( E_GeV(index) );

            result(index) = K( i, 
                               std::pow( 10.0, log10E + 0.5 * deltalog10E ), 
                               std::pow( 10.0, log10E - 0.5 * deltalog10E ), 
                               neutrino_masses_GeV );

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

        return a * a;
    }

    double B(mass_state j, mass_state k, mass_state i, mass_state l){

        double b = 0.0;

        if(j==k and i==l){

            b += 1.0;

        }

        if(j==i and k==l){

            b += 1.0;

        }

        return b * b;
    }

    double J(mass_state j, mass_state i, 
            double En_plus, double En_minus, 
            const std::array<double,3> &mass){ 

        double J_ji = 0.0;

        for(mass_state k: {one, two, three}){

            for(mass_state l: {one, two, three}){

                double a = A(j,k,i,l);
                double b = B(j,k,i,l);
                                                 
                J_ji += mass[k] 
                        * ( ( a - b / 3.0 - ( a - b ) / 2.0) * En_plus * En_plus 
                            - a * En_plus * En_minus 
                            + b / 3.0 * En_minus * En_minus * En_minus / En_plus 
                            + ( a - b ) / 2.0 * En_minus * En_minus );

            }
        }

        return constants::GF_sq_GeV * J_ji / (2.0 * M_PI);
    }

    double J(mass_state j, mass_state i, 
            double En_plus, double En_minus, double Em_plus, double Em_minus, 
            const std::array<double,3> &mass){ 

        double J_ji = 0.0;

        for(mass_state k: {one, two, three}){

            for(mass_state l: {one, two, three}){

            double a = A(j,k,i,l);
            double b = B(j,k,i,l);

            J_ji += mass[k] 
                    * ( a * (Em_plus - Em_minus) * (En_plus-En_minus) 
                        - ( b / 3.0 ) * ( 1.0 / Em_plus - 1.0 / Em_minus ) 
                            * ( std::pow(En_plus, 3) - std::pow(En_minus, 3) ) );

            }
        }

        return constants::GF_sq_GeV * J_ji / (2.0 * M_PI);
    }

    Eigen::MatrixXd I(mass_state j, mass_state i, 
                      const Eigen::VectorXd &E_GeV, 
                      const std::array<double,3> &neutrino_masses_GeV){

        Eigen::MatrixXd I_ji = Eigen::MatrixXd::Zero(E_GeV.size(), E_GeV.size());

        double deltalog10E = 
                ( std::log( E_GeV.tail(1)(0) ) - std::log( E_GeV.head(1)(0) ) ) 
                / E_GeV.size();

        for(size_t n = 0; n < E_GeV.size(); n++){

            for(size_t m = n; m < E_GeV.size(); m++){

                if(m == n){

                    double log10E = std::log10(E_GeV(n));
                    I_ji(n,n) =  J( j, i, 
                    std::pow( 10, log10E + 0.5 * deltalog10E ), 
                    std::pow( 10, log10E - 0.5 * deltalog10E ), 
                    neutrino_masses_GeV);

                } 
                else {

                    double log10En = std::log10(E_GeV(n));
                    double log10Em = std::log10(E_GeV(m));

                    I_ji(n,m) =  J( j, i, 
                                    std::pow( 10, log10En + 0.5 * deltalog10E ), 
                                    std::pow( 10, log10En - 0.5 * deltalog10E ), 
                                    std::pow( 10, log10Em + 0.5 * deltalog10E ), 
                                    std::pow( 10, log10Em - 0.5 * deltalog10E ), 
                                    neutrino_masses_GeV);

                }
            }
        }

        return 0.5 *(1.97e-14) * (1.97e-14)* I_ji;
    }

} //SM
