#include "secret_interactions.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <stdio.h>

#include <Eigen/Dense>

#include "constants.hpp"
#include "utils.hpp"

namespace SI {

    double K(const mass_state i, 
             const double E_plus, 
             const double E_minus, 
             const std::array<double,3> &mass,
             const double g,
             const double m_phi,
             const bool majorana){

        // Scalar decay width
        const double Gamma = (majorana ? 1.0 : 2.0) * std::pow(g, 2) * m_phi 
                             / ( 16.0 * M_PI );
        const double m_phi_sq = std::pow(m_phi, 2);
        const double Gamma_M = Gamma / m_phi;

        double K_tot = 0.0;

        for (mass_state j : {one, two, three}) {
        
            double s_plus  = 2.0 * mass.at(j) * E_plus  / std::pow(m_phi, 2);
            double s_minus = 2.0 * mass.at(j) * E_minus / std::pow(m_phi, 2);
    
            double K_s = std::pow(g, 4) / ( 32.0 * M_PI * m_phi_sq * Gamma);

            // If s_plus small, Taylor expand
            if(s_plus < 1e-5){

                K_s *= ( 2.0 * m_phi * 
                    ( ( Gamma_M * ( 1.0 + std::pow(Gamma_M, 2) ) + 2.0 * s_minus )
                      * ( s_plus - s_minus)
                      / ( std::pow( 1.0 + std::pow(Gamma_M, 2) , 2) )
                      + Gamma_M * std::pow( s_plus - s_minus , 2) 
                            / ( std::pow( 1.0 + std::pow(Gamma_M, 2) , 2) ) )
                    + Gamma * ( std::log1p( s_plus * ( s_plus - 2.0 ) 
                                            / ( 1.0 + std::pow(Gamma_M, 2) ) )
                              - std::log1p( s_minus * ( s_minus - 2.0 ) 
                                            / ( 1.0 + std::pow(Gamma_M, 2) ) ) 
                              ) ); 

            }
            else {

                K_s *= ( 2.0 * m_phi 
                         * utils::atan_diff( m_phi * ( s_plus  - 1.0 ) / Gamma,
                                             m_phi * ( s_minus - 1.0 ) / Gamma)
                        + Gamma * ( std::log1p( s_plus * ( s_plus - 2.0 ) 
                                                / ( 1.0 + std::pow(Gamma_M, 2) ) ) 
                                  - std::log1p( s_minus * ( s_minus - 2.0 )
                                                / ( 1.0 + std::pow(Gamma_M, 2) ) ) 
                                  )
                           );

            }

            K_s *= constants::PMNS_sq[tau][j];

            K_tot += ( m_phi_sq / ( 2.0 * mass.at(j) ) ) * K_s;

            // t-u channel without inteference
            double K_t_u = std::pow(g, 4) / ( 16.0 * M_PI * m_phi_sq)
                                * ( 2.0 * std::log1p(s_plus)  / s_plus 
                                  - 2.0 * std::log1p(s_minus) / s_minus
                                  + std::log1p(s_plus) - std::log1p(s_minus) );

        }

        return K_tot * std::pow( 1.97e-14, 2);

    }

    Eigen::VectorXd K(const mass_state i, 
                      const Eigen::VectorXd E_GeV, 
                      const std::array<double,3> &neutrino_masses_GeV,
                      const double g, const double m_phi, bool majorana) {

        Eigen::VectorXd result(E_GeV.size());

        double deltalog10E = 
                    ( std::log(E_GeV.tail(1)(0)) - std::log(E_GeV.head(1)(0)) ) 
                    / E_GeV.size();

        for(size_t index = 0; index < E_GeV.size(); index++){

            double log10E = std::log10( E_GeV(index) );

            result(index) = K( i, 
                               std::pow( 10.0, log10E + 0.5 * deltalog10E ), 
                               std::pow( 10.0, log10E - 0.5 * deltalog10E ), 
                               neutrino_masses_GeV, g, m_phi, majorana);

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
                                                 
                J_ji += 0.0;

            }
        }

        return 0.0;
    }

    double J(mass_state j, mass_state i, 
            double En_plus, double En_minus, double Em_plus, double Em_minus, 
            const std::array<double,3> &mass){ 

        double J_ji = 0.0;

        for(mass_state k: {one, two, three}){

            for(mass_state l: {one, two, three}){

                double a = A(j,k,i,l);
                double b = B(j,k,i,l);

                J_ji += 0.0; 

            }
        }

        return 0.0;
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

} //SI
