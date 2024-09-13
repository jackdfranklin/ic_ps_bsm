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

        constexpr flavour_state flavour = tau;

        // Scalar decay width
        const double decay_width = (majorana ? 1.0 : 2.0) * std::pow(g, 2) * m_phi 
                             / ( 16.0 * M_PI );
        const double m_phi_sq = std::pow(m_phi, 2);
        const double Gamma_M = Gamma / m_phi;

        double K_tot = 0.0;

        for (mass_state j : {one, two, three}) {
        
            double s_plus  = 2.0 * mass.at(j) * E_plus  / std::pow(m_phi, 2);
            double s_minus = 2.0 * mass.at(j) * E_minus / std::pow(m_phi, 2);
    
            K_tot += K_s(j, flavour, g, m_phi, majorana, decay_width, 
                         mass.at(j), s_plus, s_minus);

            K_tot += K_t_u(j, flavour, g, m_phi, majorana, decay_width, 
                         mass.at(j), s_plus, s_minus);

            K_tot += K_tu(j, flavour, g, m_phi, majorana, decay_width, 
                         mass.at(j), s_plus, s_minus);
            
            double k_st = K_st(j, flavour, g, m_phi, majorana, decay_width, 
                               mass.at(j), s_plus, s_minus);

            K_tot += k_st;

            // s-u interference (only for Majorana fermions)
            if (majorana) {

                double k_su = k_st;
                K_tot += k_su;

            }

            K_tot += K_pp(j, flavour, g, m_phi, majorana, decay_width, 
                         mass.at(j), s_plus, s_minus);

        }

        return utils::GeV2_to_cm2 * K_tot;

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

        return utils::GeV2_to_cm2 * 0.5 * I_ji;
    }

    double K_s(mass_state j, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width, double m_j,
               double s_plus, double s_minus) {

        double gamma_m = decay_width / m_phi;

        double k_s = std::pow(g, 4) 
                     / ( 32.0 * M_PI * std::pow(m_phi, 2) * decay_width);

        // If s_plus small, Taylor expand
        if(s_plus < 1e-5){

            k_s *= ( 2.0 * m_phi * 
                ( ( gamma_m * ( 1.0 + std::pow(gamma_m, 2) ) + 2.0 * s_minus )
                  * ( s_plus - s_minus)
                  / ( std::pow( 1.0 + std::pow(gamma_m, 2) , 2) )
                  + gamma_m * std::pow( s_plus - s_minus , 2) 
                        / ( std::pow( 1.0 + std::pow(gamma_m, 2) , 2) ) )
                + decay_width * ( std::log1p( s_plus * ( s_plus - 2.0 ) 
                                        / ( 1.0 + std::pow(gamma_m, 2) ) )
                          - std::log1p( s_minus * ( s_minus - 2.0 ) 
                                        / ( 1.0 + std::pow(gamma_m, 2) ) ) 
                          ) ); 

        }
        else {

            k_s *= ( 2.0 * m_phi 
                     * utils::atan_diff( m_phi * ( s_plus  - 1.0 ) / decay_width,
                                         m_phi * ( s_minus - 1.0 ) / decay_width )
                    + decay_width * ( std::log1p( s_plus * ( s_plus - 2.0 ) 
                                            / ( 1.0 + std::pow(gamma_m, 2) ) ) 
                              - std::log1p( s_minus * ( s_minus - 2.0 )
                                            / ( 1.0 + std::pow(gamma_m, 2) ) ) 
                              )
                       );

        }

        k_s *= constants::PMNS_sq[flavour][j];

        return ( std::pow(m_phi, 2) / ( 2.0 * m_j ) ) * k_s;
        

    }

    double K_t_u(mass_state j, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width, double m_j, 
               double s_plus, double s_minus) {

        // t-u channel without inteference
        double k_t_u = std::pow(g, 4) / ( 16.0 * M_PI * std::pow(m_phi, 2) )
                            * ( 2.0 * std::log1p(s_plus)  / s_plus 
                              - 2.0 * std::log1p(s_minus) / s_minus
                              + std::log1p(s_plus) - std::log1p(s_minus) 
                              );

        // If negative, rounding errors have occurred so fix to zero
        if (k_t_u < 0.0) {

            k_t_u = 0.0;

        }

        k_t_u *= 2.0 * constants::PMNS_sq[flavour][j];
        return ( std::pow(m_phi, 2) / ( 2.0 * m_j ) ) * k_t_u;

    double K_st(mass_state j, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width, double m_j, 
               double s_plus, double s_minus) {

        // Interference between u and t channels
        double k_tu =  ( std::pow(g, 4) 
                         / ( 32.0 * M_PI * std::pow(m_phi, 2) * s_minus * s_plus ) )
                       * ( s_minus * std::log1p(s_plus)  
                           * ( 2.0 + 2.0 * s_plus 
                                   + s_plus * std::log( 2.0 + s_plus) )
                         - s_plus  * std::log1p(s_minus) 
                           * (2.0 + 2.0 * s_minus 
                                  + s_minus * std::log( 2.0 + s_minus) ) 
                         + s_minus * s_plus 
                           * ( utils::dilog( -1.0 - s_plus ) 
                             - utils::dilog( -1.0 - s_minus) 
                             + utils::dilog( -s_plus )
                             - utils::dilog( -s_minus ) 
                             ) 
                         );

        k_tu *= constants::PMNS_sq[flavour][j];

        if (!majorana) {

            k_tu *= 0.5;

        }

        return ( std::pow(m_phi, 2) / ( 2.0 * m_j ) ) * k_tu;

    }

    double K_st(mass_state j, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width, double m_j, 
               double s_plus, double s_minus) {

        using namespace std::complex_literals;

        double gamma_m = decay_width / m_phi;

        std::complex<double> denom = 2i + gamma_m;

        std::complex<double> z1_plus  = 1i * ( 1.0 + s_plus )  / denom;
        std::complex<double> z1_minus = 1i * ( 1.0 + s_minus ) / denom;

        std::complex<double> z2_plus  = std::conj(z1_plus);
        std::complex<double> z2_minus = std::conj(z1_minus);

        std::complex<double> dilogdiff_z1 = utils::dilog(z1_plus) 
                                          - utils::dilog(z1_minus);
        std::complex<double> dilogdiff_z2 = utils::dilog(z2_plus) 
                                          - utils::dilog(z2_minus);

        double k_st = dilogdiff_z1.real() + dilogdiff_z2.real()
                    + gamma_m 
                      * ( dilogdiff_z2.imag() - dilogdiff_z1.imag() 
                        + 2.0 * std::arg( 1.0 - z2_plus) * std::log1p(s_plus)
                        - 2.0 * std::arg( 1.0 - z2_minus ) * std::log1p(s_minus)
                        )
                    + std::log1p( 4.0 / std::pow(gamma_m, 2) ) 
                        * ( std::log1p(s_minus) - std::log1p(s_plus) )
                    + std::log1p( std::pow( (-1.0 + s_plus ) / gamma_m , 2) )
                        * ( std::log1p(s_plus) - 1.0 - std::pow(gamma_m, 2) )
                    - std::log1p( std::pow( (-1.0 + s_minus ) / gamma_m , 2) )
                        * ( std::log1p(s_minus) - 1.0 - std::pow(gamma_m, 2) ) 
                    + 2.0 * ( utils::dilog(-s_plus) - utils::dilog(-s_minus) );

        k_st *= - ( std::pow(g, 4) 
                  / ( 32.0 * M_PI * std::pow(m_phi, 2) 
                           * (1.0 + std::pow(gamma_m, 2) ) ) 
                  );

        k_st *= constants::PMNS_sq[flavour][j];

        return ( std::pow(m_phi, 2) / ( 2.0 * mass.at(j) ) ) * k_st;
    
    }

    double K_pp(mass_state j, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width, double m_j, 
               double s_plus, double s_minus) {

        /* Double scalar production */
        double k_pp = 0.0;

        if( s_plus > 4.0 ){

            if( s_minus > 4.0 ) {

                k_pp +=  12.0 * std::sqrt( ( -4.0 + s_minus ) / s_minus );

                k_pp -= 12.0 * std::sqrt( ( -4.0 + s_plus ) / s_plus ); 
                         
                k_pp -= 2 * std::log( std::pow( std::sqrt( -4.0 + s_minus ) 
                                                - std::sqrt(s_minus), 2) / 4.0)
                          * std::log( std::pow( -2.0 + s_minus 
                                                + std::sqrt( ( -4.0 + s_minus ) 
                                                             * s_minus ) 
                                                , 2) / 4.0);
                         
                k_pp -= ( 6.0 + s_minus * std::log( ( -2.0 + s_minus ) * s_minus ) ) 
                        * std::log( std::pow( -2.0 + s_minus 
                                                + std::sqrt( ( -4.0 + s_minus ) 
                                                              * s_minus )
                                                , 2)
                                      / std::pow( 2.0 - s_minus 
                                          + std::sqrt( ( -4.0 + s_minus ) 
                                                       * s_minus )
                                                 , 2) )
                        / s_minus; 

                k_pp -= 24.0 * ( std::sqrt( ( -4.0 + s_minus ) / s_minus ) 
                                - std::sqrt( ( -4.0 + s_plus ) / s_plus ) 
                                - std::log( std::sqrt( -4.0 + s_minus ) 
                                            + std::sqrt(s_minus)) 
                                + std::log( std::sqrt( -4.0 + s_plus ) 
                                            + std::sqrt(s_plus))
                                ); 

                k_pp += 2.0 * std::log( std::pow( std::sqrt( -4.0 + s_plus ) 
                                                 - std::sqrt(s_plus), 2) / 4.0)
                            * std::log( std::pow( std::sqrt( ( -4.0 + s_plus ) 
                                                             * s_plus )
                                                  - 2.0 + s_plus 
                                                 , 2) / 4.0 ); 

                k_pp +=  ( 6.0 + s_plus * std::log( ( -2.0 + s_plus ) * s_plus ) )
                         * std::log( std::pow( -2.0 + s_plus 
                                        + std::sqrt( ( -4.0 + s_plus ) * s_plus )
                                              , 2)
                                  / std::pow( 2.0 - s_plus 
                                        + std::sqrt( ( -4.0 + s_plus ) * s_plus ) 
                                             , 2) )
                         / s_plus ;

                k_pp += 8.0 * ( utils::dilog( -4.0 
                                              / std::pow( std::sqrt(s_minus) 
                                                    + std::sqrt( -4.0 + s_minus )
                                                         , 2) )
                               - utils::dilog( -4.0 
                                               / std::pow( std::sqrt(s_plus) 
                                                    + std::sqrt( -4.0 + s_plus )
                                                          , 2) )
                              ); 

                k_pp += 2.0 * ( utils::dilog( -4.0 
                                      / std::pow( -2.0 + s_minus 
                                                  + std::sqrt( s_minus 
                                                        * ( -4.0 + s_minus ) ) 
                                             , 2) 
                                            )
                              - utils::dilog( -4.0 
                                      / std::pow( -2.0 + s_plus 
                                                  + std::sqrt( s_plus 
                                                        * ( -4.0 + s_plus ) ) 
                                             , 2) 
                                            ) 
                              );

            }
            else {

                constexpr double dilog1m = - std::pow(M_PI, 2) / 12.0;

                //Replace s_minus with 4
                k_pp -= 12.0 * std::sqrt( ( -4.0 + s_plus ) / s_plus ); 

                k_pp -= 24.0 * ( -std::sqrt( ( -4.0 + s_plus ) / s_plus ) 
                                 - std::log(2) 
                                 + std::log( std::sqrt( -4.0 + s_plus ) 
                                             + std::sqrt(s_plus)
                                           )
                               ); 

                k_pp += 2.0 * std::log( std::pow( std::sqrt( -4.0 + s_plus ) 
                                                  - std::sqrt(s_plus)
                                             , 2) 
                                        / 4.0 )
                        * std::log( std::pow( -2.0 + s_plus 
                                        + std::sqrt( ( -4.0 + s_plus ) * s_plus )
                                         , 2)
                                    / 4.0 ); 

                k_pp += ( 6.0 + s_plus * std::log( ( -2.0 + s_plus ) * s_plus ) )
                        * std::log( std::pow( -2.0 + s_plus 
                                        + std::sqrt( ( -4.0 + s_plus ) * s_plus )
                                         , 2)
                                    / std::pow( 2.0 - s_plus 
                                                + std::sqrt( ( -4.0 + s_plus ) 
                                                             * s_plus 
                                                           )
                                           , 2)
                                  )
                        / s_plus; 

                k_pp += 8.0 * ( dilogm1 
                                - utils::dilog( -4.0 
                                        / std::pow( std::sqrt( -4.0 + s_plus) 
                                                    + std::sqrt(s_plus) 
                                              , 2 ) 
                                              ) 
                              ); 

                k_pp += 2.0 * ( dilogm1 
                                - utils::dilog( -4.0  
                                        / std::pow( -2.0 + s_plus 
                                              + std::sqrt( ( -4.0 + s_plus )
                                                             * s_plus )
                                              , 2 )
                                              )
                              );

            }

            k_pp *= constants::PMNS_sq[flavour][j];

            k_pp *= std::pow(g, 4) / ( 128.0 * M_PI * std::pow(mphi, 2) )

            // For Majorana fermions, we can scatter off neutrinos and antineutrinos
            if(majorana) {
                Gamma_pp *= 2.0;
            }

        }

        return std::pow(m_phi, 2) / ( 2.0 * m_j ) * k_pp;

    }

} //SI
