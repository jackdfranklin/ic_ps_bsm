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

        K_tot *= constants::PMNS_sq[flavour][i];

        return utils::GeV2_to_cm2 * K_tot;

    }

    Eigen::VectorXd K(const mass_state i, 
                      const Eigen::VectorXd E_GeV, 
                      const std::array<double,3> &neutrino_masses_GeV,
                      const double g, const double m_phi, bool majorana) {

        Eigen::VectorXd result(E_GeV.size());

        double deltalog10E = 
                ( std::log10( E_GeV.head(2)(1) ) - std::log10( E_GeV.head(1)(0) ) ); 

        for(size_t index = 0; index < E_GeV.size(); index++){

            double log10E = std::log10( E_GeV(index) );

            result(index) = K( i, 
                               std::pow( 10.0, log10E + 0.5 * deltalog10E ), 
                               std::pow( 10.0, log10E - 0.5 * deltalog10E ), 
                               neutrino_masses_GeV, g, m_phi, majorana);

        }

        return result;
    }

    double J(mass_state j, mass_state i, 
            double En_plus, double En_minus, 
            const std::array<double,3> &mass, 
            const double g,
            const double m_phi,
            const bool majorana) {

        const flavour_state flavour = tau;

        const double decay_width = (majorana ? 1.0 : 2.0) * std::pow(g, 2) * m_phi 
                             / ( 16.0 * M_PI );

        double J_ji = 0.0;

        for(mass_state k: {one, two, three}){

            double t_plus  = -2.0 * mass.at(k) * En_plus  / std::pow(m_phi, 2);
            double t_minus = -2.0 * mass.at(k) * En_minus / std::pow(m_phi, 2);
            if( std::fabs( t_minus + 1 ) < 1e-7 ) {
                t_minus += t_minus * 1e-6;
            }
            if( std::fabs( t_plus + 1 ) < 1e-7) {
                t_plus += t_plus * 1e-6;
            }

            J_ji += J_s(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                        t_plus, t_minus);

            J_ji += J_t(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                        t_plus, t_minus);

            J_ji += J_u(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                    t_plus, t_minus);

            J_ji += J_tu(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                         t_plus, t_minus);

            double temp = J_st(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                        t_plus, t_minus);

            if(temp < 0.0){

                J_st(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                        t_plus, t_minus);
            }

            J_ji += temp;

            J_ji += J_su(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                         t_plus, t_minus);

        }

        J_ji *= constants::PMNS_sq[flavour][i] * constants::PMNS_sq[flavour][j];

        return J_ji;
    }

    double J(mass_state j, mass_state i, 
            double En_plus, double En_minus, double Em_plus, double Em_minus, 
            const std::array<double,3> &mass, 
            const double g,
            const double m_phi,
            const bool majorana) {

        double J_ji = 0.0;

        const flavour_state flavour = tau;

        const double decay_width = (majorana ? 1.0 : 2.0) * std::pow(g, 2) * m_phi 
                             / ( 16.0 * M_PI );

        for(mass_state k: {one, two, three}){

            double t_plus  = -2.0 * mass.at(k) * En_plus  / std::pow(m_phi, 2);
            double t_minus = -2.0 * mass.at(k) * En_minus / std::pow(m_phi, 2);
            double s_plus  =  2.0 * mass.at(k) * Em_plus  / std::pow(m_phi, 2);
            double s_minus =  2.0 * mass.at(k) * Em_minus / std::pow(m_phi, 2);

            J_ji += J_s(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                        t_plus, t_minus, s_plus, s_minus);

            J_ji += J_t(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                        t_plus, t_minus, s_plus, s_minus);

            J_ji += J_u(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                        t_plus, t_minus, s_plus, s_minus);

            J_ji += J_tu(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                         t_plus, t_minus, s_plus, s_minus);

            J_ji += J_st(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                         t_plus, t_minus, s_plus, s_minus);

            J_ji += J_su(k, flavour, g, m_phi, majorana, decay_width, mass.at(k), 
                         t_plus, t_minus, s_plus, s_minus);

            if(std::isnan(J_ji)) {
                std::cout<<En_minus<<"->"<<En_plus<<"; "<<Em_minus<<"->"<<Em_plus<<std::endl;
            }

        }

        J_ji *= constants::PMNS_sq[flavour][i] * constants::PMNS_sq[flavour][j];

        return J_ji;
    }

    Eigen::MatrixXd I(mass_state j, mass_state i, 
                      const Eigen::VectorXd &E_GeV, 
                      const std::array<double,3> &neutrino_masses_GeV,
                      const double g,
                      const double m_phi,
                      const bool majorana) {

        Eigen::MatrixXd I_ji = Eigen::MatrixXd::Zero(E_GeV.size(), E_GeV.size());

        double deltalog10E = 
                    ( std::log10( E_GeV.head(2)(1) ) - std::log10( E_GeV.head(1)(0) ) ); 

        for(size_t n = 0; n < E_GeV.size(); n++){

            for(size_t m = n; m < E_GeV.size(); m++){

                if(m == n){

                    double log10E = std::log10(E_GeV(n));
                    I_ji(n,n) =  J( j, i, 
                                    std::pow( 10, log10E + 0.5 * deltalog10E ), 
                                    std::pow( 10, log10E - 0.5 * deltalog10E ), 
                                    neutrino_masses_GeV, g, m_phi, majorana);

                } 
                else {

                    double log10En = std::log10(E_GeV(n));
                    double log10Em = std::log10(E_GeV(m));

                    I_ji(n,m) =  J( j, i, 
                                    std::pow( 10, log10En + 0.5 * deltalog10E ), 
                                    std::pow( 10, log10En - 0.5 * deltalog10E ), 
                                    std::pow( 10, log10Em + 0.5 * deltalog10E ), 
                                    std::pow( 10, log10Em - 0.5 * deltalog10E ), 
                                    neutrino_masses_GeV, g, m_phi, majorana);

                }
            }
        }

        return utils::GeV2_to_cm2 * I_ji;
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

    }

    double K_tu(mass_state j, flavour_state flavour, double g, double m_phi, 
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

        return ( std::pow(m_phi, 2) / ( 2.0 * m_j ) ) * k_st;
    
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

                constexpr double dilogm1 = - std::pow(M_PI, 2) / 12.0;

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

            k_pp *= std::pow(g, 4) / ( 128.0 * M_PI * std::pow(m_phi, 2) );

            // For Majorana fermions, we can scatter off neutrinos and antineutrinos
            if(majorana) {
                k_pp *= 2.0;
            }

        }

        return std::pow(m_phi, 2) / ( 2.0 * m_j ) * k_pp;

    }

    double J_s(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width, double m_k, 
               double t_plus, double t_minus) {

        double j_s = std::pow(g, 4) 
                              / ( 16.0 * M_PI * decay_width * std::pow(m_phi, 2) );
        double gamma_m = decay_width / m_phi;

        if (fabs(t_plus) < 1e-5) {
            j_s *= ( 2.0 * m_phi * ( 1.0 + t_minus ) 
                              * ( ( gamma_m * std::pow( -t_minus + t_plus , 2) ) 
                                  / std::pow( 1 + std::pow(gamma_m, 2), 2) 
                                - ( gamma_m 
                                    * ( 1.0 + std::pow(gamma_m, 2) 
                                        - 2.0 * t_minus )
                                    * ( -t_minus + t_plus )
                                    / std::pow( 1.0 + std::pow(gamma_m, 2) , 2) 
                                  ) 
                                )
                            + decay_width * ( std::log1p( 
                                            t_plus * ( t_plus + 2.0 ) 
                                            / ( 1.0 + std::pow(gamma_m, 2) ) 
                                                  )  
                                      - std::log1p( t_minus * ( t_minus + 2.0 ) 
                                            / ( 1.0 + std::pow(gamma_m, 2) ) 
                                                  ) 
                                      )
                            );
        }
        else {
            j_s *= ( 2.0 * m_phi * ( 1.0 + t_minus ) 
                              * utils::atan_diff( m_phi * ( 1.0 + t_minus ) 
                                                        / decay_width , 
                                                  m_phi * ( 1.0 + t_plus ) 
                                                        / decay_width )
                            + decay_width * ( std::log1p( t_plus * ( t_plus + 2.0 ) 
                                            / ( 1.0 + std::pow(gamma_m, 2) ) )  
                                        - std::log1p( t_minus * ( t_minus + 2.0 ) 
                                              / ( 1.0 + std::pow(gamma_m, 2) ) ) 
                                      )
                            );
        }

         j_s *= constants::PMNS_sq[flavour][k];

        if(!majorana) {
            // For Dirac, one of the final neutrinos is not observable
             j_s /= 2.0; 
        }

        return std::pow(m_phi, 4) / ( 2.0 * m_k ) * j_s;

    }

    double J_t(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width , double m_k, 
               double t_plus, double t_minus) {
        double j_t;

        if (majorana) {
            j_t = std::pow(g, 4) *
                ( 1.0 / ( 16.0 * std::pow(m_phi, 2) * M_PI * ( -1.0 + t_minus ) 
                          * t_plus ) 
                  * ( ( -2.0 + t_minus ) * ( t_minus - t_plus ) 
                    - ( -1.0 + t_minus ) * ( -2.0 + t_plus ) 
                      * ( std::log1p(-t_minus) - std::log1p(-t_plus) ) )
                 + 1.0 / ( 16.0 * std::pow(m_phi, 2) * M_PI 
                           * std::pow( 1.0 + t_minus , 2) * t_plus ) 
                 * ( ( 1.0 + t_minus ) * ( 2.0 + t_minus ) * ( t_minus - t_plus )
                   + ( -2.0 * std::pow( 1.0 + t_minus , 2) + t_plus 
                       + 2.0 * t_minus * t_plus ) 
                     * std::log1p( t_minus - t_plus )
                   - std::pow(t_minus, 2) * t_plus * std::log( t_minus / t_plus ) 
                   )
                );
            if(j_t < 0){ // Roundoff errors! Compute the integral numerically
                double a_y = t_plus, b_y = t_minus, a_x[3], b_x[3];
                // Nodes at which the integrand will be evaluated
                double y[3], x[3][3], F[3][3];
                j_t = 0;
                for(int i=0; i<3; ++i){
                    y[i] = (b_y-a_y)/2. * utils::x_integ[i] + (b_y+a_y)/2.;
                    a_x[i] = -y[i];
                    b_x[i] = -t_plus;
                    for(int j=0; j<3; ++j){
                        x[i][j] = (b_x[i]-a_x[i])/2. * utils::x_integ[j] + (b_x[i]+a_x[i])/2.;
                        F[i][j] = std::pow(y[i]/x[i][j], 2) / std::pow(y[i]-1, 2) +
                            std::pow((-x[i][j]-y[i])/x[i][j], 2) / std::pow((-x[i][j]-y[i])-1, 2);
                        j_t += 1./4. * (b_y - a_y) * (b_x[i]-a_x[i]) * utils::w_integ[i] * utils::w_integ[j] * F[i][j];
                    }
                }
                j_t *= std::pow(g, 4)/(16*M_PI * std::pow(m_phi, 4));
            }
        }
        else{
            j_t = ( 3.0 / 2.0 ) * std::pow(g, 4) 
                  / ( 32.0 * std::pow(m_phi, 2) * M_PI 
                           * ( -1.0 + t_minus ) * t_plus ) 
                  * ( ( -2.0 + t_minus ) * ( t_minus - t_plus ) 
                    - ( -1.0 + t_minus ) * ( -2.0 + t_plus ) 
                      * ( std::log1p(-t_minus) - std::log1p(-t_plus) )
                    );
            if(j_t < 0){ // Roundoff errors! Compute the integral numerically
                double a_y = t_plus, b_y = t_minus, a_x[3], b_x[3];
                // Nodes at which the integrand will be evaluated
                double y[3], x[3][3], F[3][3];
                j_t = 0;
                for(int i=0; i<3; ++i){
                    y[i] = (b_y-a_y)/2. * utils::x_integ[i] + (b_y+a_y)/2.;
                    a_x[i] = -y[i];
                    b_x[i] = -t_plus;
                    for(int j=0; j<3; ++j){
                        x[i][j] = (b_x[i]-a_x[i])/2. * utils::x_integ[j] + (b_x[i]+a_x[i])/2.;
                        F[i][j] = std::pow(y[i]/x[i][j], 2) / std::pow(y[i]-1, 2);
                        j_t += 1./4. * (b_y - a_y) * (b_x[i]-a_x[i]) * utils::w_integ[i] * utils::w_integ[j] * F[i][j];
                    }
                }
                j_t *= 3./2. * std::pow(g, 4)/(32*M_PI * std::pow(m_phi, 4));
            }
        }

        j_t *= constants::PMNS_sq[flavour][k];

        return std::pow(m_phi, 4) / ( 2.0 * m_k ) * j_t;
    }

    double J_u(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width , double m_k, 
               double t_plus, double t_minus) {

        if(majorana)
            return J_t(k, flavour, g, m_phi, majorana, decay_width, m_k, 
                       t_plus, t_minus);
        else{
            double j_u = ( 0.5 * std::pow(g, 4) 
                           / ( 32.0 * std::pow(m_phi, 4) * M_PI 
                                 * ( -1.0 + t_minus ) * t_plus ) 
                         )
                         * ( ( -2.0 + t_minus ) * ( t_minus - t_plus ) 
                             - ( -1.0 + t_minus ) * ( -2.0 + t_plus ) 
                               * ( std::log1p(-t_minus) - std::log1p(-t_plus) ) 
                  );
            if(j_u < 0){ // Roundoff errors! Compute the integral numerically
                double a_y = t_plus, b_y = t_minus, a_x[3], b_x[3];
                // Nodes at which the integrand will be evaluated
                double y[3], x[3][3], F[3][3];
                j_u = 0;
                for(int i=0; i<3; ++i){
                    y[i] = (b_y-a_y)/2. * utils::x_integ[i] + (b_y+a_y)/2.;
                    a_x[i] = -y[i];
                    b_x[i] = -t_plus;
                    for(int j=0; j<3; ++j){
                        x[i][j] = (b_x[i]-a_x[i])/2. * utils::x_integ[j] + (b_x[i]+a_x[i])/2.;
                        F[i][j] = std::pow(y[i]/x[i][j], 2) / std::pow(y[i]-1, 2);
                        j_u += 1./4. * (b_y - a_y) * (b_x[i]-a_x[i]) * utils::w_integ[i] * utils::w_integ[j] * F[i][j];
                    }
                }
                j_u *= 1./2. * std::pow(g, 4)/(32*M_PI * std::pow(m_phi, 4));
            }
            j_u *= constants::PMNS_sq[flavour][k];
            return std::pow(m_phi, 4) / ( 2.0 * m_k ) * j_u;
        }
    }

    double J_tu(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width , double m_k, 
               double t_plus, double t_minus) {
        /* t-u interference */
        double j_tu;
        if (majorana) {
            double dilog_combi;
            if(-t_plus < 1e-2 && -t_minus < 1e-2){
                double delta = t_plus/t_minus;
                dilog_combi = -( ( ( -1.0 + delta ) * t_plus 
                                   * std::log( -2.0 * t_plus ) ) 
                                 / delta ) 
                              - ( ( -1.0 + delta ) * std::pow(t_plus, 2) 
                                  * ( -2.0 + delta + delta * std::log(2) 
                                      + std::log( -2.0 / t_plus ) 
                                      - delta * std::log(-t_plus) 
                                    ) 
                                ) / ( 2.0 * std::pow(delta, 2) ) 
                              + ( std::pow(t_plus, 3) 
                                  * ( 8.0 - 30.0 * delta 
                                      + 21.0 * std::pow(delta, 2) 
                                      + std::pow(delta, 3) 
                                      - 8.0 * std::pow(delta, 3) * std::log(2) 
                                      + std::log(256) + 8.0 * std::log(-t_plus) 
                                      - 8.0 * std::pow(delta, 3) 
                                        * std::log(-t_plus) ) ) 
                                / ( 24.0 * std::pow(delta, 3) ) 
                              + ( std::pow(t_plus, 4) 
                                  * ( -32.0 + 56.0 * delta 
                                      - 51.0 * std::pow(delta, 2) 
                                      + 30.0 * std::pow(delta, 3) 
                                      - 3.0 * std::pow(delta, 4) 
                                      + std::log(4096) 
                                      - std::pow(delta, 4) * std::log(4096) 
                                      - 12.0 * std::log(-t_plus) 
                                      + 12.0 * std::pow(delta, 4) 
                                        * std::log(-t_plus) ) )
                                / ( 48.0 * std::pow(delta, 4) );

            } else
            if ( -t_plus > 1e2 && -t_minus > 1e2 ) {
                double delta = t_plus/t_minus;
                dilog_combi = ( -2.0 * ( -1.0 + delta ) 
                                * std::log( ( -1.0 + delta ) / delta ) ) / t_plus 
                              - ( 2.0 
                                  * ( -1.0 + std::log( -delta 
                                                  / ( ( -1.0 + delta ) * t_plus ) 
                                                     ) 
                                    ) ) / std::pow(t_plus, 2) 
                              + ( -6.0 + 4.0 * delta + std::pow(delta, 2) 
                                  - 2.0 * std::pow(delta, 3) 
                                  - 8.0 * std::log( ( delta - 1.0 ) / delta ) 
                                  + 8.0 * delta * std::log( ( delta - 1.0 ) / delta ) 
                                  + 2.0 * std::pow(delta, 3) 
                                        * std::log( ( delta - 1.0 ) / delta ) 
                                  - 2.0 * std::pow(delta, 4) 
                                        * std::log( ( delta - 1.0 ) / delta ) 
                                  - 6.0 * std::log(-t_plus) 
                                  + 6.0 * delta * std::log(-t_plus)
                                  ) 
                                  / ( 3.0 * ( delta - 1.0 ) 
                                          * std::pow(t_plus, 3) ) 
                                  + ( 8.0 - 12.0 * delta 
                                      + 3.0 * std::pow(delta, 2) 
                                      + 12.0 * std::log( ( delta - 1.0 ) / delta )
                                      - 24.0 * delta 
                                             * std::log( ( delta - 1.0 ) / delta ) 
                                      + 12.0 * std::pow(delta, 2)
                                             * std::log( ( delta - 1.0 ) / delta ) 
                                      + 12.0 * std::log(-t_plus) 
                                      - 24.0 * delta * std::log(-t_plus) 
                                      + 12.0 * std::pow(delta, 2) * std::log(-t_plus)
                                      )
                                      / ( 3.0 * std::pow( delta - 1.0 , 2)
                                              * std::pow(t_plus, 4) );
            } else {
                dilog_combi = utils::dilog( 1.0 + 1.0 / ( t_plus - 2.0 ) ) 
                         - utils::dilog( ( -1.0 + t_minus ) / ( -2.0 + t_plus ) )
                         + utils::dilog( 1.0 + ( 1.0 + t_minus - t_plus ) / t_plus ) 
                         - utils::dilog( 1.0 + 1.0 / t_plus );
            }

            j_tu = ( std::pow(g, 4) 
                     / ( 32.0 * M_PI * std::pow(m_phi, 4) 
                         * ( 1.0 + t_minus ) * t_plus ) )
                   * ( 2.0 * ( 2.0 * ( 1.0 + t_minus ) * ( t_minus - t_plus ) 
                               - 2.0 * ( 1.0 + t_minus ) * t_plus 
                                 * std::atanh( 1.0 / ( 1.0 - t_plus ) ) 
                                 * std::atanh( ( t_minus - t_plus ) 
                                               / ( -2.0 + t_minus + t_plus ) ) 
                               + t_minus * t_plus 
                                 * ( -std::log1p(-t_minus) + std::log1p(-t_plus) ) 
                               + ( 1.0 + t_minus ) 
                                 * ( std::log1p(-t_minus) - std::log1p(-t_plus) 
                                     - std::log1p( t_minus - t_plus ) ) 
                               + t_plus * ( -std::log1p(-t_minus) 
                                            + std::log1p(-t_plus) 
                                            + std::log1p( t_minus - t_plus ) ) 
                               - t_minus * t_plus * std::log( t_minus / t_plus ) ) 
                        + ( 1.0 + t_minus ) * t_plus 
                          * ( ( -std::pow( std::log1p(-t_minus), 2) 
                                + std::pow( std::log1p(-t_plus), 2) ) 
                              / 2.0 
                              + utils::dilog( 1.0 / ( 1.0 - t_plus ) )
                              - utils::dilog( 1.0 / ( 1.0 - t_minus ) ) 
                              - ( 1.0 + t_minus ) * t_plus 
                                * ( utils::dilog( 1.0 + t_minus )
                                    - utils::dilog( 1.0 + t_plus ) )
                              + dilog_combi
                            ) 
                     );
            if(j_tu < 0){ // Roundoff errors! Compute the integral numerically
                double a_y = t_plus, b_y = t_minus, a_x[3], b_x[3];
                // Nodes at which the integrand will be evaluated
                double y[3], x[3][3], F[3][3];
                j_tu = 0;
                for(int i=0; i<3; ++i){
                    y[i] = (b_y-a_y)/2. * utils::x_integ[i] + (b_y+a_y)/2.;
                    a_x[i] = -y[i];
                    b_x[i] = -t_plus;
                    for(int j=0; j<3; ++j){
                        x[i][j] = (b_x[i]-a_x[i])/2. * utils::x_integ[j] + (b_x[i]+a_x[i])/2.;
                        F[i][j] = 2*y[i]*(-y[i]-x[i][j])/std::pow(x[i][j], 2) / ((y[i]-1)*(-y[i]-x[i][j]-1));
                        j_tu += 1./4. * (b_y - a_y) * (b_x[i]-a_x[i]) * utils::w_integ[i] * utils::w_integ[j] * F[i][j];
                    }
                }
                j_tu *= std::pow(g, 4)/(16*M_PI * std::pow(m_phi, 4));
            }

        } else {
            j_tu = 0;
        }
        j_tu *= constants::PMNS_sq[flavour][k];
        return ( std::pow(m_phi, 4) / ( 2.0 * m_k ) ) * j_tu;

    }

    double J_st(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width , double m_k, 
               double t_plus, double t_minus) {
        double gamma_m = decay_width / m_phi; 

        using namespace std::complex_literals;

        std::complex<double> z1 = ( -1i * ( -1.0 + t_minus ) ) 
                                  / ( 2i + gamma_m );
        std::complex<double> z2 = 1.0 / ( 1.0 + t_minus );
        std::complex<double> z3 = 1.0 / ( 2.0 - 1i * gamma_m + t_minus );
        std::complex<double> z4 = ( 1.0 + t_minus - t_plus ) 
                                  / ( 2.0 - 1i * gamma_m + t_minus );
        std::complex<double> z5 = ( -1i * ( -1.0 + t_plus ) ) 
                                  / ( 2i + gamma_m );
        std::complex<double> z6 = 1.0 - t_plus / ( 1.0 + t_minus );
        std::complex<double> z7 = 1.0 - t_minus;
        std::complex<double> z8 = 1.0 - t_plus;

        std::complex<double> c_t_minus {t_minus, 0.0};
        std::complex<double> c_t_plus {t_plus, 0.0};
        // Compute dilogarithms
        std::complex<double> dilogdiff_z7z8;
        std::complex<double> dilogdiff_z5z1;
        std::complex<double> dilogdiff_z2z6;
        std::complex<double> dilogdiff_z4z3;

        if (-t_plus < 1e-5){ // We Taylor-expand dilogdiff to be fast and avoid roundoff errors
            double delta = t_plus / t_minus;

            dilogdiff_z7z8 = t_minus * ( -1.0 + std::log(c_t_minus) ) 
                + ( std::pow(t_minus, 2) * ( -1.0 + 2.0 * std::log(c_t_minus) ) ) 
                  / 4.0 
                - ( t_plus * ( -1.0 + std::log(c_t_plus) ) 
                    + std::pow(t_plus, 2) * ( -1.0 + 2.0 * std::log(c_t_plus) ) 
                      / 4.0);

            dilogdiff_z5z1 = 
                ( -t_minus + t_plus ) * std::log( 1.0 - 1i / ( 2i + gamma_m ) ) 
                + ( -std::pow(t_minus, 2) + std::pow(t_plus, 2) ) 
                  * ( 1i / ( 1i + gamma_m ) 
                      + std::log( 1.0 - 1i / ( 2i + gamma_m ) ) ) 
                  / 2.0;

            dilogdiff_z2z6 = 
                ( t_plus * ( -1.0 + delta - std::log(delta) + std::log(c_t_plus) 
                             - delta * std::log(c_t_plus) ) ) 
                / delta 
                + ( std::pow(t_plus, 2) * ( -1.0 + std::pow(delta, 2) 
                            + 2.0 * std::log(delta) - 2.0 * std::log(c_t_plus) 
                            + 4.0 * delta * std::log(c_t_plus) 
                            - 2.0 * std::pow(delta, 2) * std::log(c_t_plus) ) )
                  / ( 4.0 * std::pow(delta, 2) ) 
                + ( std::pow(t_plus, 3) 
                    * ( 7.0 - 9.0 * delta + 2.0 * std::pow(delta, 3) 
                        - 6.0 * std::log(delta) + 6.0 * std::log(c_t_plus) 
                        - 18.0 * delta * std::log(c_t_plus) 
                        + 18.0 * std::pow(delta, 2) * std::log(c_t_plus) 
                        - 6.0 * std::pow(delta, 3) * std::log(c_t_plus) ) )
                  / ( 18.0 * std::pow(delta, 3) );

            dilogdiff_z4z3 = 
                ( ( -1.0 + delta ) * t_plus 
                  * std::log( ( 1i + gamma_m ) / ( 2i + gamma_m ) ) )
                / delta 
                + ( ( -1.0 + delta ) * std::pow(t_plus, 2) 
                    * ( 1i * ( ( 1.0 + delta ) / ( 1i + gamma_m ) 
                              - 2.0 / ( 2i + gamma_m ) ) 
                      + ( -1.0 + delta ) 
                        * std::log( ( 1i + gamma_m ) / ( 2i + gamma_m ) ) ) ) 
                  / ( 2.0 * std::pow(delta, 2) );
        } 
        else {
            dilogdiff_z7z8 = utils::dilogdiff(z7, z8);

            dilogdiff_z5z1 = utils::dilogdiff(z5, z1);

            dilogdiff_z2z6 = utils::dilogdiff(z2, z6);

            dilogdiff_z4z3 = utils::dilogdiff(z4, z3);
        }

        double j_st;
        if(majorana) {
            j_st =  2.0 * M_PI * std::arg( -1.0 + 1i * gamma_m - t_minus ) - 2.0 * M_PI * std::arg( -1.0 + 1i * gamma_m - t_plus );

            j_st += 2.0 * gamma_m 
                    * ( std::imag(dilogdiff_z5z1) + std::imag(dilogdiff_z2z6) 
                        + std::imag(dilogdiff_z4z3 ) );

            j_st -= 2.0 * ( std::real(dilogdiff_z5z1) 
                            + std::real(dilogdiff_z2z6) 
                            + std::real(dilogdiff_z4z3) 
                            + std::real(dilogdiff_z7z8) );

            j_st -= std::arg( ( gamma_m + 1i * ( 1.0 + t_minus ) ) 
                              / ( 2i + gamma_m ) ) 
                    * ( 2.0 * M_PI + 2.0 * gamma_m * std::log1p(-t_minus) );
            
            j_st += std::arg( ( gamma_m + 1i * ( 1.0 + t_plus ) ) 
                              / ( 2i + gamma_m ) ) 
                    * ( 2.0 * M_PI + 2.0 * gamma_m * std::log1p(-t_plus) );
            
            j_st += ( std::arg( -1.0 + 1i * gamma_m - t_minus ) 
                      - std::arg( -1.0 + 1i * gamma_m - t_plus ) ) 
                    * ( 4.0 * gamma_m * t_minus 
                        + 2.0 * gamma_m * std::log1p(-t_minus) );

            j_st += 2.0 * gamma_m 
                    * ( std::arg( 1.0 + t_minus ) 
                        - std::arg( 2.0 - 1i * gamma_m + t_minus ) 
                        + std::arg( 1.0 - 1i * gamma_m + t_plus ) ) 
                    * std::log1p( t_minus - t_plus );
            
            j_st += std::log( 4.0 + std::pow(gamma_m, 2 ) ) 
                    * ( std::log1p(-t_plus) - std::log1p(-t_minus) );

            j_st += std::log( std::pow(gamma_m, 2) 
                              + std::pow( 2.0 + t_minus , 2) ) 
                    * std::log1p( t_minus - t_plus );

            j_st -= 2.0 * std::log1p(-t_minus) * std::log(-t_plus);

            j_st -= 2.0 * gamma_m * M_PI 
                      * ( std::log( std::pow(t_plus, 2) ) 
                          + std::log1p( t_minus - t_plus ) );

            j_st += 2.0 * gamma_m * M_PI * std::log( std::pow(t_plus, 2) );

            j_st += 4.0 * t_minus * std::log( t_minus / t_plus ); 

            j_st += ( -std::log1p(-t_plus) + std::log1p(-t_minus) 
                      - std::log1p( t_minus - t_plus ) ) 
                    * ( std::log1p( std::pow( 1 + t_plus , 2) 
                                    / std::pow(gamma_m, 2) ) 
                        + 2.0 * std::log(gamma_m) ); 

            j_st -= std::log1p( t_minus - t_plus ) 
                    * std::log1p( std::pow(t_minus, 2) + 2.0 * t_minus );

            j_st += 2.0 * ( std::pow(gamma_m, 2) + t_minus ) 
                    * ( std::log1p( std::pow( 1 + t_plus , 2) 
                                    / std::pow(gamma_m, 2) ) 
                    - std::log1p( std::pow( 1 + t_minus , 2) 
                                  / std::pow(gamma_m, 2) ) );

            j_st += 2.0 * ( std::log(-t_plus) 
                            * ( std::log1p(-t_plus) 
                                + std::log1p( t_minus - t_plus ) ) 
                            + ( std::log1p( std::pow( 1 + t_plus , 2) 
                                            / std::pow(gamma_m, 2) ) 
                            - std::log1p( std::pow( 1 + t_minus , 2) 
                                          / std::pow(gamma_m, 2) ) ) );

            j_st *= std::pow(g, 4); 

            j_st /= ( 32.0 * M_PI * ( 1.0 + std::pow(gamma_m, 2) ) 
                      * std::pow(m_phi, 4) ); 
            
        }
        else {
            j_st = gamma_m * std::imag(dilogdiff_z5z1);

            j_st -= 2.0 * ( std::real( dilogdiff_z5z1 + dilogdiff_z7z8 ) );

            j_st += 2.0 * std::arg( ( gamma_m + 1i * ( 1.0 + t_minus ) ) 
                                    / ( 2i + gamma_m ) ) 
                    * ( -M_PI - gamma_m * std::log1p(-t_minus) );
            
            j_st += 2.0 * std::arg( -1.0 + 1i * gamma_m - t_minus ) 
                    * ( M_PI + gamma_m * t_minus 
                        + gamma_m * std::log1p(-t_minus) );

            j_st -= 2.0 * std::arg( -1.0 + 1i * gamma_m - t_plus ) 
                    * ( M_PI + gamma_m * t_minus 
                        + gamma_m * std::log1p(-t_minus) );
                    
            j_st += 2.0 * std::arg( ( gamma_m + 1i * ( 1.0 + t_plus ) ) 
                                    / ( 2i + gamma_m ) ) 
                    * ( M_PI + gamma_m * std::log1p(-t_plus) );

            j_st -= 2.0 * std::log1p(-t_minus) * std::log(-t_plus);

            j_st += 2.0 * t_minus * std::log( t_minus / t_plus );

            j_st += 2.0 * std::log1p(-t_plus) * std::log(-t_plus);

            j_st += ( std::log1p(-t_plus) - std::log1p(-t_minus) ) 
                    * ( std::log( 4.0 + std::pow(gamma_m, 2) ) 
                        - 2.0 * std::log(gamma_m) 
                        - std::log1p( std::pow( 1.0 + t_plus , 2) 
                                      / std::pow(gamma_m, 2) ) );

            j_st += ( 1.0 + t_minus + std::pow(gamma_m, 2) ) 
                    * ( std::log1p( std::pow( 1 + t_plus , 2) 
                                    / std::pow(gamma_m, 2) ) 
                        - std::log1p( std::pow( 1 + t_minus , 2) 
                                    / std::pow(gamma_m, 2) ) );

            j_st *= std::pow(g, 4); 
            j_st /= ( 32.0 * M_PI * ( 1.0 + std::pow(gamma_m, 2) ) 
                      * std::pow(m_phi, 4) );
        }

        if(j_st < 0.0){
            j_st = 0.0;
        }

        j_st *= constants::PMNS_sq[flavour][k];
        return ( std::pow(m_phi, 4) / ( 2.0 * m_k ) ) * j_st;
    }

    double J_su(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width , double m_k, 
               double t_plus, double t_minus) {

        if(majorana) {
            return J_st(k, flavour, g, m_phi, majorana, decay_width, m_k, 
                        t_plus, t_minus);
        }
        else {
            return 0.0;
        }
    }

    double J_s(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width , double m_k, 
               double t_plus, double t_minus, double s_plus, double s_minus) {
        double j_s;
        double gamma_m = decay_width / m_phi;
        if ( s_plus < 1e-5 ) { // We Taylor-expand atandiff to avoid roundoff errors
            j_s = std::pow(g, 4) / ( 8.0 * M_PI * decay_width * std::pow(m_phi, 3) )
                  * ( t_minus - t_plus ) 
                  * ( gamma_m * ( s_plus - s_minus ) 
                      * ( 1.0 + std::pow(gamma_m, 2) + 2.0 * s_minus) 
                      / std::pow(  1.0  + std::pow(gamma_m, 2), 2) 
                    + gamma_m * std::pow( s_plus - s_minus , 2) 
                      / std::pow( 1.0 + std::pow(gamma_m , 2), 2)  );
        }
        else {
            j_s = std::pow(g, 4) / ( 8.0 * M_PI * decay_width * std::pow(m_phi, 3) )
                * ( t_minus - t_plus ) 
                * utils::atan_diff( m_phi * ( s_plus - 1.0 ) / decay_width,
                        m_phi * ( s_minus - 1.0 ) / decay_width) ;
        }
        j_s *= constants::PMNS_sq[flavour][k];
        if(!majorana)
            j_s /= 2.0; // For Dirac, one of the final neutrinos is not observable

        return std::pow(m_phi, 4) / ( 2.0 * m_k ) * j_s;

    }

    double J_t(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width , double m_k, 
               double t_plus, double t_minus, double s_plus, double s_minus) {
        double j_t;
        if (majorana){
            j_t =  -( ( s_minus - s_plus ) 
                      * ( 3.0 + 2.0 * t_minus * ( -1.0 + t_plus ) - 2.0 * t_plus ) 
                      * ( t_minus - t_plus ) ) 
                    / ( ( -1.0 + t_minus ) * ( -1.0 + t_plus ) ) ;

            j_t += ( 2.0 / ( ( 1.0 + t_minus ) * ( 1.0 + t_plus ) ) )
                    * ( s_minus * s_plus * ( -t_minus + t_plus ) 
                        * std::log(s_minus) 
                       + s_minus * s_plus * ( t_minus - t_plus ) 
                         * std::log(s_plus) 
                       - s_minus * s_plus 
                         * std::log1p( s_minus + t_minus ) 
                       - s_minus * s_plus * t_plus 
                         * std::log1p( s_minus + t_minus ) 
                       + s_minus * s_plus 
                         * std::log1p( s_plus + t_minus ) 
                       + s_minus * s_plus * t_plus 
                         * std::log1p( s_plus + t_minus ) 
                       - s_plus 
                         * std::log( ( 1.0 + s_minus + t_minus ) 
                                       * ( -1.0 + t_plus ) 
                                     / ( ( -1.0 + t_minus ) 
                                         * ( 1.0 + s_minus + t_plus ) ) ) 
                       - s_plus * t_minus 
                         * std::log( ( ( 1.0 + s_minus + t_minus ) 
                                       * ( -1.0 + t_plus ) ) 
                                     / ( ( -1.0 + t_minus ) 
                                         * ( 1.0 + s_minus + t_plus ) ) ) 
                       - s_plus * t_plus 
                         * std::log( ( ( 1.0 + s_minus + t_minus ) 
                                       * ( -1.0 + t_plus ) ) 
                                     / ( ( -1.0 + t_minus ) 
                                         * ( 1.0 + s_minus + t_plus ) ) ) 
                       - s_plus * t_minus * t_plus 
                         * std::log( ( ( 1.0 + s_minus + t_minus ) 
                                       * ( -1.0 + t_plus ) ) 
                                     / ( ( -1.0 + t_minus ) 
                                         * ( 1.0 + s_minus + t_plus ) ) ) 
                       + s_minus * s_plus 
                         * std::log( 1.0 + s_minus + t_plus ) 
                       + s_minus * s_plus * t_minus 
                         * std::log1p( s_minus + t_plus ) 
                       + s_minus 
                         * std::log( ( ( 1.0 + s_plus + t_minus ) 
                                       * (-1 + t_plus ) )
                                       / ( ( -1.0 + t_minus ) 
                                         * ( 1.0 + s_plus + t_plus ) ) ) 
                       + s_minus * t_minus 
                         * std::log( ( ( 1.0 + s_plus + t_minus ) 
                                       * ( -1.0 + t_plus ) )
                                     / ( ( -1.0 + t_minus ) 
                                         * ( 1.0 + s_plus + t_plus ) ) ) 
                       + s_minus * t_plus 
                         * std::log( ( ( 1.0 + s_plus + t_minus ) 
                                       * ( -1.0 + t_plus ) )
                                     / ( ( -1.0 + t_minus ) 
                                         * ( 1.0 + s_plus + t_plus ) ) ) 
                       + s_minus * t_minus * t_plus 
                         * std::log( ( ( 1.0 + s_plus + t_minus ) 
                                       * ( -1.0 + t_plus ) )
                                     / ( ( -1.0 + t_minus )
                                         *( 1.0 + s_plus + t_plus ) ) ) 
                       - s_minus * s_plus 
                         * std::log( 1.0 + s_plus + t_plus ) 
                       - s_minus * s_plus * t_minus 
                         * std::log1p( s_plus + t_plus ) )
                        / ( ( 1.0 + t_minus ) * ( 1.0 + t_plus ) ); 

            j_t -= ( ( s_minus * s_plus 
                       * std::log( ( s_minus 
                                     * ( 1.0 + s_plus + t_minus ) )
                                   / ( s_plus 
                                       * ( 1.0 + s_minus + t_minus ) ) ) ) 
                     / std::pow( 1.0 + t_minus , 2) 
                     + ( ( s_minus - s_plus ) * ( t_minus - t_plus )
                         * ( 1.0 + t_plus ) / ( 1.0 + t_minus ) 
                         - s_minus * s_plus 
                           * std::log( s_minus 
                                       * ( 1.0 + s_plus + t_plus ) 
                                       / ( s_plus 
                                           * ( 1.0 + s_minus + t_plus ) ) ) ) 
                       / std::pow( 1.0 + t_plus , 2) );

            j_t *= std::pow(g, 4) 
                   / ( s_minus * s_plus * 16.0 * M_PI 
                       * std::pow(m_phi, 4) );
            if( j_t < 0 || std::isnan(j_t) ) { // Roundoff errors! Compute the integral numerically
                double a_y = t_plus, b_y = t_minus, a_x = s_minus, b_x = s_plus;
                // Nodes at which the integrand will be evaluated
                double y[3], x[3], F[3][3];
                j_t = 0;
                for(int i=0; i<3; ++i)
                    for(int j=0; j<3; ++j){
                        y[i] = (b_y-a_y)/2. * utils::x_integ[i] + (b_y+a_y)/2.;
                        x[j] = (b_x-a_x)/2. * utils::x_integ[j] + (b_x+a_x)/2.;
                        F[i][j] = std::pow(y[i]/x[j], 2) / std::pow(y[i]-1, 2) +
                                std::pow((-x[j]-y[i])/x[j], 2) / std::pow((-x[j]-y[i])-1, 2);
                        j_t += utils::w_integ[i] * utils::w_integ[j] * F[i][j];
                    }
                j_t *= 1./4. * (b_y - a_y) * (b_x - a_x);

                j_t *= std::pow(g, 4)/(16*M_PI * std::pow(m_phi, 4));
            }

        }
        else{
            j_t = ( 3.0 / 2.0 ) * std::pow(g, 4) 
                  / ( 32.0 * M_PI * std::pow(m_phi, 4) 
                      * s_minus * s_plus * ( -1.0 + t_minus ) 
                      * ( -1.0 + t_plus ) ) 
                  * ( s_minus - s_plus ) 
                  * ( -( ( t_minus - t_plus ) 
                       * ( 2.0 + t_minus * ( -1.0 + t_plus ) - t_plus ) )
                      - 2.0 * ( -1.0 + t_minus ) * ( -1.0 + t_plus ) 
                        * ( std::log1p(-t_minus) - std::log1p(-t_plus) ) );
            if(j_t < 0){ // Roundoff errors! Compute the integral numerically
                double a_y = t_plus, b_y = t_minus, a_x = s_minus, b_x = s_plus;
                // Nodes at which the integrand will be evaluated
                double y[3], x[3], F[3][3];
                j_t = 0;
                for(int i=0; i<3; ++i)
                    for(int j=0; j<3; ++j){
                        y[i] = (b_y-a_y)/2. * utils::x_integ[i] + (b_y+a_y)/2.;
                        x[j] = (b_x-a_x)/2. * utils::x_integ[j] + (b_x+a_x)/2.;
                        F[i][j] = std::pow(y[i]/x[j], 2) / std::pow(y[i]-1, 2);
                        j_t += utils::w_integ[i] * utils::w_integ[j] * F[i][j];
                    }
                j_t *= 1./4. * (b_y - a_y) * (b_x - a_x);

                j_t *= 3./2. * std::pow(g, 4)/(32*M_PI * std::pow(m_phi, 4));
            }
        }

        j_t *= constants::PMNS_sq[flavour][k];

        return std::pow(m_phi, 4) / ( 2.0 * m_k ) * j_t;

    }

    double J_u(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width , double m_k, 
               double t_plus, double t_minus, double s_plus, double s_minus){
        if (majorana) {
            return J_t(k, flavour, g, m_phi, majorana, decay_width, m_k, 
                       t_plus, t_minus, s_plus, s_minus);
        }
        else {
            double j_u = std::pow(g, 4) 
                  / ( 64.0 * M_PI * std::pow(m_phi, 4) * s_minus 
                      * s_plus * ( -1.0 + t_minus ) * ( -1.0 + t_plus ) ) 
                  * ( s_minus - s_plus ) 
                  * ( -( ( t_minus - t_plus ) 
                         * ( 2.0 + t_minus * ( -1.0 + t_plus ) - t_plus ) )
                      - 2.0 * ( -1.0 + t_minus ) * ( -1 + t_plus ) 
                        * ( std::log1p(-t_minus) - std::log1p(-t_plus) ) );

            j_u *= constants::PMNS_sq[flavour][k];
            if( j_u < 0 || std::isnan(j_u) ) { // Roundoff errors! Compute the integral numerically
                double a_y = t_plus, b_y = t_minus, a_x = s_minus, b_x = s_plus;
                // Nodes at which the integrand will be evaluated
                double y[3], x[3], F[3][3];
                j_u = 0;
                for(int i=0; i<3; ++i)
                    for(int j=0; j<3; ++j){
                        y[i] = (b_y-a_y)/2. * utils::x_integ[i] + (b_y+a_y)/2.;
                        x[j] = (b_x-a_x)/2. * utils::x_integ[j] + (b_x+a_x)/2.;
                        F[i][j] = std::pow(y[i]/x[j], 2) / std::pow(y[i]-1, 2);
                        j_u += utils::w_integ[i] * utils::w_integ[j] * F[i][j];
                    }
                j_u *= 1./4. * (b_y - a_y) * (b_x - a_x);

                j_u *= 1./2. * std::pow(g, 4)/(32*M_PI * std::pow(m_phi, 4));
            }

            return std::pow(m_phi, 4) / ( 2.0 * m_k ) * j_u;
        }

    }

    double J_tu(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width , double m_k, 
               double t_plus, double t_minus, double s_plus, double s_minus){

        double j_tu;
        if (majorana) {
            double FCTR_t_plus;

            if (t_plus < -1) {
                FCTR_t_plus = utils::dilog( ( 1.0 + s_minus + t_plus ) 
                                           / s_minus ) 
                             - utils::dilog( ( 1.0 + s_plus + t_plus ) 
                                             / s_plus );
            }
            else {
                FCTR_t_plus = - utils::dilog( s_minus 
                                             / ( 1.0+ s_minus + t_plus ) ) 
                             + utils::dilog( s_plus 
                                             / ( 1.0 + s_plus + t_plus ) ) 
                             - 0.5 * ( std::pow( std::log1p( ( 1.0 + t_plus ) 
                                                             / s_minus )
                                                , 2) 
                                        - std::pow( std::log1p( ( 1.0 + t_plus ) 
                                                                / s_plus )
                                                   , 2) );
            }

            double FCTR_t_minus;
            if (t_minus < -1) {
                FCTR_t_minus = - utils::dilog( ( 1.0 + s_minus + t_minus ) 
                                              / s_minus ) 
                              + utils::dilog( ( 1.0 + s_plus + t_minus ) 
                                              / s_plus );
            }
            else {
                FCTR_t_minus = utils::dilog( s_minus 
                                            / ( 1.0 + s_minus + t_minus ) ) 
                              - utils::dilog( s_plus 
                                              / ( 1.0 + s_plus + t_minus ) ) 
                              + 0.5 * ( std::pow( std::log1p( ( 1.0 + t_minus ) 
                                                              / s_minus )
                                                 , 2) 
                                        - std::pow( std::log1p( ( 1.0 + t_minus )
                                                                / s_plus )
                                                   , 2) );
            }

            double log1p_abs_t_plus, log1p_abs_t_minus;
            if (t_plus > -1) {
                log1p_abs_t_plus = std::log1p(t_plus);
            }                                                                            
            else {
                log1p_abs_t_plus = std::log(-1-t_plus);
            }

            if (t_minus > -1) {
                log1p_abs_t_minus = std::log1p(t_minus);
            }
            else {
                log1p_abs_t_minus = std::log(-1-t_minus);
            }

            j_tu = -4.0 * ( s_minus - s_plus ) * ( 1.0 + t_minus ) 
                        * ( t_minus - t_plus ) * ( 1.0 + t_plus ); 
            
            j_tu += 2.0 * s_minus * s_plus * t_plus 
                    * ( std::log( s_minus / s_plus ) 
                        - std::log1p( s_minus + t_minus ) 
                        + std::log1p( s_plus + t_minus ) );

            j_tu += 2.0 * s_plus * ( 1.0 + t_minus ) * ( 1.0 + t_plus )
                    * ( std::log1p(-t_minus) - std::log1p( s_minus + t_minus ) 
                        - std::log1p(-t_plus) + std::log1p( s_minus + t_plus ) ); 

            j_tu -= 2.0 * s_minus * ( 1.0 + t_minus ) * ( 1.0 + t_plus )
                    * ( std::log1p(-t_minus) - std::log1p( s_plus + t_minus ) 
                        - std::log1p(-t_plus) + std::log1p( s_plus + t_plus ) ); 

            j_tu += 2.0 * s_minus * s_plus * ( -std::log1p( s_minus + t_minus ) 
                    + std::log1p( s_plus + t_minus ) 
                    + std::log1p( s_minus + t_plus ) 
                    - std::log1p( s_plus + t_plus ) ); 

            j_tu += s_minus * s_plus * ( 1.0 + t_minus ) * ( 1.0 + t_plus )
                    * ( std::log( ( 2.0 + s_minus ) / s_minus ) 
                        * ( std::log(s_plus) + std::log1p( s_minus + t_plus ) ) 
                        - std::log( ( 2.0 + s_plus ) / s_plus ) 
                          * ( std::log(s_minus) + std::log1p( s_plus + t_plus ) ) 
                        + std::log1p(-t_plus) 
                            * ( std::log( s_minus / s_plus ) 
                                - std::log1p( s_minus + t_plus ) 
                                + std::log1p( s_plus + t_plus ) ) ); 

            j_tu += s_minus * s_plus * ( 1.0 + t_minus ) * ( 1.0 + t_plus )
                    * ( ( std::log(s_plus) + std::log1p( s_minus + t_minus ) )
                        * ( std::log( s_minus / ( 2.0 + s_minus ) ) 
                                + std::log1p(-t_minus) - log1p_abs_t_minus ) 
                        + ( std::log(s_minus) + std::log1p( s_plus + t_minus ) )
                            * ( std::log( ( 2.0 + s_plus ) / s_plus ) 
                                - std::log1p(-t_minus) + log1p_abs_t_minus ) );

            j_tu += s_minus * s_plus 
                    * ( std::log( s_plus / s_minus ) 
                        + std::log1p( s_minus + t_plus ) 
                        - std::log1p( s_plus + t_plus ) ) 
                    * ( 2.0 * t_minus 
                        + ( 1.0 + t_minus ) * ( 1.0 + t_plus ) 
                        * log1p_abs_t_plus ); 
            
            j_tu += s_minus * s_plus * ( 1.0 + t_minus ) * ( 1.0 + t_plus )
                    * ( utils::dilog( ( 1.0 + s_minus + t_minus ) 
                                       / ( 2.0 + s_minus ) ) 
                        - utils::dilog( ( 1.0 + s_plus + t_minus ) 
                                       / ( 2.0 + s_plus ) ) 
                        - utils::dilog( ( 1.0 + s_minus + t_plus )
                                        / ( 2.0 + s_minus ) ) 
                        + utils::dilog( ( 1.0 + s_plus + t_plus ) 
                                        / ( 2.0 + s_plus ) ) ); 

            j_tu += s_minus * s_plus * ( 1.0 + t_minus ) * ( 1.0 + t_plus )
                    *( FCTR_t_plus + FCTR_t_minus );

            j_tu *= std::pow(g, 4) 
                    / ( 32.0 * M_PI * std::pow(m_phi, 4) * s_minus * s_plus 
                             * ( 1.0 + t_minus ) * ( 1.0 + t_plus ) );
            
        }
        else {
            j_tu = 0.0;
        }

        j_tu *= constants::PMNS_sq[flavour][k];

        return std::pow(m_phi, 4) / ( 2.0 * m_k ) * j_tu; 
    }

    double J_st(mass_state k, flavour_state flavour, double g, double m_phi, 
                bool majorana, double decay_width , double m_k, 
                double t_plus, double t_minus, double s_plus, double s_minus){

        double j_st;
        double gamma_m = decay_width/m_phi; 

        using namespace std::complex_literals;

        std::complex<double> z1 = (1.0 + s_minus + t_minus)/(1.0 + t_minus);
        std::complex<double> z2 = (1.0 + s_minus + t_minus)/(2.0 - 1i*gamma_m + t_minus);
        std::complex<double> z3 = (1.0 + s_plus + t_minus)/(1.0 + t_minus);
        std::complex<double> z4 = (1.0 + s_plus + t_minus)/(2.0 - 1i*gamma_m + t_minus);
        std::complex<double> z5 = (1.0 + s_minus + t_plus)/(1.0 + t_plus);
        std::complex<double> z6 = (1.0 + s_minus + t_plus)/(2.0 - 1i*gamma_m + t_plus);
        std::complex<double> z7 = (1.0 + s_plus + t_plus)/(1.0 + t_plus);
        std::complex<double> z8 = (1.0 + s_plus + t_plus)/(2.0 - 1i*gamma_m + t_plus);

        std::complex<double> dilog_z1 = utils::dilog(z1);
        std::complex<double> dilog_z2 = utils::dilog(z2);
        std::complex<double> dilog_z3 = utils::dilog(z3);
        std::complex<double> dilog_z4 = utils::dilog(z4);
        std::complex<double> dilog_z5 = utils::dilog(z5);
        std::complex<double> dilog_z6 = utils::dilog(z6);
        std::complex<double> dilog_z7 = utils::dilog(z7);
        std::complex<double> dilog_z8 = utils::dilog(z8);

        if(majorana){
                j_st = 2.0 * gamma_m 
                       * (  std::imag(dilog_z1) - std::imag(dilog_z2) 
                          - std::imag(dilog_z3) + std::imag(dilog_z4)
                          - std::imag(dilog_z5) + std::imag(dilog_z6) 
                          + std::imag(dilog_z7) - std::imag(dilog_z8) );

                j_st -= 2.0 * (  std::real(dilog_z1) - std::real(dilog_z2)
                               - std::real(dilog_z3) + std::real(dilog_z4)
                               - std::real(dilog_z5) + std::real(dilog_z6) 
                               + std::real(dilog_z7) - std::real(dilog_z8) );

                j_st += 2.0 * gamma_m * ( std::arg( -1.0 / ( 1.0 + t_minus ) ) 
                            - std::arg( ( 1.0 - 1i * gamma_m - s_minus )
                                / ( 2.0 - 1i * gamma_m + t_minus ) ) )
                        * std::log1p( s_minus + t_minus );
                
                j_st -= 2.0 * gamma_m * ( std::arg( -1.0 / ( 1.0 + t_minus ) ) 
                            - std::arg( ( 1.0 - 1i * gamma_m - s_plus )
                                        / ( 2.0 - 1i * gamma_m + t_minus ) ) )
                        * std::log1p( s_plus + t_minus );
                
                j_st += 2.0 * gamma_m * ( std::arg( -1.0 / ( 1.0 + t_plus ) ) 
                            - std::arg( ( 1.0 - 1i * gamma_m - s_plus)
                                        / ( 2.0 - 1i * gamma_m + t_plus ) ) )
                        * std::log1p( s_plus + t_plus );
                
                j_st -= 2.0 * gamma_m * ( std::arg( -1.0 / ( 1.0 + t_plus ) ) 
                            - std::arg( (1.0 - 1i * gamma_m - s_minus )
                                        / ( 2.0 - 1i * gamma_m + t_plus ) ) )
                        * std::log1p( s_minus + t_plus );

                j_st += 2.0 * ( gamma_m * std::arg( -1.0 + 1i * gamma_m + s_minus ) 
                        - gamma_m * std::arg( -1.0 + 1i * gamma_m + s_plus ) 
                        + std::log1p( std::pow( -1.0 + s_plus , 2) / std::pow(gamma_m, 2) ) / 2.0 
                        - std::log1p( std::pow( -1.0 + s_minus, 2) / std::pow(gamma_m, 2) ) / 2.0 
                        + std::log(s_minus) - std::log(s_plus) )
                        * ( 2.0 * ( t_minus - t_plus ) + ( std::log1p(-t_minus) - std::log1p(-t_plus) ) ); 

                j_st += std::log1p( s_minus + t_minus )
                        * ( std::log1p( std::pow( -1.0 + s_minus , 2) / std::pow(gamma_m, 2) ) 
                        - std::log1p( std::pow( 2.0 + t_minus , 2) / std::pow(gamma_m, 2) ) 
                        - 2.0 * ( std::log(s_minus) - std::log( std::fabs( 1.0 + t_minus ) ) ) );

                j_st -= std::log1p( s_plus + t_minus )
                    * ( std::log1p( std::pow( -1.0 + s_plus , 2) / std::pow(gamma_m, 2) ) 
                            - std::log1p( std::pow( 2.0 + t_minus , 2) / std::pow(gamma_m, 2) ) 
                            - 2.0 * ( std::log(s_plus) - std::log( std::fabs( 1.0 + t_minus ) ) ) );

                j_st -= std::log1p( s_minus + t_plus )
                        * ( std::log1p( std::pow( -1.0 + s_minus , 2) / std::pow(gamma_m, 2) ) 
                            - std::log1p( std::pow( 2.0 + t_plus , 2) / std::pow(gamma_m, 2) ) 
                            - 2.0 * ( std::log(s_minus) - std::log( std::fabs( 1.0 + t_plus ) ) ) );

                j_st += std::log1p( s_plus + t_plus )
                    * ( std::log1p( std::pow( -1.0 + s_plus , 2) / std::pow(gamma_m, 2) ) 
                            - std::log1p( std::pow( 2.0 + t_plus , 2) / std::pow(gamma_m, 2) ) 
                            - 2.0 * ( std::log(s_plus) - std::log( std::fabs( 1.0 + t_plus ) ) ) );

            j_st *= std::pow(g, 4) 
                / ( 32.0 * M_PI * ( 1.0 + std::pow(gamma_m, 2) ) 
                         * std::pow(m_phi, 4) ); 

        } else{
            j_st = ( 2.0 * gamma_m * std::arg( -1.0 + 1i * gamma_m + s_minus ) 
                      - 2.0 * gamma_m * std::arg( -1.0 + 1i * gamma_m + s_plus )
                      + 2.0 * std::log(s_minus) - 2.0 * std::log(s_plus)
                      + std::log1p( std::pow( -1.0 + s_plus , 2) / std::pow(gamma_m, 2) ) 
                      - std::log1p( std::pow( -1.0 + s_minus , 2) / std::pow(gamma_m, 2) ) )
                * ( t_minus - t_plus + std::log1p(-t_minus) - std::log1p(-t_plus) );

            j_st *= std::pow(g, 4) 
                / ( 32.0 * M_PI * ( 1.0 + std::pow(gamma_m, 2) ) 
                        * std::pow(m_phi, 4) );
        }

        j_st *= constants::PMNS_sq[flavour][k];

        return std::pow(m_phi, 4) / ( 2.0 * m_k ) * j_st;

    }

    double J_su(mass_state k, flavour_state flavour, double g, double m_phi, 
            bool majorana, double decay_width , double m_k, 
            double t_plus, double t_minus, double s_plus, double s_minus){
        if (majorana) {
            return J_st(k, flavour, g, m_phi, majorana, decay_width, m_k, 
                    t_plus, t_minus);
        }
        else {
            return 0.0;
        }
    }

} //SI
