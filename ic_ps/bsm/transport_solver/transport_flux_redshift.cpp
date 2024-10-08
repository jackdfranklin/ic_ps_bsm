#include "transport_flux_redshift.hpp"

#include <chrono>
#include <iostream>

#include <Eigen/Dense>

#include "utils.hpp"
#include "constants.hpp"

void apply_redshift(const Eigen::VectorXd &E_GeV, Eigen::VectorXd &flux, 
                    double z_0, double z_1) {

    double deltalog10E = std::log10(E_GeV.head(2)(1)) 
                        - std::log10(E_GeV.head(1)(0));

    for( auto m = 0; m < E_GeV.size(); ++m ) {
        double E_upper_m = std::pow(10, std::log10(E_GeV(m)) + 0.5 * deltalog10E); 
        double E_lower_m = std::pow(10, std::log10(E_GeV(m)) - 0.5 * deltalog10E); 

        double E_upper_z = ( 1.0 + z_0 ) / ( 1.0 + z_1 ) * E_upper_m;
        double E_lower_z = ( 1.0 + z_0 ) / ( 1.0 + z_1 ) * E_lower_m;
        double redshifted_flux = 0.0;

        for ( auto n = m; n < E_GeV.size(); ++n ) {

            double E_upper_n = std::pow(10, std::log10(E_GeV(n)) + 0.5 * deltalog10E); 
            double E_lower_n = std::pow(10, std::log10(E_GeV(n)) - 0.5 * deltalog10E); 

            if ( E_lower_n <= E_lower_z && E_upper_z <= E_upper_n ) {

                redshifted_flux += ( E_upper_z - E_lower_z ) 
                                    * flux(n) / ( E_upper_n - E_lower_n );
                
            } else
            if (   E_lower_n <= E_lower_z 
                && E_lower_z < E_upper_n 
                && E_upper_n <= E_upper_z ) {
                
                redshifted_flux += ( E_upper_n - E_lower_z ) 
                                    * flux(n) / ( E_upper_n - E_lower_n );
                
            } else 
            if (   E_lower_z <= E_lower_n 
                && E_lower_n <= E_upper_z 
                && E_upper_z <= E_upper_n ) {

                redshifted_flux += ( E_upper_z - E_lower_n ) 
                                    * flux(n) / ( E_upper_n - E_lower_n );
                
            } else
            if ( E_lower_z <= E_lower_n && E_upper_n <= E_upper_z ) {

                redshifted_flux += flux(n);


            }

        }

        flux(m) = redshifted_flux;

    }

}

void shift_bins(const Eigen::VectorXd &E_GeV, Eigen::VectorXd &flux) {

    for( auto m = 0; m < E_GeV.size() - 1; ++m ) {
        flux(m) = flux( m + 1 );
    }

    flux(E_GeV.size() - 1) = 0.0;

}

void solve_transport_eqn_z(
        const Eigen::VectorXd &energy_nodes, 
        const Eigen::VectorXd &deltaE_GeV,
        std::array<Eigen::VectorXd, 3> &fluxes, 
        const std::array<Eigen::VectorXd, 3> &Ks,
        const std::array< std::array<Eigen::MatrixXd, 3>, 3> &Is,
        double zmax,
        double relic_density_cm
        ){

    double deltalog10E = std::log10(energy_nodes.head(2)(1)) 
        - std::log10(energy_nodes.head(1)(0));

    auto N_nodes = energy_nodes.size();

    int N_z = static_cast<int>( std::log10( 1.0 + zmax ) / deltalog10E ) + 1;

    std::vector<double> zs(N_z);
    zs.at(0) = zmax;
    for(int i = 1; i < N_z; ++i){
        zs.at(i) = (1.0 + zs.at(i - 1) ) / std::pow(10, deltalog10E) - 1.0; 
    }

    for( int s = 1; s <= N_z; ++s ) {

        double z, delta_z;
        if(s == N_z) {
            z = 0.0;
            delta_z = zs.at( s - 1 );
            for ( mass_state i: {one, two, three} ) {
                apply_redshift(energy_nodes, fluxes.at(i), z, zs.at(s - 1));
            }
        }
        else {
            z = zs.at(s);
            delta_z = zs.at( s - 1 ) - z;
            for ( mass_state i: {one, two, three} ) {
                shift_bins(energy_nodes, fluxes.at(i));
            }
        }

        double H_GeV =  1.5e-42 
            * std::sqrt( 0.692 + 0.308 * std::pow( 1.0 + z , 3) );
        double n_z = relic_density_cm * std::pow( 1.0 + z , 3) / utils::cm3_to_GeV3;

        for( auto m = N_nodes - 1; m >= 0; --m ) {

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

                M(i, i) += ( H_GeV * ( 1.0 + z ) / delta_z ); 
                // Loss term from cross section
                M(i, i) += n_z * Ks.at(i)(m) / deltaE_m;

                // Gain term from up/down scattering
                for ( mass_state j: {one, two, three} ) {
                    M(j, i) -= n_z * Is[j][i](m, m) / deltaE_m;
                }

                rhs(i) += ( H_GeV * ( 1.0 + z ) / delta_z ) 
                            * fluxes.at(i)(m);
                
                if ( m != N_nodes - 1 ) {
                    double E_m_plus_1 = energy_nodes(m + 1);

                    for ( mass_state j: {one, two, three} ) {
                        
                        // Gain term from up/down scattering
                        rhs(i) += n_z 
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
}

Eigen::MatrixXd transport_flux_z(Eigen::VectorXd energy_nodes, 
                                 Eigen::VectorXd gamma_grid, 
                                 double zmax, 
                                 std::array<double,3> neutrino_masses_GeV, 
                                 double relic_density_cm, 
                                 const Loss_term &K, const Gain_term &I 
                                 ) {

    const auto start{std::chrono::steady_clock::now()};


    Eigen::VectorXd deltaE_GeV = energy_bin_widths(energy_nodes);
    double deltalog10E = std::log10(energy_nodes.head(2)(1)) 
        - std::log10(energy_nodes.head(1)(0));


    std::array< std::array<Eigen::MatrixXd, 3>, 3> Is;
    Is.at(one).at(one) =  I(one, one, energy_nodes, 
                            neutrino_masses_GeV)
                          / utils::GeV2_to_cm2;
    Is.at(two).at(one) =  I( two, one, energy_nodes, 
                            neutrino_masses_GeV)
                            / utils::GeV2_to_cm2;
    Is.at(three).at(one) =  I(three, one, energy_nodes, 
                            neutrino_masses_GeV)
                            / utils::GeV2_to_cm2;

    Is.at(one).at(two) =  I(one, two, energy_nodes, 
            neutrino_masses_GeV)
                            / utils::GeV2_to_cm2;
    Is.at(two).at(two) =  I(two, two, energy_nodes, 
            neutrino_masses_GeV)
                            / utils::GeV2_to_cm2;
    Is.at(three).at(two) =  I(three, two, energy_nodes, 
            neutrino_masses_GeV)
                            / utils::GeV2_to_cm2;

    Is.at(one).at(three) =  I(one, three, energy_nodes, 
            neutrino_masses_GeV)
                            / utils::GeV2_to_cm2;
    Is.at(two).at(three) =  I(two, three, energy_nodes, 
            neutrino_masses_GeV)
                            / utils::GeV2_to_cm2;
    Is.at(three).at(three) =  I(three, three, energy_nodes, 
            neutrino_masses_GeV)
                            / utils::GeV2_to_cm2;

    std::array<Eigen::VectorXd, 3> Ks;
    Ks.at(one) = K(one,   energy_nodes, neutrino_masses_GeV)
                            / utils::GeV2_to_cm2;
    Ks.at(two) = K(two,   energy_nodes, neutrino_masses_GeV)
                            / utils::GeV2_to_cm2;
    Ks.at(three) = K(three, energy_nodes, neutrino_masses_GeV)
                            / utils::GeV2_to_cm2;

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};

    std::cout<<"Initialisation complete after "<<elapsed_seconds<<" s"<<std::endl;

    double norm = 1.0/3.0;
    std::array<Eigen::VectorXd, 3> current_flux;

    Eigen::MatrixXd final_muon_fluxes(energy_nodes.size(), gamma_grid.size());

    for(int i = 0; i < gamma_grid.size(); ++i) {

        double gamma = gamma_grid(i);

        current_flux.at(one) = norm 
            * ( 2.0 * constants::PMNS_sq[mu][one] + constants::PMNS_sq[e][one])
            * initial_flux(energy_nodes, 1000.0, gamma).array() * deltaE_GeV.array();

        current_flux.at(two) = norm 
            * ( 2.0 * constants::PMNS_sq[mu][two] 
                    + constants::PMNS_sq[e][two])
            * initial_flux(energy_nodes, 1000.0, gamma).array() * deltaE_GeV.array();

        current_flux.at(three) = norm 
            * ( 2.0 * constants::PMNS_sq[mu][three] 
                    + constants::PMNS_sq[e][three])
            * initial_flux(energy_nodes, 1000.0, gamma).array() * deltaE_GeV.array();

        solve_transport_eqn_z(energy_nodes, deltaE_GeV, current_flux, Ks, Is, zmax, relic_density_cm); 

        final_muon_fluxes.col(i) = 
            (   constants::PMNS_sq[mu][one]   * current_flux.at(one)  
              + constants::PMNS_sq[mu][three] * current_flux.at(two) 
              + constants::PMNS_sq[mu][two]   * current_flux.at(three) ).array()
            / deltaE_GeV.array();

    }

    return final_muon_fluxes;

}

