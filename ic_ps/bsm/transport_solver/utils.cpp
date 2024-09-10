#include "utils.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <stdio.h>

#include <Eigen/Dense>

#include "constants.hpp"

Eigen::VectorXd initial_flux(const Eigen::VectorXd &E_GeV, 
                             double E0, 
                             double gamma){

    Eigen::VectorXd result(E_GeV.size());

    double deltalog10E = ( std::log(E_GeV.tail(1)(0)) 
                            - std::log(E_GeV.head(1)(0)) 
                         ) / E_GeV.size();

    for(size_t index = 0; index < E_GeV.size(); index++){

        double log10E = std::log10(E_GeV(index));

        double E_plus = std::pow( 10.0, log10E + 0.5 * deltalog10E );
        double E_minus = std::pow( 10.0, log10E - 0.5 * deltalog10E );

        result(index) = ( std::pow(E0, gamma) 
                          * ( std::pow(E_plus, 1.0 - gamma ) 
                              - std::pow( E_minus, 1.0 - gamma ) ) 
                            / ( 1.0 - gamma ) ) 
                        / (E_plus - E_minus);

    }

    return result;
}



void fix_values(const Eigen::VectorXd &deltaE_GeV, 
                Eigen::VectorXd &current_function, 
                double epsilon){

    for(size_t index = 0; index < current_function.size(); index++){

        if( current_function(index) * deltaE_GeV(index) < epsilon 
            or current_function(index) < 0.0 ){

            current_function(index) = 0.0;

        }

    }

}

Eigen::VectorXd get_diff_flux(const Eigen::VectorXd &int_flux, 
                              const Eigen::VectorXd &E_GeV){

    double deltalog10E = 
                ( std::log(E_GeV.tail(1)(0)) - std::log(E_GeV.head(1)(0)) )
                / E_GeV.size();

    Eigen::VectorXd result(int_flux.size());

    for(size_t index = 0; index < int_flux.size(); index++){

        double log10E = std::log10( E_GeV(index) );

        double E_plus =  std::pow( 10.0, log10E + 0.5 * deltalog10E );
        double E_minus = std::pow( 10.0, log10E - 0.5 * deltalog10E );

        double deltaE = E_plus - E_minus;

        result(index) = int_flux(index) / deltaE;

    }

    return result;
}

Eigen::VectorXd energy_bin_widths(const Eigen::VectorXd &E_GeV){

    double deltalog10E = 
                ( std::log( E_GeV.tail(1)(0) ) - std::log( E_GeV.head(1)(0) ) )
                / E_GeV.size();

    Eigen::VectorXd result( E_GeV.size() );

    for(size_t index = 0; index < E_GeV.size(); index++){

        double log10E = std::log10( E_GeV(index) );

        double E_plus = std::pow(  10.0, log10E + 0.5 * deltalog10E );
        double E_minus = std::pow( 10.0, log10E - 0.5 * deltalog10E );

        double deltaE = E_plus - E_minus;

        result(index) = deltaE;

    }

    return result;
}

namespace utils {

    double atan_diff(double x, double y){
        // From herbie
        return std::atan2( (x - y) , std::fma(x, y, 1.0) );
    }

} //utils
