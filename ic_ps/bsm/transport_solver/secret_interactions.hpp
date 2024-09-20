#ifndef SI_H
#define SI_H

#include <array>

#include <Eigen/Dense>

#include "utils.hpp"

namespace SI{

    double K(const mass_state i, const double E_plus, const double E_minus, 
             const std::array<double,3> &mass, 
             const double g, const double m_phi, const bool majorana);

    Eigen::VectorXd K(const mass_state i, 
                      const Eigen::VectorXd E_GeV, 
                      const std::array<double,3> &neutrino_masses_GeV,
                      const double g, const double m_phi, bool majorana); 

    double J(mass_state j, mass_state i, 
             double En_plus, double En_minus, 
             const std::array<double,3> &mass,
             double g, double m_phi, bool majorana);

    double J(mass_state j, mass_state i, 
             double En_plus, double En_minus, 
             double Em_plus, double Em_minus, 
             const std::array<double,3> &mass,
             double g, double m_phi, bool majorana);

    Eigen::MatrixXd I(mass_state j, mass_state i, 
                      const Eigen::VectorXd &E_GeV, 
                      const std::array<double,3> &neutrino_masses_GeV,
                      double g, double m_phi, bool majorana);

    double K_s(mass_state j, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width, double m_j, 
               double s_plus, double s_minus);

    double K_t_u(mass_state j, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width, double m_j, 
               double s_plus, double s_minus);

    double K_tu(mass_state j, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width, double m_j, 
               double s_plus, double s_minus);

    double K_st(mass_state j, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width, double m_j, 
               double s_plus, double s_minus);

    double K_pp(mass_state j, flavour_state flavour, double g, double m_phi, 
               bool majorana, double decay_width, double m_j, 
               double s_plus, double s_minus);

    double J_s(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double gamma , double m_k, 
               double t_plus, double t_minus);

    double J_t(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double gamma , double m_k, 
               double t_plus, double t_minus);

    double J_u(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double gamma , double m_k, 
               double t_plus, double t_minus);

    double J_tu(mass_state k, flavour_state flavour, double g, double m_phi, 
                bool majorana, double gamma , double m_k, 
                double t_plus, double t_minus);

    double J_st(mass_state k, flavour_state flavour, double g, double m_phi, 
                bool majorana, double gamma , double m_k, 
                double t_plus, double t_minus);

    double J_su(mass_state k, flavour_state flavour, double g, double m_phi, 
                bool majorana, double gamma , double m_k, 
                double t_plus, double t_minus);

    double J_s(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double gamma , double m_k, 
               double t_plus, double t_minus, double s_plus, double s_minus);

    double J_t(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double gamma , double m_k, 
               double t_plus, double t_minus, double s_plus, double s_minus);

    double J_u(mass_state k, flavour_state flavour, double g, double m_phi, 
               bool majorana, double gamma , double m_k, 
               double t_plus, double t_minus, double s_plus, double s_minus);

    double J_tu(mass_state k, flavour_state flavour, double g, double m_phi, 
                bool majorana, double gamma , double m_k, 
                double t_plus, double t_minus, double s_plus, double s_minus);

    double J_st(mass_state k, flavour_state flavour, double g, double m_phi, 
                bool majorana, double gamma , double m_k, 
                double t_plus, double t_minus, double s_plus, double s_minus);

    double J_su(mass_state k, flavour_state flavour, double g, double m_phi, 
                bool majorana, double gamma , double m_k, 
                double t_plus, double t_minus, double s_plus, double s_minus);
}

#endif
