/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(spinmagmeam,PairMAGMEAM);
PairStyle(spinmagmeam/c,PairMAGMEAM);
// clang-format on
#else

#ifndef LMP_PAIR_MAGMEAM_H
#define LMP_PAIR_MAGMEAM_H

#include "pair_spin.h"

namespace LAMMPS_NS {

class PairMAGMEAM : public PairSpin {
 public:
  PairMAGMEAM(class LAMMPS *);
  ~PairMAGMEAM() override;
  void compute(int, int) override;
  void compute_single_pair(int, double *, double *) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  void init_list(int, class NeighList *) override;
  double init_one(int, int) override;
  void *extract(const char *, int &) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage() override;

 private:
  class MAGMEAM *meam_inst;
  double cutmax;                           // max cutoff for all elements
  int nlibelements;                        // # of library elements
  std::vector<std::string> libelements;    // names of library elements
  std::vector<double> mass;                // mass of library element

  double hbar;         // Planck constant (eV.ps.rad-1)
  double **scale;    // scaling factor for adapt
  double **torque;
  int *offinv;    // for inverting screening matrix (needed for spin dynamics)
  void allocate();
  void read_files(const std::string &, const std::string &);
  void read_global_meam_file(const std::string &);
  void read_user_meam_file(const std::string &);
  void neigh_strip(int, int *, int *, int **);
};

}    // namespace LAMMPS_NS

#endif
#endif
