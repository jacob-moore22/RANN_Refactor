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

#ifdef COMMAND_CLASS
// clang-format off
CommandStyle(replica/atom/swap,ReplicaAtomSwap);
// clang-format on
#else

#ifndef LMP_REPLICA_ATOM_SWAP_H
#define LMP_REPLICA_ATOM_SWAP_H

#include "command.h"

namespace LAMMPS_NS {

class ReplicaAtomSwap : public Command {
 public:
  ReplicaAtomSwap(class LAMMPS *);
  ~ReplicaAtomSwap() override;
  void command(int, char **) override;
  class RanPark *random_equal;

 private:
  int me, me_universe;                  // my proc ID in world and universe
  int iworld, nworlds;                  // world info
  double boltz;                         // copy from output->boltz
  MPI_Comm roots;                       // MPI comm with 1 root proc from each world
  class RanPark *ranswap, *ranboltz;    // RNGs for swapping and Boltz factor
  class Compute *c_pe;
  int nevery;                           // # of timesteps between swaps
  int nswaps;                           // # of tempering swaps to perform
  int seed_swap;                        // 0 = toggle swaps, n = RNG for swap direction
  int seed_boltz;                       // seed for Boltz factor comparison
  int whichfix;                         // index of temperature fix to use
  int fixstyle;                         // what kind of temperature fix is used
  int igroup, groupbit;
  int nswaptypes;
  int *type_list;
  int niswap, njswap;                  // # of i,j swap atoms on all procs
  int niswap_local, njswap_local;      // # of swap atoms on this proc
  int niswap_before, njswap_before;    // # of swap atoms on procs < this proc
  int nswap;                           // # of swap atoms on all procs
  int nswap_local;                     // # of swap atoms on this proc
  int nswap_before;                    // # of swap atoms on procs < this proc
  int partner,partner_set_index,partner_world;
  int partner_root;

  int *local_swap_iatom_list;
  int *local_swap_jatom_list;
  int *local_swap_atom_list;
  int ke_flag;            // yes = conserve ke, no = do not conserve ke
  double **sqrt_mass_ratio;
  int atom_swap_nmax;

  double beta;
  double *qtype;
  double energy_stored;
  double energy_stored_partner;
  bool unequal_cutoffs;

  int my_set_index;     // which set temp I am simulating
  double *set_temp;    // static list of replica set temperatures
  int **set_atom_types;
  int *index2world;     // index2world[i] = world simulating set index i
  int *world2index;     // world2index[i] = index simulated by world i
  int *world2root;     // world2root[i] = root proc of world i

  void scale_velocities(int, int);
  void print_status();
  int pick_i_swap_atom(int);
  int pick_j_swap_atom(int);
  double energy_full();
  int attempt_swap();
  void update_swap_atoms_list();
};

}    // namespace LAMMPS_NS

#endif
#endif
