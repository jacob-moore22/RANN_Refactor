/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/*  ----------------------------------------------------------------------
   Contributing authors: Christopher Barrett (MSU) barrett@me.msstate.edu
                              Doyl Dickel (MSU) doyl@me.msstate.edu
    ----------------------------------------------------------------------*/
/*
“The research described and the resulting data presented herein, unless
otherwise noted, was funded under PE 0602784A, Project T53 "Military
Engineering Applied Research", Task 002 under Contract No. W56HZV-17-C-0095,
managed by the U.S. Army Combat Capabilities Development Command (CCDC) and
the Engineer Research and Development Center (ERDC).  The work described in
this document was conducted at CAVS, MSU.  Permission was granted by ERDC
to publish this information. Any opinions, findings and conclusions or
recommendations expressed in this material are those of the author(s) and
do not necessarily reflect the views of the United States Army.​”

DISTRIBUTION A. Approved for public release; distribution unlimited. OPSEC#4918
 */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(spinrann,PairRANN);
// clang-format on
#else

#ifndef LMP_PAIR_RANN
#define LMP_PAIR_RANN

#include "pair_spin.h"

namespace LAMMPS_NS {

namespace RANN {
  //forward declarations
  class Activation;
  class Fingerprint;
  class State;
}    // namespace RANN

class PairRANN : public PairSpin {
 public:
  //inherited functions
  PairRANN(class LAMMPS *);
  ~PairRANN() override;
  void compute(int, int) override;
  void compute_single_pair(int, double *,double *);
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void init_list(int, NeighList *) override;

  //wrapper functions
  void errorf(const char *, int, const char *);
  int factorial(int);
  std::vector<std::string> tokenmaker(std::string,std::string);
  
  void cull_neighbor_list(double *,double *,double *,int *,int *,int *,int,int,double);

  //modular class creation
  RANN::Fingerprint *create_fingerprint(const char *);
  RANN::Activation *create_activation(const char *);
  RANN::State *create_state(const char *);

  //global variables
  int nelements;    // # of elements (distinct from LAMMPS atom types since multiple atom types can be mapped to one element)
  int nelementsp;                // nelements+1
  char **elements;               // names of elements
  char **elementsp;              // names of elements with "all" appended as the last "element"
  double *mass;                  // mass of each element
  double cutmax;                 // max radial distance for neighbor lists
  int *map;                      // mapping from atom types to elements
  int *fingerprintcount;         // static variable used in initialization
  int *fingerprintlength;        // # of input neurons defined by fingerprints of each element.
  int *fingerprintperelement;    // # of fingerprints for each element
  int *stateequationperelement;
  int *stateequationcount;
  bool doscreen;                 //screening is calculated if any defined fingerprint uses it
  bool
      allscreen;    //all fingerprints use screening so screened neighbors can be completely ignored
  bool dospin;
  int res;    //Resolution of function tables for cubic interpolation.
  int memguess;
  double *screening_min;
  double *screening_max;
	bool ***weightdefined;
	bool ***biasdefined;
	bool **dimensiondefined;
	bool ***bundle_inputdefined;
	bool ***bundle_outputdefined;
  int nmax1;
  int nmax2;
  int fmax;
  int fnmax;
  bool is_lammps;
  //memory actively written to during each compute:
  double *xn, *yn, *zn, *Sik, *dSikx, *dSiky, *dSikz, *dSijkx, *dSijky, *dSijkz, *sx, *sy, *sz,
      **dSijkxc, **dSijkyc, **dSijkzc, *dfeaturesx, *dfeaturesy, *dfeaturesz, *features;
  double *layer, *sum, *dsum1, *ddsum1, **dlayerx, **dlayery, **dlayerz, **dlayersumx, **dlayersumy,
      **dlayersumz;
  double **dsx, **dsy, **dsz, **dssumx, **dssumy, **dssumz,*sxx,*sxy,*sxz,*syy,*syz,*szz,
      **dsxx,**dsxy,**dsxz,**dsyy,**dsyz,**dszz,**dssumxx,**dssumxy,**dssumxz,**dssumyy,**dssumyz,**dssumzz;
  int *tn, *jl;
  bool *Bij;
  double hbar;

  struct Simulation {
    int *id;
    bool forces;
    bool spins;
    double **x;
    double **f;
    double **s;
    double box[3][3];
    double origin[3];
    double **features;
    double **dfx;
    double **dfy;
    double **dfz;
    double **dsx;
    double **dsy;
    double **dsz;
    int *ilist, *numneigh, **firstneigh, *type, inum, gnum;
  };

	struct NNarchitecture{
	  int layers;
	  int *dimensions;//vector of length layers with entries for neurons per layer
	  int *activations;//unused
	  int maxlayer;//longest layer (for memory allocation)
	  int sumlayers;
	  int *startI;
	  bool bundle;
	  int *bundles;
	  int **bundleinputsize;
	  int **bundleoutputsize;
	  bool **identitybundle;
	  int ***bundleinput;
	  int ***bundleoutput;
	  double ***bundleW;
	  double ***bundleB;
	  bool ***freezeW;
	  bool ***freezeB;
	};


  Simulation *sims;
  NNarchitecture *net;    //array of networks, 1 for each element.

 protected:
  RANN::Activation ****activation;
  RANN::Fingerprint ***fingerprints;
  RANN::State ***state;

 private:
  //new functions
  void allocate(
      const std::vector<std::string>
          &);    //called after reading element list, but before reading the rest of the potential
  void deallocate();
  void read_file(char *);    //read potential file
  void read_atom_types(std::vector<std::string>, char *, int);
  void read_fpe(
      std::vector<std::string>, std::vector<std::string>, char *,
      int);    //fingerprints per element. Count total fingerprints defined for each 1st element in element combinations
  void read_fingerprints(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_fingerprint_constants(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_network_layers(std::vector<std::string>, std::vector<std::string>, char *,
                           int);    //include input and output layer (hidden layers + 2)
  void read_layer_size(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_weight(std::vector<std::string>, std::vector<std::string>, FILE *, char *,
                   int *);    //weights should be formatted as properly shaped matrices
  void read_bias(std::vector<std::string>, std::vector<std::string>, FILE *, char *,
                 int *);    //biases should be formatted as properly shaped vectors
  void read_activation_functions(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_screening(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_mass(const std::vector<std::string> &, const std::vector<std::string> &, const char *,
                 int);
  void read_eospe(std::vector<std::string>, std::vector<std::string>, char *,int);    
  void read_eos(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_eos_constants(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_bundles(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_bundle_input(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_bundle_output(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_bundle_id(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_parameters(std::vector<std::string>, std::vector<std::string>, FILE *, char *, int *,char *);
  bool check_potential();    //after finishing reading potential file
  void propagateforward(double *, double **, int,
                        int);    //called by compute to get force and energy
  void propagateforwardspin(double *, double **, double **, double **, int,
                            int);    //called by compute to get force and energy
  void screening(int, int, int);
  void screen_neighbor_list(int *);
  void create_compute_arrays();
  void create_random_weights(int,int,int,int,int);
  void create_random_biases(int,int,int,int);
  void create_identity_wb(int,int,int,int,int);
};

}    // namespace LAMMPS_NS

#endif
#endif
