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

#ifndef LMP_MAGMEAM_H
#define LMP_MAGMEAM_H

#include <cmath>
#include <string>
#include "pair_spin.h"
#define maxelt 5

namespace LAMMPS_NS {
class Memory;

  typedef enum { FCC, BCC, HCP, DIM, DIA, DIA3, B1, C11, L12, B2, CH4, LIN, ZIG, TRI , MAGBCC} lattice_t;

class MAGMEAM {
 public:
  MAGMEAM(Memory *mem);
  ~MAGMEAM();

 private:
  Memory *memory;

  // cutforce = force cutoff
  // cutforcesq = force cutoff squared

  double cutforce, cutforcesq;

  // Ec_meam = cohesive energy
  // re_meam = nearest-neighbor distance
  // B_meam = bulk modulus
  // ielt_meam = atomic number of element
  // A_meam = adjustable parameter
  // alpha_meam = sqrt(9*Omega*B/Ec)
  // rho0_meam = density scaling parameter
  // delta_meam = heat of formation for alloys
  // beta[0-3]_meam = electron density constants
  // t[0-3]_meam = coefficients on densities in Gamma computation
  // rho_ref_meam = background density for reference structure
  // ibar_meam(i) = selection parameter for Gamma function for elt i,
  // lattce_meam(i,j) = lattce configuration for elt i or alloy (i,j)
  // neltypes = maximum number of element type defined
  // eltind = index number of pair (similar to Voigt notation; ij = ji)
  // phir = pair potential function array
  // phirar[1-6] = spline coeffs
  // attrac_meam = attraction parameter in Rose energy
  // repuls_meam = repulsion parameter in Rose energy
  // nn2_meam = 1 if second nearest neighbors are to be computed, else 0
  // zbl_meam = 1 if zbl potential for small r to be use, else 0
  // emb_lin_neg = 1 if linear embedding function for rhob to be used, else 0
  // bkgd_dyn = 1 if reference densities follows Dynamo, else 0
  // Cmin_meam, Cmax_meam = min and max values in screening cutoff
  // rc_meam = cutoff distance for meam
  // delr_meam = cutoff region for meam
  // ebound_meam = factor giving maximum boundary of sceen fcn ellipse
  // augt1 = flag for whether t1 coefficient should be augmented
  // ialloy = flag for newer alloy formulation (as in dynamo code)
  // mix_ref_t = flag to recover "old" way of computing t in reference config
  // erose_form = selection parameter for form of E_rose function
  // gsmooth_factor = factor determining length of G smoothing region
  // vind[23]D = Voight notation index maps for 2 and 3D
  // v2D,v3D = array of factors to apply for Voight notation

  // nr,dr = pair function discretization parameters
  // nrar,rdrar = spline coeff array parameters

  // theta = angle between three atoms in line, zigzag, and trimer reference structures
  // stheta_meam = sin(theta/2) in radian used in line, zigzag, and trimer reference structures
  // ctheta_meam = cos(theta/2) in radian used in line, zigzag, and trimer reference structures

  // imag = 1 if magnetic MEAM is used
  // mag_Ec, mag_re, mag_alpha, mag_lattce parameters for the magnetic pair potential
  // mag_B_meam = parameter for magnetic meam
  // mag_delrho0_meam, mag_delrho1_meam, mag_delrho2_meam, mag_delrho3_meam, mag_delrho00_meam magnitude of magnetic density terms
  // mag_beta0_meam, mag_beta1_meam, mag_beta2_meam, mag_beta3_meam, mag_beta4_meam length scale of magnetic interactions
  // Cmin_magmeam, Cmax_magmeam = min and max values in magnetic screening cutoff
  // Cmin_magpair, Cmax_magpair = min and max values in magnetic screening pair potential cutoff

  
  double Ec_meam[maxelt][maxelt], re_meam[maxelt][maxelt];
  double A_meam[maxelt], alpha_meam[maxelt][maxelt], rho0_meam[maxelt];
  double delta_meam[maxelt][maxelt];
  double beta0_meam[maxelt], beta1_meam[maxelt];
  double beta2_meam[maxelt], beta3_meam[maxelt];
  double t0_meam[maxelt], t1_meam[maxelt];
  double t2_meam[maxelt], t3_meam[maxelt];
  double rho_ref_meam[maxelt];
  int ibar_meam[maxelt], ielt_meam[maxelt];
  lattice_t lattce_meam[maxelt][maxelt];
  int nn2_meam[maxelt][maxelt];
  int zbl_meam[maxelt][maxelt];
  int eltind[maxelt][maxelt];
  int neltypes;

  double **phir;

  double **phirar, **phirar1, **phirar2, **phirar3, **phirar4, **phirar5, **phirar6;

  double **magr;

  double **magrar, **magrar1, **magrar2, **magrar3, **magrar4, **magrar5, **magrar6;

  double attrac_meam[maxelt][maxelt], repuls_meam[maxelt][maxelt];
  double mag_attrac[maxelt][maxelt], mag_repuls[maxelt][maxelt];

  double Cmin_meam[maxelt][maxelt][maxelt];
  double Cmax_meam[maxelt][maxelt][maxelt];
  double Cmin_magmeam[maxelt][maxelt][maxelt];
  double Cmax_magmeam[maxelt][maxelt][maxelt];
  double Cmin_magpair[maxelt][maxelt][maxelt];
  double Cmax_magpair[maxelt][maxelt][maxelt];
  double rc_meam, delr_meam, ebound_meam[maxelt][maxelt];
  int augt1, ialloy, mix_ref_t, erose_form;
  int emb_lin_neg, bkgd_dyn;
  double gsmooth_factor;

  int vind2D[3][3], vind3D[3][3][3], vind4D[3][3][3][3];    // x-y-z to Voigt-like index
  int v2D[6], v3D[10], v4D[15];                  // multiplicity of Voigt index (i.e. [1] -> xy+yx = 2

  int nr, nrar;
  double dr, rdrar;

  // parameters for magnetic MEAM
  int imag;
  double mag_B_meam[maxelt], mag_delrho0_meam[maxelt], mag_delrho1_meam[maxelt], mag_delrho2_meam[maxelt], mag_delrho3_meam[maxelt], mag_delrho4_meam[maxelt], mag_delrho00_meam[maxelt];
  double mag_beta0_meam[maxelt], mag_beta1_meam[maxelt], mag_beta2_meam[maxelt], mag_beta3_meam[maxelt], mag_beta4_meam[maxelt], mag_beta00_meam[maxelt];
  double mag_Ec[maxelt][maxelt], mag_re[maxelt][maxelt], mag_alpha[maxelt][maxelt]; 
  lattice_t mag_lattce[maxelt][maxelt];
 public:
  int nmax;
  double *rho, *rho0, *rho1, *rho2, *rho3, *frhop, *frhopp;
  double *rho0m, *rho1m, *rho2m, *rho3m, *rho4m, *rho00;
  double *gamma, *dgamma1, *dgamma2, *dgamma3, *d2gamma2, *arho0m, *arho2b, *arho2bm, *arho4cm, *arho00, *arho00m;
  double **arho1, **arho1m, **arho2, **arho2m, **arho3, **arho3b, **arho3m, **arho3bm, **arho4m, **arho4bm, **t_ave, **tsq_ave;

  int maxneigh;
  double *scrfcn, *dscrfcn, *fcpair;
  double *scrfcnmag, *dscrfcnmag;
  double *scrfcnpair, *dscrfcnpair;
  //angle for trimer, zigzag, line reference structures
  double stheta_meam[maxelt][maxelt];
  double ctheta_meam[maxelt][maxelt];

 protected:
  // meam_funcs.cpp

  //-----------------------------------------------------------------------------
  // Cutoff function
  //
  static double fcut(const double xi)
  {
    double a;
    if (xi >= 1.0)
      return 1.0;
    else if (xi <= 0.0)
      return 0.0;
    else {
      // ( 1.d0 - (1.d0 - xi)**4 )**2, but with better codegen
      a = 1.0 - xi;
      a *= a;
      a *= a;
      a = 1.0 - a;
      return a * a;
    }
  }

  //-----------------------------------------------------------------------------
  // Cutoff function and derivative
  //
  static double dfcut(const double xi, double &dfc)
  {
    double a, a3, a4, a1m4;
    if (xi >= 1.0) {
      dfc = 0.0;
      return 1.0;
    } else if (xi <= 0.0) {
      dfc = 0.0;
      return 0.0;
    } else {
      a = 1.0 - xi;
      a3 = a * a * a;
      a4 = a * a3;
      a1m4 = 1.0 - a4;

      dfc = 8 * a1m4 * a3;
      return a1m4 * a1m4;
    }
  }

  //-----------------------------------------------------------------------------
  // Derivative of Cikj w.r.t. rij
  //     Inputs: rij,rij2,rik2,rjk2
  //
  static double dCfunc(const double rij2, const double rik2, const double rjk2)
  {
    double rij4, a, asq, b, denom;

    rij4 = rij2 * rij2;
    a = rik2 - rjk2;
    b = rik2 + rjk2;
    asq = a * a;
    denom = rij4 - asq;
    denom = denom * denom;
    return -4 * (-2 * rij2 * asq + rij4 * b + asq * b) / denom;
  }

  //-----------------------------------------------------------------------------
  // Derivative of Cikj w.r.t. rik and rjk
  //     Inputs: rij,rij2,rik2,rjk2
  //
  static void dCfunc2(const double rij2, const double rik2, const double rjk2, double &dCikj1,
                      double &dCikj2)
  {
    double rij4, rik4, rjk4, a, denom;

    rij4 = rij2 * rij2;
    rik4 = rik2 * rik2;
    rjk4 = rjk2 * rjk2;
    a = rik2 - rjk2;
    denom = rij4 - a * a;
    denom = denom * denom;
    dCikj1 = 4 * rij2 * (rij4 + rik4 + 2 * rik2 * rjk2 - 3 * rjk4 - 2 * rij2 * a) / denom;
    dCikj2 = 4 * rij2 * (rij4 - 3 * rik4 + 2 * rik2 * rjk2 + rjk4 + 2 * rij2 * a) / denom;
  }

  double G_gam(const double gamma, const int ibar, int &errorflag) const;
  double dG_gam(const double gamma, const int ibar, double &dG, double &ddG) const;
  static double zbl(const double r, const int z1, const int z2);
  double embedding(const double A, const double B, const double Ec, const double rhobar, double &dF, double &ddF) const;
  static double erose(const double r, const double re, const double alpha, const double Ec,
                      const double repuls, const double attrac, const int form);

  static void get_shpfcn(const lattice_t latt, const double sthe, const double cthe,
                         double (&s)[3]);

  static int get_Zij2(const lattice_t latt, const double cmin, const double cmax, const double sthe,
                      double &a, double &S);
  static int get_Zij2_b2nn(const lattice_t latt, const double cmin, const double cmax, double &S);

 protected:
  void meam_checkindex(int, int, int, int *, int *);
  void getscreen(int i, double *scrfcn, double *dscrfcn, double *scrfcnmag, double *dscrfcnmag, double *scrfcnpair,
		 double *dscrfcnpair, double *fcpair, double **x, int numneigh, int *firstneigh, int numneigh_full,
		 int *firstneigh_full, int ntype, int *type, int *fmap);
  void calc_rho1(int i, int ntype, int *type, int *fmap, double **x, double **sp, int numneigh, int *firstneigh, double *scrfcn, double *scrfcnmag, double *fcpair);
  void update_rho1(int i, int ntype, int *type, int *fmap, double **x, double **sp, int numneigh, int *firstneigh, double *scrfcn, double *scrfcnmag, double *fcpair);

  void alloyparams();
  void compute_pair_meam();
  void compute_pair_mag();
  double phi_meam(double, int, int);
  double phi_mag(double, int, int);
  double phi_meam_series(const double scrn, const int Z1, const int Z2, const int a, const int b,
                         const double r, const double arat);
  void compute_reference_density();
  void get_tavref(double *, double *, double *, double *, double *, double *, double, double,
                  double, double, double, double, double, int, int, lattice_t);
  void get_tmagavref(double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double, double, double, double,
		     double, double, double, double, double, double, double, double, double, int, int, lattice_t);
  void get_sijk(double, int, int, int, double *);
  void get_densref(double, int, int, double *, double *, double *, double *, double *, double *,
                   double *, double *);
  void get_magdensref(double, int, int, double *, double *, double *, double *, double *, double *, double *, double *,
		      double *, double *, double *, double *);
  void interpolate_meam(int);
  void interpolate_pair(int);

 public:
  // clang-format off
  //-----------------------------------------------------------------------------
  // convert lattice spec to lattice_t
  // only use single-element lattices if single=true
  // return false on failure
  // return true and set lat on success
  static bool str_to_lat(const std::string & str, bool single, lattice_t& lat)
  {
    if (str == "fcc") lat = FCC;
    else if (str == "bcc") lat = BCC;
    else if (str == "hcp") lat = HCP;
    else if (str == "dim") lat = DIM;
    else if (str == "dia") lat = DIA;
    else if (str == "dia3") lat = DIA3;
    else if (str == "lin") lat = LIN;
    else if (str == "zig") lat = ZIG;
    else if (str == "tri") lat = TRI;
    else if (str == "magbcc") lat = MAGBCC;
    else {
      if (single)
        return false;

      if (str == "b1") lat = B1;
      else if (str == "c11") lat = C11;
      else if (str == "l12") lat = L12;
      else if (str == "b2") lat = B2;
      else if (str == "ch4") lat = CH4;
      else if (str == "lin") lat =LIN;
      else if (str == "zig") lat = ZIG;
      else if (str == "tri") lat = TRI;
      else return false;
    }
    return true;
  }
  // clang-format on
  static int get_Zij(const lattice_t latt);
  void meam_setup_global(int nelt, lattice_t *lat, int *ielement, double *atwt, double *alpha,
                         double *b0, double *b1, double *b2, double *b3, double *alat, double *esub,
                         double *asub, double *t0, double *t1, double *t2, double *t3,
                         double *rozero, int *ibar);
  void meam_setup_param(int which, double value, int nindex, int *index /*index(3)*/,
                        int *errorflag);
  void meam_setup_done(double *cutmax);
  void meam_dens_setup(int atom_nmax, int nall, int n_neigh);
  void meam_dens_init(int i, int ntype, int *type, int *fmap, double **x, double **sp, int numneigh,
                      int *firstneigh, int numneigh_full, int *firstneigh_full, int fnoffset);
  void meam_dens_update(int i, int ntype, int *type, int *fmap, double **x, double **sp, int numneigh,
                      int *firstneigh, int numneigh_full, int *firstneigh_full, int fnoffset);
  void meam_dens_final(int nlocal, int eflag_either, int eflag_global, int eflag_atom,
                       double *eng_vdwl, double *eatom, int ntype, int *type, int *fmap,
                       double **scale, int &errorflag, int engflag);
  void update_dens_final(int i, int ntype, int *type, int *fmap, double **scale, int &errorflag);
  void meam_force(int i, int eflag_global, int eflag_atom, int vflag_global, int vflag_atom,
                  double *eng_vdwl, double *eatom, int ntype, int *type, int *fmap, double **scale,
                  double **x, double **sp, int numneigh, int *firstneigh, int numneigh_full,
                  int *firstneigh_full, int fnoffset, double **f, double **fm, double **fmds, double **vatom, double *virial, double hbar, double emag);
  void meam_spin(int i, int ntype, int *type, int *fmap, double **scale, double **x, double **sp,
		 int numneigh, int *firstneigh, int fnoffset, double *fmi, double *fmdsi, double hbar);
};

// Functions we need for compat

static inline bool iszero(const double f)
{
  return fabs(f) < 1e-20;
}

static inline bool isone(const double f)
{
  return fabs(f - 1.0) < 1e-20;
}

// Helper functions

static inline double fdiv_zero(const double n, const double d)
{
  if (iszero(d)) return 0.0;
  return n / d;
}

}    // namespace LAMMPS_NS
#endif
