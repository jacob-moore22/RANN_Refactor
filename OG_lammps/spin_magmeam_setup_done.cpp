// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#include "spin_magmeam.h"

#include "math_special.h"
#include "memory.h"

#include <cmath>
#include <algorithm>
using namespace LAMMPS_NS;

void MAGMEAM::meam_setup_done(double* cutmax)
{
  int nv2, nv3, nv4, m, n, p, q;
  
  //     Force cutoff
  this->cutforce = this->rc_meam;
  this->cutforcesq = this->cutforce * this->cutforce;

  //     Pass cutoff back to calling program
  *cutmax = this->cutforce;

  //     Augment t1 term
  for (int i = 0; i < maxelt; i++)
    this->t1_meam[i] = this->t1_meam[i] + this->augt1 * 3.0 / 5.0 * this->t3_meam[i];

  // Parameters for magnetic MEAM
  if (this->imag == 0) {
    for (int i = 0; i < maxelt; i++) {
      for (int j = i; j < maxelt; j++) {
	this->mag_Ec[i][j] = 0.0;
	this->mag_alpha[i][j] = 0.0;
	this->mag_re[i][j] = 0.0;
      }
      this->mag_B_meam[i] = 0.0;
      this->mag_delrho0_meam[i] = 0.0;
      this->mag_delrho1_meam[i] = 0.0;
      this->mag_delrho2_meam[i] = 0.0;
      this->mag_delrho3_meam[i] = 0.0;
      this->mag_delrho4_meam[i] = 0.0;
      this->mag_delrho00_meam[i] = 0.0;
      this->mag_beta0_meam[i] = 0.0;
      this->mag_beta1_meam[i] = 0.0;
      this->mag_beta2_meam[i] = 0.0;
      this->mag_beta3_meam[i] = 0.0;
      this->mag_beta4_meam[i] = 0.0;
      this->mag_beta00_meam[i] = 0.0;
    }
  } else {
    for (int i = 0; i < maxelt; i++) {
      if (this->mag_beta0_meam[i] == 0)
	this->mag_beta0_meam[i] = this->beta0_meam[i];
      if (this->mag_beta1_meam[i] == 0)
	this->mag_beta1_meam[i] = this->beta1_meam[i];
      if (this->mag_beta2_meam[i] == 0)
	this->mag_beta2_meam[i] = this->beta2_meam[i];
      if (this->mag_beta3_meam[i] == 0)
	this->mag_beta3_meam[i] = this->beta3_meam[i];
      if (this->mag_beta4_meam[i] == 0)
	this->mag_beta4_meam[i] = this->beta0_meam[i];
      if (this->mag_beta00_meam[i] == 0)
	this->mag_beta00_meam[i] = this->mag_beta0_meam[i];
    }
  }
  
  //     Compute off-diagonal alloy parameters
  alloyparams();

  // indices and factors for Voight notation
  nv2 = 0;
  nv3 = 0;
  nv4 = 0;
  for (m = 0; m < 3; m++) {
    for (n = m; n < 3; n++) {
      this->vind2D[m][n] = nv2;
      this->vind2D[n][m] = nv2;
      nv2 = nv2 + 1;
      for (p = n; p < 3; p++) {
        this->vind3D[m][n][p] = nv3;
        this->vind3D[m][p][n] = nv3;
        this->vind3D[n][m][p] = nv3;
        this->vind3D[n][p][m] = nv3;
        this->vind3D[p][m][n] = nv3;
        this->vind3D[p][n][m] = nv3;
        nv3 = nv3 + 1;
	for (q = p; q < 3; q++) {
	  this->vind4D[m][n][p][q] = nv4;
	  this->vind4D[m][n][q][p] = nv4;
	  this->vind4D[m][p][n][q] = nv4;
	  this->vind4D[m][p][q][n] = nv4;
	  this->vind4D[m][q][n][p] = nv4;
	  this->vind4D[m][q][p][n] = nv4;
	  this->vind4D[n][m][p][q] = nv4;
	  this->vind4D[n][m][q][p] = nv4;
	  this->vind4D[n][p][m][q] = nv4;
	  this->vind4D[n][p][q][m] = nv4;
	  this->vind4D[n][q][m][p] = nv4;
	  this->vind4D[n][q][p][m] = nv4;
	  this->vind4D[p][m][n][q] = nv4;
	  this->vind4D[p][m][q][n] = nv4;
	  this->vind4D[p][n][m][q] = nv4;
	  this->vind4D[p][n][q][m] = nv4;
	  this->vind4D[p][q][m][n] = nv4;
	  this->vind4D[p][q][n][m] = nv4;
	  this->vind4D[q][m][n][p] = nv4;
	  this->vind4D[q][m][p][n] = nv4;
	  this->vind4D[q][n][m][p] = nv4;
	  this->vind4D[q][n][p][m] = nv4;
	  this->vind4D[q][p][m][n] = nv4;
	  this->vind4D[q][p][n][m] = nv4;
	  nv4 = nv4 + 1;
	}
      }
    }
  }

  this->v2D[0] = 1;
  this->v2D[1] = 2;
  this->v2D[2] = 2;
  this->v2D[3] = 1;
  this->v2D[4] = 2;
  this->v2D[5] = 1;

  this->v3D[0] = 1;
  this->v3D[1] = 3;
  this->v3D[2] = 3;
  this->v3D[3] = 3;
  this->v3D[4] = 6;
  this->v3D[5] = 3;
  this->v3D[6] = 1;
  this->v3D[7] = 3;
  this->v3D[8] = 3;
  this->v3D[9] = 1;

  this->v4D[0] = 1;
  this->v4D[1] = 4;
  this->v4D[2] = 4;
  this->v4D[3] = 6;
  this->v4D[4] = 12;
  this->v4D[5] = 6;
  this->v4D[6] = 4;
  this->v4D[7] = 12;
  this->v4D[8] = 12;
  this->v4D[9] = 4;
  this->v4D[10] = 1;
  this->v4D[11] = 4;
  this->v4D[12] = 6;
  this->v4D[13] = 4;
  this->v4D[14] = 1;

  nv2 = 0;
  for (m = 0; m < this->neltypes; m++) {
    for (n = m; n < this->neltypes; n++) {
      this->eltind[m][n] = nv2;
      this->eltind[n][m] = nv2;
      nv2 = nv2 + 1;
    }
  }

  //     Compute background densities for reference structure
  compute_reference_density();

  //     Compute pair potentials and setup arrays for interpolation
  this->nr = 1000;
  this->dr = 1.1 * this->rc_meam / this->nr;
  compute_pair_meam();
  if (this->imag == 1)
    compute_pair_mag();
}

// ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
// Fill off-diagonal alloy parameters
void MAGMEAM::alloyparams()
{

  int i, j, k;
  double eb, ebmag, ebpair;

  // Loop over pairs
  for (i = 0; i < this->neltypes; i++) {
    for (j = 0; j < this->neltypes; j++) {
      // Treat off-diagonal pairs
      // If i>j, set all equal to i<j case (which has already been set,
      // here or in the input file)
      if (i > j) {
        this->re_meam[i][j] = this->re_meam[j][i];
        this->Ec_meam[i][j] = this->Ec_meam[j][i];
        this->alpha_meam[i][j] = this->alpha_meam[j][i];
        this->lattce_meam[i][j] = this->lattce_meam[j][i];
        this->mag_re[i][j] = this->mag_re[j][i];
        this->mag_Ec[i][j] = this->mag_Ec[j][i];
        this->mag_alpha[i][j] = this->mag_alpha[j][i];
        this->mag_lattce[i][j] = this->mag_lattce[j][i];
        this->nn2_meam[i][j] = this->nn2_meam[j][i];
        // theta for lin,tri,zig references
        this->stheta_meam[i][j] = this->stheta_meam[j][i];
        this->ctheta_meam[i][j] = this->ctheta_meam[j][i];
        // If i<j and term is unset, use default values (e.g. mean of i-i and
        // j-j)
      } else if (j > i) {
        if (iszero(this->Ec_meam[i][j])) {
          if (this->lattce_meam[i][j] == L12)
            this->Ec_meam[i][j] =
              (3 * this->Ec_meam[i][i] + this->Ec_meam[j][j]) / 4.0 - this->delta_meam[i][j];
          else if (this->lattce_meam[i][j] == C11) {
            if (this->lattce_meam[i][i] == DIA)
              this->Ec_meam[i][j] =
                (2 * this->Ec_meam[i][i] + this->Ec_meam[j][j]) / 3.0 - this->delta_meam[i][j];
            else
              this->Ec_meam[i][j] =
                (this->Ec_meam[i][i] + 2 * this->Ec_meam[j][j]) / 3.0 - this->delta_meam[i][j];
          } else
            this->Ec_meam[i][j] = (this->Ec_meam[i][i] + this->Ec_meam[j][j]) / 2.0 - this->delta_meam[i][j];
        }
        if (iszero(this->alpha_meam[i][j]))
          this->alpha_meam[i][j] = (this->alpha_meam[i][i] + this->alpha_meam[j][j]) / 2.0;
        if (iszero(this->re_meam[i][j]))
          this->re_meam[i][j] = (this->re_meam[i][i] + this->re_meam[j][j]) / 2.0;
      }
    }
  }

  // Cmin[i][k][j] is symmetric in i-j, but not k.  For all triplets
  // where i>j, set equal to the i<j element.  Likewise for Cmax.
  for (i = 1; i < this->neltypes; i++) {
    for (j = 0; j < i; j++) {
      for (k = 0; k < this->neltypes; k++) {
        this->Cmin_meam[i][j][k] = this->Cmin_meam[j][i][k];
        this->Cmax_meam[i][j][k] = this->Cmax_meam[j][i][k];
        this->Cmin_magmeam[i][j][k] = this->Cmin_magmeam[j][i][k];
        this->Cmax_magmeam[i][j][k] = this->Cmax_magmeam[j][i][k];
	this->Cmin_magpair[i][j][k] = this->Cmin_magpair[j][i][k];
        this->Cmax_magpair[i][j][k] = this->Cmax_magpair[j][i][k];
      }
    }
  }

  // ebound gives the squared distance such that, for rik2 or rjk2>ebound,
  // atom k definitely lies outside the screening function ellipse (so
  // there is no need to calculate its effects).  Here, compute it for all
  // triplets [i][j][k] so that ebound[i][j] is the maximized over k
  for (i = 0; i < this->neltypes; i++) {
    for (j = 0; j < this->neltypes; j++) {
      for (k = 0; k < this->neltypes; k++) {
        eb = (this->Cmax_meam[i][j][k] * this->Cmax_meam[i][j][k]) / (4.0 * (this->Cmax_meam[i][j][k] - 1.0));
        ebmag = (this->Cmax_magmeam[i][j][k] * this->Cmax_magmeam[i][j][k]) / (4.0 * (this->Cmax_magmeam[i][j][k] - 1.0));	
        ebpair = (this->Cmax_magpair[i][j][k] * this->Cmax_magpair[i][j][k]) / (4.0 * (this->Cmax_magpair[i][j][k] - 1.0));	
        this->ebound_meam[i][j] = std::max(this->ebound_meam[i][j], eb);
        this->ebound_meam[i][j] = std::max(this->ebound_meam[i][j], ebmag);
        this->ebound_meam[i][j] = std::max(this->ebound_meam[i][j], ebpair);
      }
    }
  }
}

//-----------------------------------------------------------------------
// compute MEAM pair potential for each pair of element types
//

void MAGMEAM::compute_pair_meam()
{
  double r;
  int j, a, b, nv2;
  double astar, frac, phizbl;
  int Z1, Z2;
  double arat, rarat, scrn, scrn2;
  double phiaa, phibb /*unused:,phitmp*/;
  double C, s111, s112, s221, S11, S22;

  // check for previously allocated arrays and free them
  if (this->phir != nullptr)
    memory->destroy(this->phir);
  if (this->phirar != nullptr)
    memory->destroy(this->phirar);
  if (this->phirar1 != nullptr)
    memory->destroy(this->phirar1);
  if (this->phirar2 != nullptr)
    memory->destroy(this->phirar2);
  if (this->phirar3 != nullptr)
    memory->destroy(this->phirar3);
  if (this->phirar4 != nullptr)
    memory->destroy(this->phirar4);
  if (this->phirar5 != nullptr)
    memory->destroy(this->phirar5);
  if (this->phirar6 != nullptr)
    memory->destroy(this->phirar6);

  // allocate memory for array that defines the potential
  memory->create(this->phir, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:phir");

  // allocate coeff memory

  memory->create(this->phirar, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:phirar");
  memory->create(this->phirar1, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:phirar1");
  memory->create(this->phirar2, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:phirar2");
  memory->create(this->phirar3, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:phirar3");
  memory->create(this->phirar4, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:phirar4");
  memory->create(this->phirar5, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:phirar5");
  memory->create(this->phirar6, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:phirar6");

  // loop over pairs of element types
  nv2 = 0;
  for (a = 0; a < this->neltypes; a++) {
    for (b = a; b < this->neltypes; b++) {
      // loop over r values and compute
      for (j = 0; j < this->nr; j++) {
        r = j * this->dr;

        this->phir[nv2][j] = phi_meam(r, a, b);

        // if using second-nearest neighbor, solve recursive problem
        // (see Lee and Baskes, PRB 62(13):8564 eqn.(21))
        if (this->nn2_meam[a][b] == 1) {
          Z1 = get_Zij(this->lattce_meam[a][b]);
          Z2 = get_Zij2(this->lattce_meam[a][b], this->Cmin_meam[a][a][b],
                     this->Cmax_meam[a][a][b], this->stheta_meam[a][b], arat, scrn);

          //     The B1, B2,  and L12 cases with NN2 have a trick to them; we need to
          //     compute the contributions from second nearest neighbors, like a-a
          //     pairs, but need to include NN2 contributions to those pairs as
          //     well.
          if (this->lattce_meam[a][b] == B1 || this->lattce_meam[a][b] == B2 ||
              this->lattce_meam[a][b] == L12 || this->lattce_meam[a][b] == DIA) {
            rarat = r * arat;

            //               phi_aa
            phiaa = phi_meam(rarat, a, a);
            Z1 = get_Zij(this->lattce_meam[a][a]);
            Z2 = get_Zij2(this->lattce_meam[a][a], this->Cmin_meam[a][a][a],
                     this->Cmax_meam[a][a][a], this->stheta_meam[a][a], arat, scrn);
            phiaa+= phi_meam_series(scrn, Z1, Z2, a, a, rarat, arat);

            //               phi_bb
            phibb = phi_meam(rarat, b, b);
            Z1 = get_Zij(this->lattce_meam[b][b]);
            Z2 = get_Zij2(this->lattce_meam[b][b], this->Cmin_meam[b][b][b],
                     this->Cmax_meam[b][b][b], this->stheta_meam[b][b], arat, scrn);
            phibb+= phi_meam_series(scrn, Z1, Z2, b, b, rarat, arat);

            if (this->lattce_meam[a][b] == B1 || this->lattce_meam[a][b] == B2 ||
                this->lattce_meam[a][b] == DIA) {
              //     Add contributions to the B1 or B2 potential
              Z1 = get_Zij(this->lattce_meam[a][b]);
              Z2 = get_Zij2(this->lattce_meam[a][b], this->Cmin_meam[a][a][b],
                       this->Cmax_meam[a][a][b], this->stheta_meam[a][b],  arat, scrn);
              this->phir[nv2][j] = this->phir[nv2][j] - Z2 * scrn / (2 * Z1) * phiaa;
              Z2 = get_Zij2(this->lattce_meam[a][b], this->Cmin_meam[b][b][a],
                       this->Cmax_meam[b][b][a], this->stheta_meam[a][b], arat, scrn2);

              this->phir[nv2][j] = this->phir[nv2][j] - Z2 * scrn2 / (2 * Z1) * phibb;

            } else if (this->lattce_meam[a][b] == L12) {
              //     The L12 case has one last trick; we have to be careful to
              //     compute
              //     the correct screening between 2nd-neighbor pairs.  1-1
              //     second-neighbor pairs are screened by 2 type 1 atoms and
              //     two type
              //     2 atoms.  2-2 second-neighbor pairs are screened by 4 type
              //     1
              //     atoms.
              C = 1.0;
              get_sijk(C, a, a, a, &s111);
              get_sijk(C, a, a, b, &s112);
              get_sijk(C, b, b, a, &s221);
              S11 = s111 * s111 * s112 * s112;
              S22 = pow(s221, 4);
              this->phir[nv2][j] = this->phir[nv2][j] - 0.75 * S11 * phiaa - 0.25 * S22 * phibb;
            }

          } else {
            this->phir[nv2][j]+= phi_meam_series(scrn, Z1, Z2, a, b, r, arat);
          }
        }

        // For Zbl potential:
        // if astar <= -3
        //   potential is zbl potential
        // else if -3 < astar < -1
        //   potential is linear combination with zbl potential
        // endif
        if (this->zbl_meam[a][b] == 1) {
          astar = this->alpha_meam[a][b] * (r / this->re_meam[a][b] - 1.0);
          if (astar <= -3.0)
            this->phir[nv2][j] = zbl(r, this->ielt_meam[a], this->ielt_meam[b]);
          else if (astar > -3.0 && astar < -1.0) {
            frac = fcut(1 - (astar + 1.0) / (-3.0 + 1.0));
            phizbl = zbl(r, this->ielt_meam[a], this->ielt_meam[b]);
            this->phir[nv2][j] = frac * this->phir[nv2][j] + (1 - frac) * phizbl;
          }
        }
      }

      // call interpolation
      interpolate_meam(nv2);

      nv2 = nv2 + 1;
    }
  }
}

//----------------------------------------------------------------------c
// Compute MEAM pair potential for distance r, element types a and b
//
double MAGMEAM::phi_meam(double r, int a, int b)
{
  /*unused:double a1,a2,a12;*/
  double t11av, t21av, t31av, t12av, t22av, t32av;
  double G1, G2, s1[3], s2[3], rho0_1, rho0_2;
  double Gam1, Gam2, Z1, Z2;
  double rhobar1, rhobar2, F1, F2, dF, ddF;
  double rho01, rho11, rho21, rho31;
  double rho02, rho12, rho22, rho32;
  double scalfac, phiaa, phibb;
  double Eu;
  double arat, scrn, scrn2;
  int Z12, errorflag;
  int Z1nn, Z2nn;
  lattice_t latta /*unused:,lattb*/;
  double rho_bkgd1, rho_bkgd2;
  double b11s, b22s;

  double phi_m = 0.0;

  // Equation numbers below refer to:
  //   I. Huang et.al., Modelling simul. Mater. Sci. Eng. 3:615

  // get number of neighbors in the reference structure
  //   Nref[i][j] = # of i's neighbors of type j
  Z1 = get_Zij(this->lattce_meam[a][a]);
  Z2 = get_Zij(this->lattce_meam[b][b]);
  Z12 = get_Zij(this->lattce_meam[a][b]);

  get_densref(r, a, b, &rho01, &rho11, &rho21, &rho31, &rho02, &rho12, &rho22, &rho32);

  // if densities are too small, numerical problems may result; just return zero
  if (rho01 <= 1e-14 && rho02 <= 1e-14)
    return 0.0;

  // calculate average weighting factors for the reference structure
  if (this->lattce_meam[a][b] == C11) {
    if (this->ialloy == 2) {
      t11av = this->t1_meam[a];
      t12av = this->t1_meam[b];
      t21av = this->t2_meam[a];
      t22av = this->t2_meam[b];
      t31av = this->t3_meam[a];
      t32av = this->t3_meam[b];
    } else {
      scalfac = 1.0 / (rho01 + rho02);
      t11av = scalfac * (this->t1_meam[a] * rho01 + this->t1_meam[b] * rho02);
      t12av = t11av;
      t21av = scalfac * (this->t2_meam[a] * rho01 + this->t2_meam[b] * rho02);
      t22av = t21av;
      t31av = scalfac * (this->t3_meam[a] * rho01 + this->t3_meam[b] * rho02);
      t32av = t31av;
    }
  } else {
    // average weighting factors for the reference structure, eqn. I.8
    get_tavref(&t11av, &t21av, &t31av, &t12av, &t22av, &t32av, this->t1_meam[a], this->t2_meam[a],
               this->t3_meam[a], this->t1_meam[b], this->t2_meam[b], this->t3_meam[b], r, a, b,
               this->lattce_meam[a][b]);
  }

  // for c11b structure, calculate background electron densities
  if (this->lattce_meam[a][b] == C11) {
    latta = this->lattce_meam[a][a];
    if (latta == DIA) {
      rhobar1 = MathSpecial::square((Z12 / 2) * (rho02 + rho01))
                + t11av * MathSpecial::square(rho12 - rho11)
                + t21av / 6.0 * MathSpecial::square(rho22 + rho21)
                + 121.0 / 40.0 * t31av * MathSpecial::square(rho32 - rho31);
      rhobar1 = sqrt(rhobar1);
      rhobar2 = MathSpecial::square(Z12 * rho01) + 2.0 / 3.0 * t21av * MathSpecial::square(rho21);
      rhobar2 = sqrt(rhobar2);
    } else {
      rhobar2 = MathSpecial::square((Z12 / 2) * (rho01 + rho02))
                + t12av * MathSpecial::square(rho11 - rho12)
                + t22av / 6.0 * MathSpecial::square(rho21 + rho22)
                + 121.0 / 40.0 * t32av * MathSpecial::square(rho31 - rho32);
      rhobar2 = sqrt(rhobar2);
      rhobar1 = MathSpecial::square(Z12 * rho02) + 2.0 / 3.0 * t22av * MathSpecial::square(rho22);
      rhobar1 = sqrt(rhobar1);
    }
  } else {
    // for other structures, use formalism developed in Huang's paper
    //
    //     composition-dependent scaling, equation I.7
    //     If using mixing rule for t, apply to reference structure; else
    //     use precomputed values
    if (this->mix_ref_t == 1) {
      if (this->ibar_meam[a] <= 0)
        G1 = 1.0;
      else {
        get_shpfcn(this->lattce_meam[a][a], this->stheta_meam[a][a], this->ctheta_meam[a][a], s1);
        Gam1 = (s1[0] * t11av + s1[1] * t21av + s1[2] * t31av) / (Z1 * Z1);
        G1 = G_gam(Gam1, this->ibar_meam[a], errorflag);
      }
      if (this->ibar_meam[b] <= 0)
        G2 = 1.0;
      else {
        get_shpfcn(this->lattce_meam[b][b], this->stheta_meam[b][b], this->ctheta_meam[b][b],  s2);
        Gam2 = (s2[0] * t12av + s2[1] * t22av + s2[2] * t32av) / (Z2 * Z2);
        G2 = G_gam(Gam2, this->ibar_meam[b], errorflag);
      }
      rho0_1 = this->rho0_meam[a] * Z1 * G1;
      rho0_2 = this->rho0_meam[b] * Z2 * G2;
    }
    Gam1 = (t11av * rho11 + t21av * rho21 + t31av * rho31);
    if (rho01 < 1.0e-14)
      Gam1 = 0.0;
    else
      Gam1 = Gam1 / (rho01 * rho01);

    Gam2 = (t12av * rho12 + t22av * rho22 + t32av * rho32);
    if (rho02 < 1.0e-14)
      Gam2 = 0.0;
    else
      Gam2 = Gam2 / (rho02 * rho02);

    G1 = G_gam(Gam1, this->ibar_meam[a], errorflag);
    G2 = G_gam(Gam2, this->ibar_meam[b], errorflag);
    if (this->mix_ref_t == 1) {
      rho_bkgd1 = rho0_1;
      rho_bkgd2 = rho0_2;
    } else {
      if (this->bkgd_dyn == 1) {
        rho_bkgd1 = this->rho0_meam[a] * Z1;
        rho_bkgd2 = this->rho0_meam[b] * Z2;
      } else {
        rho_bkgd1 = this->rho_ref_meam[a];
        rho_bkgd2 = this->rho_ref_meam[b];
      }
    }
    rhobar1 = rho01 / rho_bkgd1 * G1;
    rhobar2 = rho02 / rho_bkgd2 * G2;
  }

  // compute embedding functions, eqn I.5

  F1 = embedding(this->A_meam[a], this->mag_B_meam[a], this->Ec_meam[a][a], rhobar1, dF, ddF);
  F2 = embedding(this->A_meam[b], this->mag_B_meam[b], this->Ec_meam[b][b], rhobar2, dF, ddF);


  // compute Rose function, I.16
  Eu = erose(r, this->re_meam[a][b], this->alpha_meam[a][b], this->Ec_meam[a][b], this->repuls_meam[a][b],
             this->attrac_meam[a][b], this->erose_form);

  // calculate the pair energy
  if (this->lattce_meam[a][b] == C11) {
    latta = this->lattce_meam[a][a];
    if (latta == DIA) {
      phiaa = phi_meam(r, a, a);
      phi_m = (3 * Eu - F2 - 2 * F1 - 5 * phiaa) / Z12;
    } else {
      phibb = phi_meam(r, b, b);
      phi_m = (3 * Eu - F1 - 2 * F2 - 5 * phibb) / Z12;
    }
  } else if (this->lattce_meam[a][b] == L12) {
    phiaa = phi_meam(r, a, a);
    //       account for second neighbor a-a potential here...
    Z1nn = get_Zij(this->lattce_meam[a][a]);
    Z2nn = get_Zij2(this->lattce_meam[a][a], this->Cmin_meam[a][a][a],
             this->Cmax_meam[a][a][a], this->stheta_meam[a][b], arat, scrn);


    phiaa += phi_meam_series(scrn, Z1nn, Z2nn, a, a, r, arat);
    phi_m = Eu / 3.0 - F1 / 4.0 - F2 / 12.0 - phiaa;

  } else if (this->lattce_meam[a][b] == CH4) {
    phi_m = (5 * Eu - F1 - 4*F2)/4;

  } else if (this->lattce_meam[a][b] == ZIG) {
      if (a==b) {
        phi_m = (2 * Eu - F1 - F2) / Z12;
      } else{
        Z1 = get_Zij(this->lattce_meam[a][b]);
        Z2 = get_Zij2_b2nn(this->lattce_meam[a][b], this->Cmin_meam[a][a][b], this->Cmax_meam[a][a][b], scrn);
        b11s = -Z2/Z1*scrn;
        Z2 = get_Zij2_b2nn(this->lattce_meam[a][b], this->Cmin_meam[b][b][a], this->Cmax_meam[b][b][a], scrn2);
        b22s = -Z2/Z1*scrn2;

        phiaa = phi_meam(2.0*this->stheta_meam[a][b]*r, a, a);
        phibb = phi_meam(2.0*this->stheta_meam[a][b]*r, b, b);
        phi_m = (2.0*Eu - F1 - F2 + phiaa*b11s + phibb*b22s) / Z12;
      }

  } else if (this->lattce_meam[a][b] == TRI) {
      if (a==b) {
        phi_m = (3.0*Eu - 2.0*F1 - F2) / Z12;
     } else {
        Z1 = get_Zij(this->lattce_meam[a][b]);
        Z2 = get_Zij2_b2nn(this->lattce_meam[a][b], this->Cmin_meam[a][a][b], this->Cmax_meam[a][a][b], scrn);
        b11s = -Z2/Z1*scrn;
        phiaa = phi_meam(2.0*this->stheta_meam[a][b]*r, a, a);
        phi_m = (3.0*Eu - 2.0*F1 - F2 + phiaa*b11s) / Z12;
      }

  } else {
    // potential is computed from Rose function and embedding energy
    phi_m = (2 * Eu - F1 - F2) / Z12;
  }

  // if r = 0, just return 0
  if (iszero(r)) {
    phi_m = 0.0;
  }

  return phi_m;
}

//----------------------------------------------------------------------c
// Compute 2NN series terms for phi
//   To avoid nan values of phir due to rapid decrease of b2nn^n or/and
//   argument of phi_meam, i.e. r*arat^n, in some cases (3NN dia with low Cmin value)
//
double MAGMEAM::phi_meam_series(const double scrn, const int Z1, const int Z2, const int a, const int b,
                             const double r, const double arat)
{
  double phi_sum = 0.0;
  double b2nn, phi_val;
  if (scrn > 0.0) {
    b2nn = -Z2*scrn/Z1;
    for (int n = 1; n <= 10; n++) {
      phi_val = MathSpecial::powint(b2nn,n) * phi_meam(r * MathSpecial::powint(arat, n), a, b);
      if (iszero(phi_val)) {
        // once either term becomes zero at some point, all folliwng will also be zero
        // necessary to avoid numerical error (nan or infty) due to exponential decay in phi_meam
        break;
      }
      phi_sum += phi_val;
    }
  }
  return phi_sum;
}

//----------------------------------------------------------------------c
// Compute background density for reference structure of each element
void MAGMEAM::compute_reference_density()
{
  int a, Z, Z2, errorflag;
  double gam, Gbar, shp[3];
  double rho0, rho0_2nn, arat, scrn;

  // loop over element types
  for (a = 0; a < this->neltypes; a++) {
    Z = get_Zij(this->lattce_meam[a][a]);
    if (this->ibar_meam[a] <= 0)
      Gbar = 1.0;
    else {
      get_shpfcn(this->lattce_meam[a][a], this->stheta_meam[a][a], this->ctheta_meam[a][a], shp);
      gam = (this->t1_meam[a] * shp[0] + this->t2_meam[a] * shp[1] + this->t3_meam[a] * shp[2]) / (Z * Z);
      Gbar = G_gam(gam, this->ibar_meam[a], errorflag);
    }

    //     The zeroth order density in the reference structure, with
    //     equilibrium spacing, is just the number of first neighbors times
    //     the rho0_meam coefficient...
    rho0 = this->rho0_meam[a] * Z;

    //     ...unless we have unscreened second neighbors, in which case we
    //     add on the contribution from those (accounting for partial
    //     screening)
    if (this->nn2_meam[a][a] == 1) {
      Z2 = get_Zij2(this->lattce_meam[a][a], this->Cmin_meam[a][a][a],
               this->Cmax_meam[a][a][a], this->stheta_meam[a][a], arat, scrn);
      rho0_2nn = this->rho0_meam[a] * MathSpecial::fm_exp(-this->beta0_meam[a] * (arat - 1));
      rho0 = rho0 + Z2 * rho0_2nn * scrn;
    }

    this->rho_ref_meam[a] = rho0 * Gbar;
  }
}

//------------------------------------------------------------------------------c
// Average weighting factors for the reference structure
void MAGMEAM::get_tavref(double* t11av, double* t21av, double* t31av, double* t12av, double* t22av, double* t32av,
                      double t11, double t21, double t31, double t12, double t22, double t32, double r, int a,
                      int b, lattice_t latt)
{
  double rhoa01, rhoa02, a1, a2, rho01 /*,rho02*/;

  //     For ialloy = 2, no averaging is done
  if (this->ialloy == 2) {
    *t11av = t11;
    *t21av = t21;
    *t31av = t31;
    *t12av = t12;
    *t22av = t22;
    *t32av = t32;
  } else switch (latt)  {
    case FCC:
    case BCC:
    case DIA:
    case DIA3:
    case HCP:
    case B1:
    case DIM:
    case B2:
    case CH4:
    case LIN:
    case ZIG:
    case TRI:
      //     all neighbors are of the opposite type
      *t11av = t12;
      *t21av = t22;
      *t31av = t32;
      *t12av = t11;
      *t22av = t21;
      *t32av = t31;
      break;
    default:
      a1 = r / this->re_meam[a][a] - 1.0;
      a2 = r / this->re_meam[b][b] - 1.0;
      rhoa01 = this->rho0_meam[a] * MathSpecial::fm_exp(-this->beta0_meam[a] * a1);
      rhoa02 = this->rho0_meam[b] * MathSpecial::fm_exp(-this->beta0_meam[b] * a2);
      if (latt == L12) {
        rho01 = 8 * rhoa01 + 4 * rhoa02;
        *t11av = (8 * t11 * rhoa01 + 4 * t12 * rhoa02) / rho01;
        *t12av = t11;
        *t21av = (8 * t21 * rhoa01 + 4 * t22 * rhoa02) / rho01;
        *t22av = t21;
        *t31av = (8 * t31 * rhoa01 + 4 * t32 * rhoa02) / rho01;
        *t32av = t31;
      } else {
        //      call error('Lattice not defined in get_tavref.')
      }
  }
}

//------------------------------------------------------------------------------c
void MAGMEAM::get_sijk(double C, int i, int j, int k, double* sijk)
{
  double x;
  x = (C - this->Cmin_meam[i][j][k]) / (this->Cmax_meam[i][j][k] - this->Cmin_meam[i][j][k]);
  *sijk = fcut(x);
}

//------------------------------------------------------------------------------c
// Calculate density functions, assuming reference configuration
void MAGMEAM::get_densref(double r, int a, int b, double* rho01, double* rho11, double* rho21, double* rho31,
                       double* rho02, double* rho12, double* rho22, double* rho32)
{
  double a1, a2;
  double s[3];
  lattice_t lat;
  int Zij,Zij2nn;
  double rhoa01nn, rhoa02nn;
  double rhoa01, rhoa11, rhoa21, rhoa31;
  double rhoa02, rhoa12, rhoa22, rhoa32;
  double arat, scrn, denom;
  double C, s111, s112, s221, S11, S22;

  a1 = r / this->re_meam[a][a] - 1.0;
  a2 = r / this->re_meam[b][b] - 1.0;

  rhoa01 = this->rho0_meam[a] * MathSpecial::fm_exp(-this->beta0_meam[a] * a1);
  rhoa11 = this->rho0_meam[a] * MathSpecial::fm_exp(-this->beta1_meam[a] * a1);
  rhoa21 = this->rho0_meam[a] * MathSpecial::fm_exp(-this->beta2_meam[a] * a1);
  rhoa31 = this->rho0_meam[a] * MathSpecial::fm_exp(-this->beta3_meam[a] * a1);
  rhoa02 = this->rho0_meam[b] * MathSpecial::fm_exp(-this->beta0_meam[b] * a2);
  rhoa12 = this->rho0_meam[b] * MathSpecial::fm_exp(-this->beta1_meam[b] * a2);
  rhoa22 = this->rho0_meam[b] * MathSpecial::fm_exp(-this->beta2_meam[b] * a2);
  rhoa32 = this->rho0_meam[b] * MathSpecial::fm_exp(-this->beta3_meam[b] * a2);

  lat = this->lattce_meam[a][b];

  Zij = get_Zij(lat);

  *rho11 = 0.0;
  *rho21 = 0.0;
  *rho31 = 0.0;
  *rho12 = 0.0;
  *rho22 = 0.0;
  *rho32 = 0.0;

  switch (lat) {
    case FCC:
      *rho01 = 12.0 * rhoa02;
      *rho02 = 12.0 * rhoa01;
      break;
    case BCC:
      *rho01 = 8.0 * rhoa02;
      *rho02 = 8.0 * rhoa01;
      break;
    case B1:
      *rho01 = 6.0 * rhoa02;
      *rho02 = 6.0 * rhoa01;
      break;
    case DIA:
    case DIA3:
      *rho01 = 4.0 * rhoa02;
      *rho02 = 4.0 * rhoa01;
      *rho31 = 32.0 / 9.0 * rhoa32 * rhoa32;
      *rho32 = 32.0 / 9.0 * rhoa31 * rhoa31;
      break;
    case HCP:
      *rho01 = 12 * rhoa02;
      *rho02 = 12 * rhoa01;
      *rho31 = 1.0 / 3.0 * rhoa32 * rhoa32;
      *rho32 = 1.0 / 3.0 * rhoa31 * rhoa31;
      break;
    case DIM:
      get_shpfcn(DIM, 0, 0, s);
      *rho01 = rhoa02;
      *rho02 = rhoa01;
      *rho11 = s[0] * rhoa12 * rhoa12;
      *rho12 = s[0] * rhoa11 * rhoa11;
      *rho21 = s[1] * rhoa22 * rhoa22;
      *rho22 = s[1] * rhoa21 * rhoa21;
      *rho31 = s[2] * rhoa32 * rhoa32;
      *rho32 = s[2] * rhoa31 * rhoa31;
      break;
    case C11:
      *rho01 = rhoa01;
      *rho02 = rhoa02;
      *rho11 = rhoa11;
      *rho12 = rhoa12;
      *rho21 = rhoa21;
      *rho22 = rhoa22;
      *rho31 = rhoa31;
      *rho32 = rhoa32;
      break;
    case L12:
      *rho01 = 8 * rhoa01 + 4 * rhoa02;
      *rho02 = 12 * rhoa01;
      if (this->ialloy == 1) {
        *rho21 = 8. / 3. * MathSpecial::square(rhoa21 * this->t2_meam[a] - rhoa22 * this->t2_meam[b]);
        denom = 8 * rhoa01 * MathSpecial::square(this->t2_meam[a]) + 4 * rhoa02 * MathSpecial::square(this->t2_meam[b]);
        if (denom > 0.)
          *rho21 = *rho21 / denom * *rho01;
      } else
        *rho21 = 8. / 3. * (rhoa21 - rhoa22) * (rhoa21 - rhoa22);
      break;
    case B2:
      *rho01 = 8.0 * rhoa02;
      *rho02 = 8.0 * rhoa01;
      break;
    case CH4:
      *rho01 = 4.0 * rhoa02; //in assumption that 'a' represent carbon
      *rho02 = rhoa01;       //in assumption that 'b' represent hydrogen

      get_shpfcn(DIM, 0, 0, s); //H
      *rho12 = s[0] * rhoa11 * rhoa11;
      *rho22 = s[1] * rhoa21 * rhoa21;
      *rho32 = s[2] * rhoa31 * rhoa31;

      get_shpfcn(CH4, 0, 0, s); //C
      *rho11 = s[0] * rhoa12 * rhoa12;
      *rho21 = s[1] * rhoa22 * rhoa22;
      *rho31 = s[2] * rhoa32 * rhoa32;
      break;
    case LIN:
      *rho01 = rhoa02*Zij;
      *rho02 = rhoa01*Zij;

      get_shpfcn(LIN, this->stheta_meam[a][b], this->ctheta_meam[a][b], s);
      *rho12 = s[0] * rhoa11 * rhoa11;
      *rho22 = s[1] * rhoa21 * rhoa21;
      *rho32 = s[2] * rhoa31 * rhoa31;
      *rho11 = s[0] * rhoa12 * rhoa12;
      *rho21 = s[1] * rhoa22 * rhoa22;
      *rho31 = s[2] * rhoa32 * rhoa32;
      break;
    case ZIG:
      *rho01 = rhoa02*Zij;
      *rho02 = rhoa01*Zij;

      get_shpfcn(ZIG, this->stheta_meam[a][b], this->ctheta_meam[a][b], s);
      *rho12 = s[0] * rhoa11 * rhoa11;
      *rho22 = s[1] * rhoa21 * rhoa21;
      *rho32 = s[2] * rhoa31 * rhoa31;
      *rho11 = s[0] * rhoa12 * rhoa12;
      *rho21 = s[1] * rhoa22 * rhoa22;
      *rho31 = s[2] * rhoa32 * rhoa32;
      break;
    case TRI:
      *rho01 = rhoa02;
      *rho02 = rhoa01*Zij;

      get_shpfcn(TRI, this->stheta_meam[a][b], this->ctheta_meam[a][b], s);
      *rho12 = s[0] * rhoa11 * rhoa11;
      *rho22 = s[1] * rhoa21 * rhoa21;
      *rho32 = s[2] * rhoa31 * rhoa31;
      s[0] = 1.0;
      s[1] = 2.0/3.0;
      s[2] = 1.0 - 0.6*s[0];

      *rho11 = s[0] * rhoa12 * rhoa12;
      *rho21 = s[1] * rhoa22 * rhoa22;
      *rho31 = s[2] * rhoa32 * rhoa32;
      break;


    // default:
    //        call error('Lattice not defined in get_densref.')
  }

  if (this->nn2_meam[a][b] == 1) {


    Zij2nn = get_Zij2(lat, this->Cmin_meam[a][a][b], this->Cmax_meam[a][a][b],
                      this->stheta_meam[a][b], arat, scrn);

    a1 = arat * r / this->re_meam[a][a] - 1.0;
    a2 = arat * r / this->re_meam[b][b] - 1.0;

    rhoa01nn = this->rho0_meam[a] * MathSpecial::fm_exp(-this->beta0_meam[a] * a1);
    rhoa02nn = this->rho0_meam[b] * MathSpecial::fm_exp(-this->beta0_meam[b] * a2);

    if (lat == L12) {
      //     As usual, L12 thinks it's special; we need to be careful computing
      //     the screening functions
      C = 1.0;
      get_sijk(C, a, a, a, &s111);
      get_sijk(C, a, a, b, &s112);
      get_sijk(C, b, b, a, &s221);
      S11 = s111 * s111 * s112 * s112;
      S22 = s221 * s221 * s221 * s221;
      *rho01 = *rho01 + 6 * S11 * rhoa01nn;
      *rho02 = *rho02 + 6 * S22 * rhoa02nn;

    } else {
      //     For other cases, assume that second neighbor is of same type,
      //     first neighbor may be of different type

      *rho01 = *rho01 + Zij2nn * scrn * rhoa01nn;

      //     Assume Zij2nn and arat don't depend on order, but scrn might
      Zij2nn = get_Zij2(lat, this->Cmin_meam[b][b][a], this->Cmax_meam[b][b][a],
                        this->stheta_meam[a][b], arat, scrn);
      *rho02 = *rho02 + Zij2nn * scrn * rhoa02nn;
    }
  }
}

void MAGMEAM::interpolate_meam(int ind)
{
  int j;
  double drar;

  // map to coefficient space

  this->nrar = this->nr;
  drar = this->dr;
  this->rdrar = 1.0 / drar;

  // phir interp
  for (j = 0; j < this->nrar; j++) {
    this->phirar[ind][j] = this->phir[ind][j];
  }
  this->phirar1[ind][0] = this->phirar[ind][1] - this->phirar[ind][0];
  this->phirar1[ind][1] = 0.5 * (this->phirar[ind][2] - this->phirar[ind][0]);
  this->phirar1[ind][this->nrar - 2] =
    0.5 * (this->phirar[ind][this->nrar - 1] - this->phirar[ind][this->nrar - 3]);
  this->phirar1[ind][this->nrar - 1] = 0.0;
  for (j = 2; j < this->nrar - 2; j++) {
    this->phirar1[ind][j] = ((this->phirar[ind][j - 2] - this->phirar[ind][j + 2]) +
                             8.0 * (this->phirar[ind][j + 1] - this->phirar[ind][j - 1])) /
                            12.;
  }

  for (j = 0; j < this->nrar - 1; j++) {
    this->phirar2[ind][j] = 3.0 * (this->phirar[ind][j + 1] - this->phirar[ind][j]) -
                            2.0 * this->phirar1[ind][j] - this->phirar1[ind][j + 1];
    this->phirar3[ind][j] = this->phirar1[ind][j] + this->phirar1[ind][j + 1] -
                            2.0 * (this->phirar[ind][j + 1] - this->phirar[ind][j]);
  }
  this->phirar2[ind][this->nrar - 1] = 0.0;
  this->phirar3[ind][this->nrar - 1] = 0.0;

  for (j = 0; j < this->nrar; j++) {
    this->phirar4[ind][j] = this->phirar1[ind][j] / drar;
    this->phirar5[ind][j] = 2.0 * this->phirar2[ind][j] / drar;
    this->phirar6[ind][j] = 3.0 * this->phirar3[ind][j] / drar;
  }
}

//-----------------------------------------------------------------------
// compute magnetic pair potential for each pair of element types
//

void MAGMEAM::compute_pair_mag()
{
  double r;
  int j, a, b, nv2;
  double astar, frac, phizbl;
  int Z1, Z2;
  double arat, rarat, scrn, scrn2;
  double phiaa, phibb /*unused:,phitmp*/;
  double C, s111, s112, s221, S11, S22;

  // check for previously allocated arrays and free them
  if (this->magr != nullptr)
    memory->destroy(this->magr);
  if (this->magrar != nullptr)
    memory->destroy(this->magrar);
  if (this->magrar1 != nullptr)
    memory->destroy(this->magrar1);
  if (this->magrar2 != nullptr)
    memory->destroy(this->magrar2);
  if (this->magrar3 != nullptr)
    memory->destroy(this->magrar3);
  if (this->magrar4 != nullptr)
    memory->destroy(this->magrar4);
  if (this->magrar5 != nullptr)
    memory->destroy(this->magrar5);
  if (this->magrar6 != nullptr)
    memory->destroy(this->magrar6);

  // allocate memory for array that defines the potential
  memory->create(this->magr, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:magr");

  // allocate coeff memory

  memory->create(this->magrar, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:magrar");
  memory->create(this->magrar1, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:magrar1");
  memory->create(this->magrar2, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:magrar2");
  memory->create(this->magrar3, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:magrar3");
  memory->create(this->magrar4, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:magrar4");
  memory->create(this->magrar5, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:magrar5");
  memory->create(this->magrar6, (this->neltypes * (this->neltypes + 1)) / 2, this->nr, "pair:magrar6");

  // loop over pairs of element types
  nv2 = 0;
  for (a = 0; a < this->neltypes; a++) {
    for (b = a; b < this->neltypes; b++) {
      // loop over r values and compute
      for (j = 0; j < this->nr; j++) {
        r = j * this->dr;

        this->magr[nv2][j] = phi_mag(r, a, b);

      }

      // call interpolation
      interpolate_pair(nv2);

      nv2 = nv2 + 1;
    }
  }
}

//----------------------------------------------------------------------c
// Compute Magnetic pair potential for distance r, element types a and b
//
double MAGMEAM::phi_mag(double r, int a, int b)
{
  /*unused:double a1,a2,a12;*/
  double t11av, t21av, t31av, t001av, t12av, t22av, t32av, t002av;
  double tm01av, tm11av, tm21av, tm31av, tm41av, tm02av, tm12av, tm22av, tm32av, tm42av;
  double G1, G2, s1[3], s2[3], rho0_1, rho0_2;
  double Gam1, Gam2, Z1, Z2;
  double G1mag, G2mag, Gam1mag, Gam2mag;
  double rhobar1, rhobar2, F1, F2, dF, ddF;
  double rhobarm1, rhobarm2, F1m, F2m, dFm, ddFm;
  double rho01, rho11, rho21, rho31;
  double rho02, rho12, rho22, rho32;
  double rhom01, rhom11, rhom21, rhom31, rhom41, rhom001;
  double rhom02, rhom12, rhom22, rhom32, rhom42, rhom002;
  double scalfac, phiaa, phibb;
  double Eum;
  double arat, scrn, scrn2;
  int Z12, errorflag;
  int Z1nn, Z2nn;
  lattice_t latta /*unused:,lattb*/;
  double rho_bkgd1, rho_bkgd2;
  double b11s, b22s;

  double phi_m = 0.0;

  // Equation numbers below refer to:
  //   I. Huang et.al., Modelling simul. Mater. Sci. Eng. 3:615

  // get number of neighbors in the reference structure
  //   Nref[i][j] = # of i's neighbors of type j
  Z1 = get_Zij(this->lattce_meam[a][a]);
  Z2 = get_Zij(this->lattce_meam[b][b]);
  Z12 = get_Zij(this->mag_lattce[a][b]);

  get_densref(r, a, b, &rho01, &rho11, &rho21, &rho31, &rho02, &rho12, &rho22, &rho32);
  get_magdensref(r, a, b, &rhom01, &rhom11, &rhom21, &rhom31, &rhom41, &rhom001, &rhom02, &rhom12, &rhom22, &rhom32, &rhom42, &rhom002);

  // if densities are too small, numerical problems may result; just return zero
  if (rho01 <= 1e-14 && rho02 <= 1e-14)
    return 0.0;

  // calculate average weighting factors for the reference structure
  // average weighting factors for the reference structure, eqn. I.8
  get_tavref(&t11av, &t21av, &t31av, &t12av, &t22av, &t32av, this->t1_meam[a], this->t2_meam[a],
	     this->t3_meam[a], this->t1_meam[b], this->t2_meam[b], this->t3_meam[b], r, a, b,
	     this->lattce_meam[a][b]);
  get_tmagavref(&tm01av, &tm11av, &tm21av, &tm31av, &tm41av, &t001av, &tm02av, &tm12av, &tm22av, &tm32av, &tm42av, &t002av, this->mag_delrho0_meam[a],
	     this->mag_delrho1_meam[a], this->mag_delrho2_meam[a], this->mag_delrho3_meam[a], this->mag_delrho4_meam[a], this->mag_delrho00_meam[a],
	     this->mag_delrho0_meam[b], this->mag_delrho1_meam[b], this->mag_delrho2_meam[b], this->mag_delrho3_meam[b], this->mag_delrho4_meam[b], this->mag_delrho00_meam[b],
	     r, a, b, this->mag_lattce[a][b]);

    // use formalism developed in Huang's paper
    //     composition-dependent scaling, equation I.7
    //     If using mixing rule for t, apply to reference structure; else
    //     use precomputed values
  if (this->mix_ref_t == 1) {
    if (this->ibar_meam[a] <= 0)
      G1 = 1.0;
    else {
      get_shpfcn(this->lattce_meam[a][a], this->stheta_meam[a][a], this->ctheta_meam[a][a], s1);
      Gam1 = (s1[0] * t11av + s1[1] * t21av + s1[2] * t31av) / (Z1 * Z1);
      G1 = G_gam(Gam1, this->ibar_meam[a], errorflag);
    }
    if (this->ibar_meam[b] <= 0)
      G2 = 1.0;
    else {
      get_shpfcn(this->lattce_meam[b][b], this->stheta_meam[b][b], this->ctheta_meam[b][b],  s2);
      Gam2 = (s2[0] * t12av + s2[1] * t22av + s2[2] * t32av) / (Z2 * Z2);
      G2 = G_gam(Gam2, this->ibar_meam[b], errorflag);
    }
    rho0_1 = this->rho0_meam[a] * Z1 * G1;
    rho0_2 = this->rho0_meam[b] * Z2 * G2;
  }
  Gam1 = (t11av * rho11 + t21av * rho21 + t31av * rho31);
  if (rho01 < 1.0e-14)
    Gam1 = 0.0;
  else
    Gam1 = Gam1 / (rho01 * rho01);
  
  Gam2 = (t12av * rho12 + t22av * rho22 + t32av * rho32);
  if (rho02 < 1.0e-14)
    Gam2 = 0.0;
  else
    Gam2 = Gam2 / (rho02 * rho02);

  G1 = G_gam(Gam1, this->ibar_meam[a], errorflag);
  G2 = G_gam(Gam2, this->ibar_meam[b], errorflag);

  Gam1mag = (t11av * rho11 + t21av * rho21 + t31av * rho31 +
	     tm01av * rhom01 * rhom01 + tm11av * rhom11 * rhom11 +
	     tm21av * rhom21 * rhom21 + tm31av * rhom31 * rhom31 + tm41av * rhom41 * rhom41 + t001av * rhom01 * rhom001);

  if (rho01 < 1.0e-14)
    Gam1mag = 0.0;
  else
    Gam1mag = Gam1mag / (rho01 * rho01);
  
  Gam2mag = (t12av * rho12 + t22av * rho22 + t32av * rho32 +
	     tm02av * rhom02 * rhom02 + tm12av * rhom12 * rhom12 +
	     tm22av * rhom22 * rhom22 + tm32av * rhom32 * rhom32 + tm42av * rhom42 * rhom42 + t002av * rhom02 * rhom002);
  if (rho02 < 1.0e-14)
    Gam2mag = 0.0;
  else
    Gam2mag = Gam2mag / (rho02 * rho02);

  G1mag = G_gam(Gam1mag, this->ibar_meam[a], errorflag);
  G2mag = G_gam(Gam2mag, this->ibar_meam[b], errorflag);

  if (this->mix_ref_t == 1) {
    rho_bkgd1 = rho0_1;
    rho_bkgd2 = rho0_2;
  } else {
    if (this->bkgd_dyn == 1) {
      rho_bkgd1 = this->rho0_meam[a] * Z1;
      rho_bkgd2 = this->rho0_meam[b] * Z2;
    } else {
      rho_bkgd1 = this->rho_ref_meam[a];
      rho_bkgd2 = this->rho_ref_meam[b];
    }
  }
  rhobar1 = rho01 / rho_bkgd1 * G1;
  rhobar2 = rho02 / rho_bkgd2 * G2;

  rhobarm1 = rho01 / rho_bkgd1 * G1mag;
  rhobarm2 = rho02 / rho_bkgd2 * G2mag;
  
  // compute embedding functions, eqn I.5

  F1 = embedding(this->A_meam[a], this->mag_B_meam[a], this->Ec_meam[a][a], rhobar1, dF, ddF);
  F2 = embedding(this->A_meam[b], this->mag_B_meam[b], this->Ec_meam[b][b], rhobar2, dF, ddF);

  F1m = embedding(this->A_meam[a], this->mag_B_meam[a], this->Ec_meam[a][a], rhobarm1, dFm, ddFm);
  F2m = embedding(this->A_meam[b], this->mag_B_meam[b], this->Ec_meam[b][b], rhobarm2, dFm, ddFm);

  // compute Rose function, I.16
  Eum = erose(r, this->mag_re[a][b], this->mag_alpha[a][b], this->mag_Ec[a][b], this->mag_repuls[a][b], this->mag_attrac[a][b], this->erose_form);

    // potential is computed from Rose function and embedding energy
  phi_m = (2 * Eum - F1m - F2m + F1 + F2) / Z12;
  // if r = 0, just return 0
  if (iszero(r)) {
    phi_m = 0.0;
  }

  return phi_m;
}


//------------------------------------------------------------------------------c
// Average weighting factors for the reference structure
void MAGMEAM::get_tmagavref(double* t01av, double* t11av, double* t21av, double* t31av, double* t41av, double* t001av, double* t02av, double* t12av, double* t22av, double* t32av, double* t42av,
			    double* t002av, double t01, double t11, double t21, double t31, double t41, double t001, double t02, double t12, double t22, double t32, double t42, double t002,
			    double r, int a, int b, lattice_t latt)
{
  double rhoa01, rhoa02, a1, a2, rho01 /*,rho02*/;

  //     For ialloy = 2, no averaging is done
  if (this->ialloy == 2) {
    *t01av = t01;
    *t11av = t11;
    *t21av = t21;
    *t31av = t31;
    *t41av = t41;
    *t001av = t001;
    *t02av = t02;
    *t12av = t12;
    *t22av = t22;
    *t32av = t32;
    *t42av = t42;
    *t002av = t002;
  } else switch (latt)  {
    case MAGBCC:
      //     all neighbors are of the opposite type
      *t01av = t02;
      *t11av = t12;
      *t21av = t22;
      *t31av = t32;
      *t41av = t42;
      *t001av = t002;
      *t02av = t01;
      *t12av = t11;
      *t22av = t21;
      *t32av = t31;
      *t42av = t41;
      *t002av = t001;
      break;
    default:
      a1 = r / this->mag_re[a][a] - 1.0;
      a2 = r / this->mag_re[b][b] - 1.0;
      rhoa01 = this->rho0_meam[a] * MathSpecial::fm_exp(-this->beta0_meam[a] * a1);
      rhoa02 = this->rho0_meam[b] * MathSpecial::fm_exp(-this->beta0_meam[b] * a2);
  }
}

//------------------------------------------------------------------------------c
// Calculate density functions, assuming reference configuration
void MAGMEAM::get_magdensref(double r, int a, int b, double* rho01, double* rho11, double* rho21, double* rho31, double* rho41, double* rho001,
			     double* rho02, double* rho12, double* rho22, double* rho32, double* rho42, double* rho002)
{
  double a1, a2;
  double s[3];
  lattice_t lat;
  int Zij,Zij2nn;
  double rhoa001nn, rhoa002nn;
  double rhoa01, rhoa11, rhoa21, rhoa31, rhoa41, rhoa001;
  double rhoa02, rhoa12, rhoa22, rhoa32, rhoa42, rhoa002;
  double arat, scrn, denom;
  double C, s111, s112, s221, S11, S22;

  a1 = r / this->re_meam[a][a] - 1.0;
  a2 = r / this->re_meam[b][b] - 1.0;

  rhoa01 = this->rho0_meam[a] * MathSpecial::fm_exp(-this->mag_beta0_meam[a] * a1);
  rhoa001 = this->rho0_meam[a] * MathSpecial::fm_exp(-this->mag_beta00_meam[a] * a1);
  rhoa11 = this->rho0_meam[a] * MathSpecial::fm_exp(-this->mag_beta1_meam[a] * a1);
  rhoa21 = this->rho0_meam[a] * MathSpecial::fm_exp(-this->mag_beta2_meam[a] * a1);
  rhoa31 = this->rho0_meam[a] * MathSpecial::fm_exp(-this->mag_beta3_meam[a] * a1);
  rhoa41 = this->rho0_meam[a] * MathSpecial::fm_exp(-this->mag_beta4_meam[a] * a1);
  rhoa02 = this->rho0_meam[b] * MathSpecial::fm_exp(-this->mag_beta0_meam[b] * a2);
  rhoa002 = this->rho0_meam[b] * MathSpecial::fm_exp(-this->mag_beta00_meam[b] * a2);
  rhoa12 = this->rho0_meam[b] * MathSpecial::fm_exp(-this->mag_beta1_meam[b] * a2);
  rhoa22 = this->rho0_meam[b] * MathSpecial::fm_exp(-this->mag_beta2_meam[b] * a2);
  rhoa32 = this->rho0_meam[b] * MathSpecial::fm_exp(-this->mag_beta3_meam[b] * a2);
  rhoa42 = this->rho0_meam[b] * MathSpecial::fm_exp(-this->mag_beta4_meam[b] * a2);

  lat = this->mag_lattce[a][b];

  Zij = get_Zij(lat);

  *rho11 = 0.0;
  *rho21 = 0.0;
  *rho31 = 0.0;
  *rho41 = 0.0;
  *rho12 = 0.0;
  *rho22 = 0.0;
  *rho32 = 0.0;
  *rho42 = 0.0;

  switch (lat) {
    case MAGBCC:
      *rho01 = 8.0 * rhoa02;
      *rho02 = 8.0 * rhoa01;
      *rho001 = 8.0 * rhoa002;
      *rho002 = 8.0 * rhoa001;
      *rho41 = pow(512.0/135.0,0.5) * rhoa42;
      *rho42 = pow(512.0/135.0,0.5) * rhoa41;
      break;

    // default:
    //        call error('Lattice not defined in get_magdensref.')
  }
  if (this->nn2_meam[a][b] == 1) {


    Zij2nn = get_Zij2(lat, this->Cmin_magmeam[a][a][b], this->Cmax_magmeam[a][a][b],
                      this->stheta_meam[a][b], arat, scrn);

    a1 = arat * r / this->re_meam[a][a] - 1.0;
    a2 = arat * r / this->re_meam[b][b] - 1.0;

    rhoa001nn = this->rho0_meam[a] * MathSpecial::fm_exp(-this->mag_beta00_meam[a] * a1);
    rhoa002nn = this->rho0_meam[b] * MathSpecial::fm_exp(-this->mag_beta00_meam[b] * a2);

    *rho001 = *rho001 + Zij2nn * scrn * rhoa001nn;
    Zij2nn = get_Zij2(lat, this->Cmin_magmeam[b][b][a], this->Cmax_magmeam[b][b][a],
		      this->stheta_meam[a][b], arat, scrn);
    *rho002 = *rho002 + Zij2nn * scrn * rhoa002nn;
  }
}

void MAGMEAM::interpolate_pair(int ind)
{
  int j;
  double drar;

  // map to coefficient space

  this->nrar = this->nr;
  drar = this->dr;
  this->rdrar = 1.0 / drar;

  // magr interp
  for (j = 0; j < this->nrar; j++) {
    this->magrar[ind][j] = this->magr[ind][j];
  }
  this->magrar1[ind][0] = this->magrar[ind][1] - this->magrar[ind][0];
  this->magrar1[ind][1] = 0.5 * (this->magrar[ind][2] - this->magrar[ind][0]);
  this->magrar1[ind][this->nrar - 2] =
    0.5 * (this->magrar[ind][this->nrar - 1] - this->magrar[ind][this->nrar - 3]);
  this->magrar1[ind][this->nrar - 1] = 0.0;
  for (j = 2; j < this->nrar - 2; j++) {
    this->magrar1[ind][j] = ((this->magrar[ind][j - 2] - this->magrar[ind][j + 2]) +
                             8.0 * (this->magrar[ind][j + 1] - this->magrar[ind][j - 1])) /
                            12.;
  }

  for (j = 0; j < this->nrar - 1; j++) {
    this->magrar2[ind][j] = 3.0 * (this->magrar[ind][j + 1] - this->magrar[ind][j]) -
                            2.0 * this->magrar1[ind][j] - this->magrar1[ind][j + 1];
    this->magrar3[ind][j] = this->magrar1[ind][j] + this->magrar1[ind][j + 1] -
                            2.0 * (this->magrar[ind][j + 1] - this->magrar[ind][j]);
  }
  this->magrar2[ind][this->nrar - 1] = 0.0;
  this->magrar3[ind][this->nrar - 1] = 0.0;

  for (j = 0; j < this->nrar; j++) {
    this->magrar4[ind][j] = this->magrar1[ind][j] / drar;
    this->magrar5[ind][j] = 2.0 * this->magrar2[ind][j] / drar;
    this->magrar6[ind][j] = 3.0 * this->magrar3[ind][j] / drar;
  }
}

