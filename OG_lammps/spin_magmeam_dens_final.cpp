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
#include <iostream>
#include <iomanip>
#include <fstream>
using std::ofstream;
using namespace LAMMPS_NS;

void
MAGMEAM::meam_dens_final(int nlocal, int eflag_either, int eflag_global, int eflag_atom, double* eng_vdwl,
			 double* eatom, int /*ntype*/, int* type, int* fmap, double** scale, int& errorflag, int engflag)
{
  int i, elti;
  int m;
  double rhob, G, dG, ddG, Gbar, dGbar, ddGbar, gam, shp[3], Z;
  double denom, rho_bkgd, Fl;
  double scaleii;
  ofstream outdata;
  //     Complete the calculation of density
  
  for (i = 0; i < nlocal; i++) {
    elti = fmap[type[i]];
    if (elti >= 0) {
      scaleii = scale[type[i]][type[i]];

      rho0m[i] = arho0m[i] * arho0m[i];
      rho00[i] = arho00[i] * arho00m[i];
      rho1[i] = 0.0;
      rho1m[i] = 0.0;
      rho2[i] = -1.0 / 3.0 * arho2b[i] * arho2b[i];
      rho2m[i] = -1.0 / 3.0 * arho2bm[i] * arho2bm[i];
      rho3[i] = 0.0;
      rho3m[i] = 0.0; 
      rho4m[i] = 3.0 / 35.0 * arho4cm[i] * arho4cm[i]; 
      
      for (m = 0; m < 3; m++) {
        rho1[i] = rho1[i] + arho1[i][m] * arho1[i][m];
        rho1m[i] = rho1m[i] + arho1m[i][m] * arho1m[i][m];
        rho3[i] = rho3[i] - 3.0 / 5.0 * arho3b[i][m] * arho3b[i][m];
        rho3m[i] = rho3m[i] - 3.0 / 5.0 * arho3bm[i][m] * arho3bm[i][m];
      }
      for (m = 0; m < 6; m++) {
        rho2[i] = rho2[i] + this->v2D[m] * arho2[i][m] * arho2[i][m];
	rho2m[i] = rho2m[i] + this->v2D[m] * arho2m[i][m] * arho2m[i][m];
	rho4m[i] = rho4m[i] - 6.0 / 7.0 * this->v2D[m] * arho4bm[i][m] * arho4bm[i][m];
      }
      for (m = 0; m < 10; m++) {
        rho3[i] = rho3[i] + this->v3D[m] * arho3[i][m] * arho3[i][m];
        rho3m[i] = rho3m[i] + this->v3D[m] * arho3m[i][m] * arho3m[i][m];
      }
      for (m = 0; m < 15; m++) {
        rho4m[i] = rho4m[i] + this->v4D[m] * arho4m[i][m] * arho4m[i][m];
      }
      if (rho0[i] > 0.0) {
        if (this->ialloy == 1) {
          t_ave[i][0] = fdiv_zero(t_ave[i][0], tsq_ave[i][0]);
          t_ave[i][1] = fdiv_zero(t_ave[i][1], tsq_ave[i][1]);
          t_ave[i][2] = fdiv_zero(t_ave[i][2], tsq_ave[i][2]);
          t_ave[i][3] = fdiv_zero(t_ave[i][3], tsq_ave[i][3]);
          t_ave[i][4] = fdiv_zero(t_ave[i][4], tsq_ave[i][4]);
          t_ave[i][5] = fdiv_zero(t_ave[i][5], tsq_ave[i][5]);
          t_ave[i][6] = fdiv_zero(t_ave[i][6], tsq_ave[i][6]);
          t_ave[i][7] = fdiv_zero(t_ave[i][7], tsq_ave[i][7]);
          t_ave[i][8] = fdiv_zero(t_ave[i][8], tsq_ave[i][8]);
        } else if (this->ialloy == 2) {
          t_ave[i][0] = this->t1_meam[elti];
          t_ave[i][1] = this->t2_meam[elti];
          t_ave[i][2] = this->t3_meam[elti];
          t_ave[i][3] = this->mag_delrho0_meam[elti];
          t_ave[i][4] = this->mag_delrho1_meam[elti];
          t_ave[i][5] = this->mag_delrho2_meam[elti];
          t_ave[i][6] = this->mag_delrho3_meam[elti];
          t_ave[i][7] = this->mag_delrho4_meam[elti];
          t_ave[i][8] = this->mag_delrho00_meam[elti];
        } else {
          t_ave[i][0] = t_ave[i][0] / rho0[i];
          t_ave[i][1] = t_ave[i][1] / rho0[i];
          t_ave[i][2] = t_ave[i][2] / rho0[i];
          t_ave[i][3] = t_ave[i][3] / rho0[i];
          t_ave[i][4] = t_ave[i][4] / rho0[i];
          t_ave[i][5] = t_ave[i][5] / rho0[i];
          t_ave[i][6] = t_ave[i][6] / rho0[i];
          t_ave[i][7] = t_ave[i][7] / rho0[i];
          t_ave[i][8] = t_ave[i][8] / rho0[i];
        }
      }
      gamma[i] = t_ave[i][0] * rho1[i] + t_ave[i][1] * rho2[i] + t_ave[i][2] * rho3[i] +
	t_ave[i][3] * rho0m[i] + t_ave[i][4] * rho1m[i] + t_ave[i][5] * rho2m[i] + t_ave[i][6] * rho3m[i] + t_ave[i][7] * rho4m[i] + t_ave[i][8] * rho00[i];
      if (rho0[i] > 0.0) {
        gamma[i] = gamma[i] / (rho0[i] * rho0[i]);
      }

      Z = get_Zij(this->lattce_meam[elti][elti]);

      G = G_gam(gamma[i], this->ibar_meam[elti], errorflag);
      if (errorflag != 0)
        return;

      get_shpfcn(this->lattce_meam[elti][elti], this->stheta_meam[elti][elti], this->ctheta_meam[elti][elti], shp);

      if (this->ibar_meam[elti] <= 0) {
        Gbar = 1.0;
        dGbar = 0.0;
      } else {
        if (this->mix_ref_t == 1) {
          gam = (t_ave[i][0] * shp[0] + t_ave[i][1] * shp[1] + t_ave[i][2] * shp[2]) / (Z * Z);
        } else {
          gam = (this->t1_meam[elti] * shp[0] + this->t2_meam[elti] * shp[1] + this->t3_meam[elti] * shp[2]) /
                (Z * Z);
        }
        Gbar = G_gam(gam, this->ibar_meam[elti], errorflag);
      }
      rho[i] = rho0[i] * G;

      if (this->mix_ref_t == 1) {
        if (this->ibar_meam[elti] <= 0) {
          Gbar = 1.0;
          dGbar = 0.0;
        } else {
          gam = (t_ave[i][0] * shp[0] + t_ave[i][1] * shp[1] + t_ave[i][2] * shp[2]) / (Z * Z);
          Gbar = dG_gam(gam, this->ibar_meam[elti], dGbar, ddGbar);
        }
        rho_bkgd = this->rho0_meam[elti] * Z * Gbar;
      } else {
        if (this->bkgd_dyn == 1) {
          rho_bkgd = this->rho0_meam[elti] * Z;
        } else {
          rho_bkgd = this->rho_ref_meam[elti];
        }
      }
      rhob = rho[i] / rho_bkgd;
      denom = 1.0 / rho_bkgd;
      
      G = dG_gam(gamma[i], this->ibar_meam[elti], dG, ddG);

      dgamma1[i] = (G - 2 * dG * gamma[i]) * denom;

      if (!iszero(rho0[i])) {
        dgamma2[i] = (dG / rho0[i]) * denom;
	d2gamma2[i] = (ddG / rho0[i] / rho0[i] / rho0[i]) * denom;
      } else {
        dgamma2[i] = 0.0;
	d2gamma2[i] = 0.0;
      }

      //     dgamma3 is nonzero only if we are using the "mixed" rule for
      //     computing t in the reference system (which is not correct, but
      //     included for backward compatibility
      if (this->mix_ref_t == 1) {
        dgamma3[i] = rho0[i] * G * dGbar / (Gbar * Z * Z) * denom;
      } else {
        dgamma3[i] = 0.0;
      }

      Fl = embedding(this->A_meam[elti], this->mag_B_meam[elti], this->Ec_meam[elti][elti], rhob, frhop[i], frhopp[i]);
      
      if (eflag_either != 0) {
        Fl *= scaleii;
        if (eflag_global != 0) {
          *eng_vdwl = *eng_vdwl + Fl;
        }
        if (eflag_atom != 0) {
          eatom[i] = eatom[i] + Fl;
	}
      }
    }
  }
}

void
MAGMEAM::update_dens_final(int i, int /*ntype*/, int* type, int* fmap, double** scale, int& errorflag)
{
  int elti;
  int m;
  double rhob, G, dG, ddG, Gbar, dGbar, ddGbar, gam, shp[3], Z;
  double denom, rho_bkgd, Fl;
  double scaleii;

  //     Complete the calculation of density
  
  elti = fmap[type[i]];
  if (elti >= 0) {
    scaleii = scale[type[i]][type[i]];
    rho0m[i] = arho0m[i] * arho0m[i];
    rho00[i] = arho00[i] * arho00m[i];
    rho1[i] = 0.0;
    rho1m[i] = 0.0;
    rho2[i] = -1.0 / 3.0 * arho2b[i] * arho2b[i];
    rho2m[i] = -1.0 / 3.0 * arho2bm[i] * arho2bm[i];
    rho3[i] = 0.0;
    rho3m[i] = 0.0;
    rho4m[i] = 3.0 / 35.0 * arho4cm[i] * arho4cm[i]; 
      
    for (m = 0; m < 3; m++) {
      rho1[i] = rho1[i] + arho1[i][m] * arho1[i][m];
      rho1m[i] = rho1m[i] + arho1m[i][m] * arho1m[i][m];
      rho3[i] = rho3[i] - 3.0 / 5.0 * arho3b[i][m] * arho3b[i][m];
      rho3m[i] = rho3m[i] - 3.0 / 5.0 * arho3bm[i][m] * arho3bm[i][m];
    }
    for (m = 0; m < 6; m++) {
      rho2[i] = rho2[i] + this->v2D[m] * arho2[i][m] * arho2[i][m];
      rho2m[i] = rho2m[i] + this->v2D[m] * arho2m[i][m] * arho2m[i][m];
      rho4m[i] = rho4m[i] - 6.0 / 7.0 * this->v2D[m] * arho4bm[i][m] * arho4bm[i][m];
    }
    for (m = 0; m < 10; m++) {
      rho3[i] = rho3[i] + this->v3D[m] * arho3[i][m] * arho3[i][m];
      rho3m[i] = rho3m[i] + this->v3D[m] * arho3m[i][m] * arho3m[i][m];
    }
    for (m = 0; m < 15; m++) {
      rho4m[i] = rho4m[i] + this->v4D[m] * arho4m[i][m] * arho4m[i][m];
    }
    if (rho0[i] > 0.0) {
      if (this->ialloy == 1) {
	t_ave[i][0] = fdiv_zero(t_ave[i][0], tsq_ave[i][0]);
	t_ave[i][1] = fdiv_zero(t_ave[i][1], tsq_ave[i][1]);
	t_ave[i][2] = fdiv_zero(t_ave[i][2], tsq_ave[i][2]);
	t_ave[i][3] = fdiv_zero(t_ave[i][3], tsq_ave[i][3]);
	t_ave[i][4] = fdiv_zero(t_ave[i][4], tsq_ave[i][4]);
	t_ave[i][5] = fdiv_zero(t_ave[i][5], tsq_ave[i][5]);
	t_ave[i][6] = fdiv_zero(t_ave[i][6], tsq_ave[i][6]);
	t_ave[i][7] = fdiv_zero(t_ave[i][7], tsq_ave[i][7]);
	t_ave[i][8] = fdiv_zero(t_ave[i][8], tsq_ave[i][8]);
      } else if (this->ialloy == 2) {
	t_ave[i][0] = this->t1_meam[elti];
	t_ave[i][1] = this->t2_meam[elti];
	t_ave[i][2] = this->t3_meam[elti];
	t_ave[i][3] = this->mag_delrho0_meam[elti];
	t_ave[i][4] = this->mag_delrho1_meam[elti];
	t_ave[i][5] = this->mag_delrho2_meam[elti];
	t_ave[i][6] = this->mag_delrho3_meam[elti];
	t_ave[i][7] = this->mag_delrho4_meam[elti];
	t_ave[i][8] = this->mag_delrho00_meam[elti];
      } else {
	t_ave[i][0] = t_ave[i][0] / rho0[i];
	t_ave[i][1] = t_ave[i][1] / rho0[i];
	t_ave[i][2] = t_ave[i][2] / rho0[i];
	t_ave[i][3] = t_ave[i][3] / rho0[i];
	t_ave[i][4] = t_ave[i][4] / rho0[i];
	t_ave[i][5] = t_ave[i][5] / rho0[i];
	t_ave[i][6] = t_ave[i][6] / rho0[i];
	t_ave[i][7] = t_ave[i][7] / rho0[i];
	t_ave[i][8] = t_ave[i][8] / rho0[i];
      }
    }
    gamma[i] = t_ave[i][0] * rho1[i] + t_ave[i][1] * rho2[i] + t_ave[i][2] * rho3[i] +
      t_ave[i][3] * rho0m[i] + t_ave[i][4] * rho1m[i] + t_ave[i][5] * rho2m[i] + t_ave[i][6] * rho3m[i] + t_ave[i][7] * rho4m[i] + t_ave[i][8] * rho00[i];

    if (rho0[i] > 0.0) {
      gamma[i] = gamma[i] / (rho0[i] * rho0[i]);
    }

    Z = get_Zij(this->lattce_meam[elti][elti]);

    G = G_gam(gamma[i], this->ibar_meam[elti], errorflag);
    if (errorflag != 0)
      return;

    get_shpfcn(this->lattce_meam[elti][elti], this->stheta_meam[elti][elti], this->ctheta_meam[elti][elti], shp);

    if (this->ibar_meam[elti] <= 0) {
      Gbar = 1.0;
      dGbar = 0.0;
    } else {
      if (this->mix_ref_t == 1) {
	gam = (t_ave[i][0] * shp[0] + t_ave[i][1] * shp[1] + t_ave[i][2] * shp[2]) / (Z * Z);
      } else {
	gam = (this->t1_meam[elti] * shp[0] + this->t2_meam[elti] * shp[1] + this->t3_meam[elti] * shp[2]) /
	  (Z * Z);
      }
      Gbar = G_gam(gam, this->ibar_meam[elti], errorflag);
    }
    rho[i] = rho0[i] * G;

    if (this->mix_ref_t == 1) {
      if (this->ibar_meam[elti] <= 0) {
	Gbar = 1.0;
	dGbar = 0.0;
      } else {
	gam = (t_ave[i][0] * shp[0] + t_ave[i][1] * shp[1] + t_ave[i][2] * shp[2]) / (Z * Z);
	Gbar = dG_gam(gam, this->ibar_meam[elti], dGbar, ddGbar);
      }
      rho_bkgd = this->rho0_meam[elti] * Z * Gbar;
    } else {
      if (this->bkgd_dyn == 1) {
	rho_bkgd = this->rho0_meam[elti] * Z;
      } else {
	rho_bkgd = this->rho_ref_meam[elti];
      }
    }
    rhob = rho[i] / rho_bkgd;
    denom = 1.0 / rho_bkgd;

    G = dG_gam(gamma[i], this->ibar_meam[elti], dG, ddG);

    dgamma1[i] = (G - 2 * dG * gamma[i]) * denom;

    if (!iszero(rho0[i])) {
      dgamma2[i] = (dG / rho0[i]) * denom;
      d2gamma2[i] = (ddG / rho0[i] / rho0[i] / rho0[i]) * denom;
    } else {
      dgamma2[i] = 0.0;
      d2gamma2[i] = 0.0;
    }

    //     dgamma3 is nonzero only if we are using the "mixed" rule for
    //     computing t in the reference system (which is not correct, but
    //     included for backward compatibility
    if (this->mix_ref_t == 1) {
      dgamma3[i] = rho0[i] * G * dGbar / (Gbar * Z * Z) * denom;
    } else {
      dgamma3[i] = 0.0;
    }
    Fl = embedding(this->A_meam[elti], this->mag_B_meam[elti], this->Ec_meam[elti][elti], rhob, frhop[i], frhopp[i]);
  }
}
