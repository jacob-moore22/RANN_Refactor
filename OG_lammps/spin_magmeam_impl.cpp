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

/* ----------------------------------------------------------------------
   Contributing author: Sebastian Hütter (OvGU)
------------------------------------------------------------------------- */

#include "spin_magmeam.h"

#include "memory.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MAGMEAM::MAGMEAM(Memory* mem)
  : memory(mem)
{
  phir = phirar = phirar1 = phirar2 = phirar3 = phirar4 = phirar5 = phirar6 = nullptr;
  magr = magrar = magrar1 = magrar2 = magrar3 = magrar4 = magrar5 = magrar6 = nullptr;

  nmax = 0;
  rho = rho0 = rho1 = rho2 = rho3 = frhop = frhopp = nullptr;
  rho0m = rho1m = rho2m = rho3m = rho4m = rho00 = nullptr;
  gamma = dgamma1 = dgamma2 = dgamma3 = d2gamma2 = arho2b = arho2bm = arho4cm = arho0m = arho00 = arho00m = nullptr;
  arho1 = arho2 = arho3 = arho3b = t_ave = tsq_ave = nullptr;
  arho1m = arho2m = arho3m = arho3bm = arho4bm = arho4m = nullptr;
  maxneigh = 0;
  scrfcn = dscrfcn = fcpair = nullptr;
  scrfcnmag = dscrfcnmag = nullptr;
  scrfcnpair = dscrfcnpair = nullptr;

  imag = 0;
  neltypes = 0;
  for (int i = 0; i < maxelt; i++) {
    A_meam[i] = rho0_meam[i] = beta0_meam[i] =
      beta1_meam[i]= beta2_meam[i] = beta3_meam[i] =
      t0_meam[i] = t1_meam[i] = t2_meam[i] = t3_meam[i] =
      rho_ref_meam[i] = ibar_meam[i] = ielt_meam[i] =
      mag_B_meam[i] = mag_delrho0_meam[i] = mag_delrho1_meam[i] = mag_delrho2_meam[i] = mag_delrho3_meam[i] = mag_delrho4_meam[i] = 
      mag_delrho00_meam[i] = mag_beta0_meam[i] = mag_beta1_meam[i] = mag_beta2_meam[i] = mag_beta3_meam[i] = mag_beta4_meam[i] = mag_beta00_meam[i] = 0.0;
    for (int j = 0; j < maxelt; j++) {
      lattce_meam[i][j] = FCC;
      mag_lattce[i][j] = MAGBCC;
      Ec_meam[i][j] = re_meam[i][j] = alpha_meam[i][j] = delta_meam[i][j] = ebound_meam[i][j] = attrac_meam[i][j] = repuls_meam[i][j] = 0.0;
      nn2_meam[i][j] = zbl_meam[i][j] = eltind[i][j] = 0;
      mag_Ec[i][j] = mag_re[i][j] = mag_alpha[i][j] = mag_attrac[i][j] = mag_repuls[i][j] = 0.0;
    }
  }
}

MAGMEAM::~MAGMEAM()
{
  memory->destroy(this->phirar6);
  memory->destroy(this->phirar5);
  memory->destroy(this->phirar4);
  memory->destroy(this->phirar3);
  memory->destroy(this->phirar2);
  memory->destroy(this->phirar1);
  memory->destroy(this->phirar);
  memory->destroy(this->phir);

  memory->destroy(this->magrar6);
  memory->destroy(this->magrar5);
  memory->destroy(this->magrar4);
  memory->destroy(this->magrar3);
  memory->destroy(this->magrar2);
  memory->destroy(this->magrar1);
  memory->destroy(this->magrar);
  memory->destroy(this->magr);

  memory->destroy(this->rho);
  memory->destroy(this->rho0);
  memory->destroy(this->rho00);
  memory->destroy(this->rho0m);
  memory->destroy(this->rho1);
  memory->destroy(this->rho1m);
  memory->destroy(this->rho2);
  memory->destroy(this->rho2m);
  memory->destroy(this->rho3);
  memory->destroy(this->rho3m);
  memory->destroy(this->rho4m);
  memory->destroy(this->frhop);
  memory->destroy(this->frhopp);
  memory->destroy(this->gamma);
  memory->destroy(this->dgamma1);
  memory->destroy(this->dgamma2);
  memory->destroy(this->dgamma3);
  memory->destroy(this->d2gamma2);
  memory->destroy(this->arho0m);
  memory->destroy(this->arho00);
  memory->destroy(this->arho00m);
  memory->destroy(this->arho2b);
  memory->destroy(this->arho2bm);
  memory->destroy(this->arho4cm);

  memory->destroy(this->arho1);
  memory->destroy(this->arho1m);
  memory->destroy(this->arho2);
  memory->destroy(this->arho2m);
  memory->destroy(this->arho3);
  memory->destroy(this->arho3b);
  memory->destroy(this->arho3m);
  memory->destroy(this->arho3bm);
  memory->destroy(this->arho4m);
  memory->destroy(this->arho4bm);

  memory->destroy(this->t_ave);
  memory->destroy(this->tsq_ave);

  memory->destroy(this->scrfcn);
  memory->destroy(this->dscrfcn);
  memory->destroy(this->scrfcnmag);
  memory->destroy(this->dscrfcnmag);
  memory->destroy(this->scrfcnpair);
  memory->destroy(this->dscrfcnpair);
  memory->destroy(this->fcpair);
}
