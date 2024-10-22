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

using namespace LAMMPS_NS;

void
MAGMEAM::meam_dens_setup(int atom_nmax, int nall, int n_neigh)
{
  int i, j;

  // grow local arrays if necessary

  if (atom_nmax > nmax) {
    memory->destroy(rho);
    memory->destroy(rho0);
    memory->destroy(rho00);
    memory->destroy(rho0m);
    memory->destroy(rho1);
    memory->destroy(rho1m);
    memory->destroy(rho2);
    memory->destroy(rho2m);
    memory->destroy(rho3);
    memory->destroy(rho3m);
    memory->destroy(rho4m);
    memory->destroy(frhop);
    memory->destroy(frhopp);
    memory->destroy(gamma);
    memory->destroy(dgamma1);
    memory->destroy(dgamma2);
    memory->destroy(dgamma3);
    memory->destroy(d2gamma2);
    memory->destroy(arho0m);
    memory->destroy(arho00m);
    memory->destroy(arho00);
    memory->destroy(arho2b);
    memory->destroy(arho2bm);
    memory->destroy(arho1);
    memory->destroy(arho1m);
    memory->destroy(arho2);
    memory->destroy(arho2m);
    memory->destroy(arho3);
    memory->destroy(arho3m);
    memory->destroy(arho3b);
    memory->destroy(arho3bm);
    memory->destroy(arho4m);
    memory->destroy(arho4bm);
    memory->destroy(arho4cm);
    memory->destroy(t_ave);
    memory->destroy(tsq_ave);

    nmax = atom_nmax;

    memory->create(rho, nmax, "pair:rho");
    memory->create(rho0, nmax, "pair:rho0");
    memory->create(rho00, nmax, "pair:rho00");
    memory->create(rho0m, nmax, "pair:rho0m");
    memory->create(rho1, nmax, "pair:rho1");
    memory->create(rho1m, nmax, "pair:rho1m");
    memory->create(rho2, nmax, "pair:rho2");
    memory->create(rho2m, nmax, "pair:rho2m");
    memory->create(rho3, nmax, "pair:rho3");
    memory->create(rho3m, nmax, "pair:rho3m");
    memory->create(rho4m, nmax, "pair:rho4m");
    memory->create(frhop, nmax, "pair:frhop");
    memory->create(frhopp, nmax, "pair:frhop");
    memory->create(gamma, nmax, "pair:gamma");
    memory->create(dgamma1, nmax, "pair:dgamma1");
    memory->create(dgamma2, nmax, "pair:dgamma2");
    memory->create(dgamma3, nmax, "pair:dgamma3");
    memory->create(d2gamma2, nmax, "pair:d2gamma2");
    memory->create(arho0m, nmax, "pair:arho0m");
    memory->create(arho00m, nmax, "pair:arho00m");
    memory->create(arho00, nmax, "pair:arho00");
    memory->create(arho2b, nmax, "pair:arho2b");
    memory->create(arho2bm, nmax, "pair:arho2bm");
    memory->create(arho1, nmax, 3, "pair:arho1");
    memory->create(arho1m, nmax, 3, "pair:arho1m");
    memory->create(arho2, nmax, 6, "pair:arho2");
    memory->create(arho2m, nmax, 6, "pair:arho2m");
    memory->create(arho3, nmax, 10, "pair:arho3");
    memory->create(arho3b, nmax, 3, "pair:arho3b");
    memory->create(arho3m, nmax, 10, "pair:arho3m");
    memory->create(arho3bm, nmax, 3, "pair:arho3bm");
    memory->create(arho4m, nmax, 15, "pair:arho4m");
    memory->create(arho4bm, nmax, 6, "pair:arho4bm");
    memory->create(arho4cm, nmax, "pair:arho4cm");
    memory->create(t_ave, nmax, 9, "pair:t_ave");
    memory->create(tsq_ave, nmax, 9, "pair:tsq_ave");
  }

  if (n_neigh > maxneigh) {
    memory->destroy(scrfcn);
    memory->destroy(dscrfcn);
    memory->destroy(scrfcnmag);
    memory->destroy(dscrfcnmag);
    memory->destroy(scrfcnpair);
    memory->destroy(dscrfcnpair);
    memory->destroy(fcpair);
    maxneigh = n_neigh;
    memory->create(scrfcn, maxneigh, "pair:scrfcn");
    memory->create(dscrfcn, maxneigh, "pair:dscrfcn");
    memory->create(scrfcnmag, maxneigh, "pair:scrfcnmag");
    memory->create(dscrfcnmag, maxneigh, "pair:dscrfcnmag");
    memory->create(scrfcnpair, maxneigh, "pair:scrfcnpair");
    memory->create(dscrfcnpair, maxneigh, "pair:dscrfcnpair");
    memory->create(fcpair, maxneigh, "pair:fcpair");
  }

  // zero out local arrays

  for (i = 0; i < nall; i++) {
    rho0[i] = 0.0;
    arho0m[i] = 0.0;
    arho00m[i] = 0.0;
    arho00[i] = 0.0;
    arho2b[i] = 0.0;
    arho2bm[i] = 0.0;
    arho4cm[i] = 0.0;
    arho1[i][0] = arho1[i][1] = arho1[i][2] = 0.0;
    arho1m[i][0] = arho1m[i][1] = arho1m[i][2] = 0.0;
    for (j = 0; j < 6; j++){
      arho2[i][j] = 0.0;
      arho2m[i][j] = 0.0;
      arho4bm[i][j] = 0.0;
    }
    for (j = 0; j < 10; j++){
      arho3[i][j] = 0.0;
      arho3m[i][j] = 0.0;
    }
    arho3b[i][0] = arho3b[i][1] = arho3b[i][2] = 0.0;
    arho3bm[i][0] = arho3bm[i][1] = arho3bm[i][2] = 0.0;
    for (j = 0; j < 15; j++)
      arho4m[i][j] = 0.0;
    for (j = 0; j < 9; j++){
      t_ave[i][j] = 0.0;
      tsq_ave[i][j] = 0.0;
    }
  }
}

void
MAGMEAM::meam_dens_init(int i, int ntype, int* type, int* fmap, double** x, double** sp,
                     int numneigh, int* firstneigh,
                     int numneigh_full, int* firstneigh_full, int fnoffset)
{
  //     Compute screening function and derivatives
  getscreen(i, &scrfcn[fnoffset], &dscrfcn[fnoffset], &scrfcnmag[fnoffset], &dscrfcnmag[fnoffset],
	    &scrfcnpair[fnoffset], &dscrfcnpair[fnoffset], &fcpair[fnoffset], x, numneigh, firstneigh,
            numneigh_full, firstneigh_full, ntype, type, fmap);

  //     Calculate intermediate density terms to be communicated
  calc_rho1(i, ntype, type, fmap, x, sp, numneigh_full, firstneigh_full, &scrfcn[fnoffset], &scrfcnmag[fnoffset], &fcpair[fnoffset]);
}

// ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

void
MAGMEAM::getscreen(int i, double* scrfcn, double* dscrfcn, double* scrfcnmag, double* dscrfcnmag, double* scrfcnpair, double* dscrfcnpair, double* fcpair, double** x, int numneigh,
                int* firstneigh, int numneigh_full, int* firstneigh_full, int /*ntype*/, int* type, int* fmap)
{
  int jn, j, kn, k;
  int elti, eltj, eltk;
  double xitmp, yitmp, zitmp, delxij, delyij, delzij, rij2, rij;
  double xjtmp, yjtmp, zjtmp, delxik, delyik, delzik, rik2 /*,rik*/;
  double xktmp, yktmp, zktmp, delxjk, delyjk, delzjk, rjk2 /*,rjk*/;
  double xik, xjk, sij, fcij, sfcij, dfcij, sikj, dfikj, cikj;
  double sijmag, sfcijmag;
  double sijpair, sfcijpair;
  double Cmin, Cmax, delc, /*ebound,*/ a, coef1, coef2;
  double dCikj;
  double rnorm, fc, dfc, drinv;
  
  drinv = 1.0 / this->delr_meam;
  elti = fmap[type[i]];
  if (elti < 0) return;

  xitmp = x[i][0];
  yitmp = x[i][1];
  zitmp = x[i][2];

  for (jn = 0; jn < numneigh_full; jn++) {
    j = firstneigh_full[jn];

    eltj = fmap[type[j]];
    if (eltj < 0) continue;

    //     First compute screening function itself, sij
    xjtmp = x[j][0];
    yjtmp = x[j][1];
    zjtmp = x[j][2];
    delxij = xjtmp - xitmp;
    delyij = yjtmp - yitmp;
    delzij = zjtmp - zitmp;
    rij2 = delxij * delxij + delyij * delyij + delzij * delzij;

    if (rij2 > this->cutforcesq) {
      dscrfcn[jn] = 0.0;
      scrfcn[jn] = 0.0;
      dscrfcnmag[jn] = 0.0;
      scrfcnmag[jn] = 0.0;
      dscrfcnpair[jn] = 0.0;
      scrfcnpair[jn] = 0.0;
      fcpair[jn] = 0.0;
      continue;
    }

    const double rbound = this->ebound_meam[elti][eltj] * rij2;
    rij = sqrt(rij2);
    rnorm = (this->cutforce - rij) * drinv;
    sij = 1.0;
    sijmag = 1.0;
    sijpair = 1.0;
    //     if rjk2 > ebound*rijsq, atom k is definitely outside the ellipse
    for (kn = 0; kn < numneigh_full; kn++) {
      k = firstneigh_full[kn];
      if (k == j) continue;
      eltk = fmap[type[k]];
      if (eltk < 0) continue;

      xktmp = x[k][0];
      yktmp = x[k][1];
      zktmp = x[k][2];

      delxjk = xktmp - xjtmp;
      delyjk = yktmp - yjtmp;
      delzjk = zktmp - zjtmp;
      rjk2 = delxjk * delxjk + delyjk * delyjk + delzjk * delzjk;
      if (rjk2 > rbound) continue;

      delxik = xktmp - xitmp;
      delyik = yktmp - yitmp;
      delzik = zktmp - zitmp;
      rik2 = delxik * delxik + delyik * delyik + delzik * delzik;
      if (rik2 > rbound) continue;

      xik = rik2 / rij2;
      xjk = rjk2 / rij2;
      a = 1 - (xik - xjk) * (xik - xjk);
      //     if a < 0, then ellipse equation doesn't describe this case and
      //     atom k can't possibly screen i-j
      if (a <= 0.0) continue;

      cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
      Cmax = this->Cmax_meam[elti][eltj][eltk];
      Cmin = this->Cmin_meam[elti][eltj][eltk];
      if (cikj >= Cmax) continue;
      //     note that cikj may be slightly negative (within numerical
      //     tolerance) if atoms are colinear, so don't reject that case here
      //     (other negative cikj cases were handled by the test on "a" above)
      else if (cikj <= Cmin) {
        sij = 0.0;
        break;
      } else {
        delc = Cmax - Cmin;
        cikj = (cikj - Cmin) / delc;
        sikj = fcut(cikj);
      }
      sij *= sikj;
    }

    // Magnetic Screening Terms
    for (kn = 0; kn < numneigh_full; kn++) {
      k = firstneigh_full[kn];
      if (k == j) continue;
      eltk = fmap[type[k]];
      if (eltk < 0) continue;

      xktmp = x[k][0];
      yktmp = x[k][1];
      zktmp = x[k][2];

      delxjk = xktmp - xjtmp;
      delyjk = yktmp - yjtmp;
      delzjk = zktmp - zjtmp;
      rjk2 = delxjk * delxjk + delyjk * delyjk + delzjk * delzjk;
      if (rjk2 > rbound) continue;

      delxik = xktmp - xitmp;
      delyik = yktmp - yitmp;
      delzik = zktmp - zitmp;
      rik2 = delxik * delxik + delyik * delyik + delzik * delzik;
      if (rik2 > rbound) continue;

      xik = rik2 / rij2;
      xjk = rjk2 / rij2;
      a = 1 - (xik - xjk) * (xik - xjk);
      //     if a < 0, then ellipse equation doesn't describe this case and
      //     atom k can't possibly screen i-j
      if (a <= 0.0) continue;

      cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
      Cmax = this->Cmax_magmeam[elti][eltj][eltk];
      Cmin = this->Cmin_magmeam[elti][eltj][eltk];
      if (cikj >= Cmax) continue;
      //     note that cikj may be slightly negative (within numerical
      //     tolerance) if atoms are colinear, so don't reject that case here
      //     (other negative cikj cases were handled by the test on "a" above)
      else if (cikj <= Cmin) {
        sijmag = 0.0;
	sijpair = 0.0;
        break;
      } else {
        delc = Cmax - Cmin;
        cikj = (cikj - Cmin) / delc;
        sikj = fcut(cikj);
      }
      sijmag *= sikj;
    }
    
    fc = dfcut(rnorm, dfc);
    fcij = fc;
    dfcij = dfc * drinv;

    //     Now compute derivatives
    dscrfcn[jn] = 0.0;
    sfcij = sij * fcij;
    if (!iszero(sfcij) && !isone(sfcij)) {
      for (kn = 0; kn < numneigh_full; kn++) {
        k = firstneigh_full[kn];
        if (k == j) continue;
        eltk = fmap[type[k]];
        if (eltk < 0) continue;

        delxjk = x[k][0] - xjtmp;
        delyjk = x[k][1] - yjtmp;
        delzjk = x[k][2] - zjtmp;
        rjk2 = delxjk * delxjk + delyjk * delyjk + delzjk * delzjk;
        if (rjk2 > rbound) continue;

        delxik = x[k][0] - xitmp;
        delyik = x[k][1] - yitmp;
        delzik = x[k][2] - zitmp;
        rik2 = delxik * delxik + delyik * delyik + delzik * delzik;
        if (rik2 > rbound) continue;

        xik = rik2 / rij2;
        xjk = rjk2 / rij2;
        a = 1 - (xik - xjk) * (xik - xjk);
        //     if a < 0, then ellipse equation doesn't describe this case and
        //     atom k can't possibly screen i-j
        if (a <= 0.0) continue;

        cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
        Cmax = this->Cmax_meam[elti][eltj][eltk];
        Cmin = this->Cmin_meam[elti][eltj][eltk];
        if (cikj >= Cmax) {
          continue;
          //     Note that cikj may be slightly negative (within numerical
          //     tolerance) if atoms are colinear, so don't reject that case
          //     here
          //     (other negative cikj cases were handled by the test on "a"
          //     above)
          //     Note that we never have 0<cikj<Cmin here, else sij=0
          //     (rejected above)
        } else {
          delc = Cmax - Cmin;
          cikj = (cikj - Cmin) / delc;
          sikj = dfcut(cikj, dfikj);
          coef1 = dfikj / (delc * sikj);
          dCikj = dCfunc(rij2, rik2, rjk2);
          dscrfcn[jn] = dscrfcn[jn] + coef1 * dCikj;
        }
      }
      coef1 = sfcij;
      coef2 = sij * dfcij / rij;
      dscrfcn[jn] = dscrfcn[jn] * coef1 - coef2;
      //dscrfcn[jn] = 0.0;
    }

    //     Now compute magnetic derivatives
    dscrfcnmag[jn] = 0.0;
    sfcijmag = sijmag * fcij;
    if (!iszero(sfcijmag) && !isone(sfcijmag)) {
      for (kn = 0; kn < numneigh_full; kn++) {
        k = firstneigh_full[kn];
        if (k == j) continue;
        eltk = fmap[type[k]];
        if (eltk < 0) continue;

        delxjk = x[k][0] - xjtmp;
        delyjk = x[k][1] - yjtmp;
        delzjk = x[k][2] - zjtmp;
        rjk2 = delxjk * delxjk + delyjk * delyjk + delzjk * delzjk;
        if (rjk2 > rbound) continue;

        delxik = x[k][0] - xitmp;
        delyik = x[k][1] - yitmp;
        delzik = x[k][2] - zitmp;
        rik2 = delxik * delxik + delyik * delyik + delzik * delzik;
        if (rik2 > rbound) continue;

        xik = rik2 / rij2;
        xjk = rjk2 / rij2;
        a = 1 - (xik - xjk) * (xik - xjk);
        //     if a < 0, then ellipse equation doesn't describe this case and
        //     atom k can't possibly screen i-j
        if (a <= 0.0) continue;

        cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
        Cmax = this->Cmax_magmeam[elti][eltj][eltk];
        Cmin = this->Cmin_magmeam[elti][eltj][eltk];
        if (cikj >= Cmax) {
          continue;
          //     Note that cikj may be slightly negative (within numerical
          //     tolerance) if atoms are colinear, so don't reject that case
          //     here
          //     (other negative cikj cases were handled by the test on "a"
          //     above)
          //     Note that we never have 0<cikj<Cmin here, else sij=0
          //     (rejected above)
        } else {
          delc = Cmax - Cmin;
          cikj = (cikj - Cmin) / delc;
          sikj = dfcut(cikj, dfikj);
          coef1 = dfikj / (delc * sikj);
          dCikj = dCfunc(rij2, rik2, rjk2);
          dscrfcnmag[jn] = dscrfcnmag[jn] + coef1 * dCikj;
        }
      }
      coef1 = sfcijmag;
      coef2 = sijmag * dfcij / rij;
      dscrfcnmag[jn] = dscrfcnmag[jn] * coef1 - coef2;
      //dscrfcnmag[jn] = 0.;
    }

    scrfcn[jn] = sij;
    scrfcnmag[jn] = sijmag;
    scrfcnpair[jn] = sijmag;
    dscrfcnpair[jn] = dscrfcnmag[jn];
    fcpair[jn] = fcij;
    //scrfcn[jn] = 1;
    //scrfcnmag[jn] = 1;
    //fcpair[jn] = 1;

  }
}

// ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

void
MAGMEAM::calc_rho1(int i, int /*ntype*/, int* type, int* fmap, double** x, double** sp, int numneigh, int* firstneigh,
		   double* scrfcn, double* scrfcnmag, double* fcpair)
{
  int jn, j, m, n, p, q, elti, eltj;
  int nv2, nv3, nv4;
  double xtmp, ytmp, ztmp, delij[3], rij2, rij, sij, sijmag;
  double spi[3], spj[3];
  double aj, rhoa0j, rhoa1j, rhoa2j, rhoa3j, A1j, A2j, A3j, A1jm, A2jm, A3jm, A4jm;
  double delrhoa0j, delrhoa0mj, delrhoa00j, delrhoa1j, delrhoa2j, delrhoa3j, delrhoa4j;
  // double G,Gbar,gam,shp[3+1];
  double ro0i, ro0j;
  double sdot, smag;
  
  elti = fmap[type[i]];
  xtmp = x[i][0];
  ytmp = x[i][1];
  ztmp = x[i][2];

  spi[0] = sp[i][0];
  spi[1] = sp[i][1];
  spi[2] = sp[i][2];

  for (jn = 0; jn < numneigh; jn++) {
    if (!iszero(scrfcn[jn]) || !iszero(scrfcnmag[jn])) {
      j = firstneigh[jn];
      sij = scrfcn[jn] * fcpair[jn];
      sijmag = scrfcnmag[jn] *fcpair[jn];
      delij[0] = x[j][0] - xtmp;
      delij[1] = x[j][1] - ytmp;
      delij[2] = x[j][2] - ztmp;
      rij2 = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];

      spj[0] = sp[j][0];
      spj[1] = sp[j][1];
      spj[2] = sp[j][2];
      sdot = spi[0]*spj[0] + spi[1]*spj[1] + spi[2]*spj[2];
      smag = 0.5*(1 - sdot);
      //smag = sdot;

      if (rij2 < this->cutforcesq) {
        eltj = fmap[type[j]];
        rij = sqrt(rij2);
        aj = rij / this->re_meam[eltj][eltj] - 1.0;
	//        ro0i = this->rho0_meam[elti];
        ro0j = this->rho0_meam[eltj];
        rhoa0j = ro0j * MathSpecial::fm_exp(-this->beta0_meam[eltj] * aj) * sij;
        rhoa1j = ro0j * MathSpecial::fm_exp(-this->beta1_meam[eltj] * aj) * sij;
        rhoa2j = ro0j * MathSpecial::fm_exp(-this->beta2_meam[eltj] * aj) * sij;
        rhoa3j = ro0j * MathSpecial::fm_exp(-this->beta3_meam[eltj] * aj) * sij;

        delrhoa0j = ro0j * MathSpecial::fm_exp(-this->mag_beta0_meam[eltj] * aj) * sijmag * smag;
        delrhoa0mj = ro0j * MathSpecial::fm_exp(-this->mag_beta0_meam[eltj] * aj) * sijmag * smag * smag;
        delrhoa00j = ro0j * MathSpecial::fm_exp(-this->mag_beta00_meam[eltj] * aj) * sijmag;
        delrhoa1j = ro0j * MathSpecial::fm_exp(-this->mag_beta1_meam[eltj] * aj) * sijmag * smag;
	delrhoa2j = ro0j * MathSpecial::fm_exp(-this->mag_beta2_meam[eltj] * aj) * sijmag * smag;
	delrhoa3j = ro0j * MathSpecial::fm_exp(-this->mag_beta3_meam[eltj] * aj) * sijmag * smag;
	delrhoa4j = ro0j * MathSpecial::fm_exp(-this->mag_beta4_meam[eltj] * aj) * sijmag * smag;

	if (this->ialloy == 1) {
          rhoa1j = rhoa1j * this->t1_meam[eltj];
          rhoa2j = rhoa2j * this->t2_meam[eltj];
          rhoa3j = rhoa3j * this->t3_meam[eltj];

	  delrhoa0j = delrhoa0j * this->mag_delrho0_meam[eltj];
	  delrhoa00j = delrhoa00j * this->mag_delrho00_meam[eltj];
	  delrhoa1j = delrhoa1j * this->mag_delrho1_meam[eltj];
	  delrhoa2j = delrhoa2j * this->mag_delrho2_meam[eltj];
	  delrhoa3j = delrhoa3j * this->mag_delrho3_meam[eltj];
	  delrhoa4j = delrhoa4j * this->mag_delrho4_meam[eltj];
	}
        rho0[i] = rho0[i] + rhoa0j;
        // For ialloy = 2, use single-element value (not average)
        if (this->ialloy != 2) {
          t_ave[i][0] = t_ave[i][0] + this->t1_meam[eltj] * rhoa0j;
          t_ave[i][1] = t_ave[i][1] + this->t2_meam[eltj] * rhoa0j;
          t_ave[i][2] = t_ave[i][2] + this->t3_meam[eltj] * rhoa0j;
          t_ave[i][3] = t_ave[i][3] + this->mag_delrho0_meam[eltj] * rhoa0j;
          t_ave[i][4] = t_ave[i][4] + this->mag_delrho1_meam[eltj] * rhoa0j;
          t_ave[i][5] = t_ave[i][5] + this->mag_delrho2_meam[eltj] * rhoa0j;
          t_ave[i][6] = t_ave[i][6] + this->mag_delrho3_meam[eltj] * rhoa0j;
          t_ave[i][7] = t_ave[i][7] + this->mag_delrho4_meam[eltj] * rhoa0j;
          t_ave[i][8] = t_ave[i][8] + this->mag_delrho00_meam[eltj] * rhoa0j;
        }
        if (this->ialloy == 1) {
          tsq_ave[i][0] = tsq_ave[i][0] + this->t1_meam[eltj] * this->t1_meam[eltj] * rhoa0j;
          tsq_ave[i][1] = tsq_ave[i][1] + this->t2_meam[eltj] * this->t2_meam[eltj] * rhoa0j;
          tsq_ave[i][2] = tsq_ave[i][2] + this->t3_meam[eltj] * this->t3_meam[eltj] * rhoa0j;
          tsq_ave[i][3] = tsq_ave[i][3] + this->mag_delrho0_meam[eltj] * this->mag_delrho0_meam[eltj] * rhoa0j;
          tsq_ave[i][4] = tsq_ave[i][4] + this->mag_delrho1_meam[eltj] * this->mag_delrho1_meam[eltj] * rhoa0j;
          tsq_ave[i][5] = tsq_ave[i][5] + this->mag_delrho2_meam[eltj] * this->mag_delrho2_meam[eltj] * rhoa0j;
          tsq_ave[i][6] = tsq_ave[i][6] + this->mag_delrho3_meam[eltj] * this->mag_delrho3_meam[eltj] * rhoa0j;
          tsq_ave[i][7] = tsq_ave[i][7] + this->mag_delrho4_meam[eltj] * this->mag_delrho4_meam[eltj] * rhoa0j;
          tsq_ave[i][8] = tsq_ave[i][8] + this->mag_delrho00_meam[eltj] * this->mag_delrho00_meam[eltj] * rhoa0j;
        }
        arho0m[i] = arho0m[i] + delrhoa0j;
        arho00[i] = arho00[i] + delrhoa00j;
        arho00m[i] = arho00m[i] + delrhoa0mj;
        arho2b[i] = arho2b[i] + rhoa2j;
	arho2bm[i] = arho2bm[i] + delrhoa2j;
	arho4cm[i] = arho4cm[i] + delrhoa4j;

        A1j = rhoa1j / rij;
        A1jm = delrhoa1j / rij;
        A2j = rhoa2j / rij2;
	A2jm = delrhoa2j / rij2;
        A3j = rhoa3j / (rij2 * rij);
	A3jm = delrhoa3j / (rij2 * rij);
	A4jm = delrhoa4j / rij2;
        nv2 = 0;
        nv3 = 0;
	nv4 = 0;
        for (m = 0; m < 3; m++) {
          arho1[i][m] = arho1[i][m] + A1j * delij[m];
          arho1m[i][m] = arho1m[i][m] + A1jm * delij[m];
          arho3b[i][m] = arho3b[i][m] + rhoa3j * delij[m] / rij;
          arho3bm[i][m] = arho3bm[i][m] + delrhoa3j * delij[m] / rij;
          for (n = m; n < 3; n++) {
            arho2[i][nv2] = arho2[i][nv2] + A2j * delij[m] * delij[n];
            arho2m[i][nv2] = arho2m[i][nv2] + A2jm * delij[m] * delij[n];
            arho4bm[i][nv2] = arho4bm[i][nv2] + A4jm * delij[m] * delij[n];
            nv2 = nv2 + 1;
            for (p = n; p < 3; p++) {
              arho3[i][nv3] = arho3[i][nv3] + A3j * delij[m] * delij[n] * delij[p];
              arho3m[i][nv3] = arho3m[i][nv3] + A3jm * delij[m] * delij[n] * delij[p];
              nv3 = nv3 + 1;
	      for (q = p; q < 3; q++) {
		arho4m[i][nv4] = arho4m[i][nv4] + A4jm / rij2 * delij[m] * delij[n] * delij[p] * delij[q];
		nv4 = nv4 + 1;
	      }
            }
          }
        }
      }
    }
  }
}

void
MAGMEAM::meam_dens_update(int i, int ntype, int* type, int* fmap, double** x, double** sp,
                     int numneigh, int* firstneigh,
                     int numneigh_full, int* firstneigh_full, int fnoffset)
{
  //     Calculate intermediate density terms to be communicated
  update_rho1(i, ntype, type, fmap, x, sp, numneigh_full, firstneigh_full, &scrfcn[fnoffset], &scrfcnmag[fnoffset], &fcpair[fnoffset]);
}

void
MAGMEAM::update_rho1(int i, int /*ntype*/, int* type, int* fmap, double** x, double** sp, int numneigh, int* firstneigh, double* scrfcn, double* scrfcnmag, double* fcpair)
{
  int jn, j, m, n, p, q, elti, eltj;
  int nv2, nv3, nv4;
  double xtmp, ytmp, ztmp, delij[3], rij2, rij, sij, sijmag;
  double spi[3], spj[3];
  double aj, rhoa0j, rhoa1j, rhoa2j, rhoa3j, A1j, A2j, A3j, A1jm, A2jm, A3jm, A4jm;
  double delrhoa0j, delrhoa0mj, delrhoa00j, delrhoa1j, delrhoa2j, delrhoa3j, delrhoa4j;
  // double G,Gbar,gam,shp[3+1];
  double ro0i, ro0j;
  double sdot, smag;
  
  // Zero out arrays

  rho0[i] = 0.0;
  arho0m[i] = 0.0;
  arho00[i] = 0.0;
  arho00m[i] = 0.0;
  arho2b[i] = 0.0;
  arho2bm[i] = 0.0;
  arho4cm[i] = 0.0;
  arho1[i][0] = arho1[i][1] = arho1[i][2] = 0.0;
  arho1m[i][0] = arho1m[i][1] = arho1m[i][2] = 0.0;
  for (j = 0; j < 6; j++){
    arho2[i][j] = 0.0;
    arho2m[i][j] = 0.0;
    arho4bm[i][j] = 0.0;
  }
  for (j = 0; j < 10; j++){
    arho3[i][j] = 0.0;
    arho3m[i][j] = 0.0;
  }
  arho3b[i][0] = arho3b[i][1] = arho3b[i][2] = 0.0;
  arho3bm[i][0] = arho3bm[i][1] = arho3bm[i][2] = 0.0;
  for (j = 0; j < 15; j++)
    arho4m[i][j] = 0.0;
  for (j = 0; j < 9; j++){
    t_ave[i][j] = 0.0;
    tsq_ave[i][j] = 0.0;
  }

  elti = fmap[type[i]];
  xtmp = x[i][0];
  ytmp = x[i][1];
  ztmp = x[i][2];

  spi[0] = sp[i][0];
  spi[1] = sp[i][1];
  spi[2] = sp[i][2];

  for (jn = 0; jn < numneigh; jn++) {
    if (!iszero(scrfcn[jn]) || !iszero(scrfcnmag[jn])) {
      j = firstneigh[jn];
      sij = scrfcn[jn] * fcpair[jn];
      sijmag = scrfcnmag[jn] *fcpair[jn];
      delij[0] = x[j][0] - xtmp;
      delij[1] = x[j][1] - ytmp;
      delij[2] = x[j][2] - ztmp;
      rij2 = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];

      spj[0] = sp[j][0];
      spj[1] = sp[j][1];
      spj[2] = sp[j][2];
      sdot = spi[0]*spj[0] + spi[1]*spj[1] + spi[2]*spj[2];
      smag = 0.5*(1 - sdot);
      //smag = sdot;


      if (rij2 < this->cutforcesq) {
        eltj = fmap[type[j]];
        rij = sqrt(rij2);
        //ai = rij / this->re_meam[elti][elti] - 1.0;
        aj = rij / this->re_meam[eltj][eltj] - 1.0;
	//ro0i = this->rho0_meam[elti];
        ro0j = this->rho0_meam[eltj];
        rhoa0j = ro0j * MathSpecial::fm_exp(-this->beta0_meam[eltj] * aj) * sij;
        rhoa1j = ro0j * MathSpecial::fm_exp(-this->beta1_meam[eltj] * aj) * sij;
        rhoa2j = ro0j * MathSpecial::fm_exp(-this->beta2_meam[eltj] * aj) * sij;
        rhoa3j = ro0j * MathSpecial::fm_exp(-this->beta3_meam[eltj] * aj) * sij;
        //rhoa0i = ro0i * MathSpecial::fm_exp(-this->beta0_meam[elti] * ai) * sij;
        //rhoa1i = ro0i * MathSpecial::fm_exp(-this->beta1_meam[elti] * ai) * sij;
        //rhoa2i = ro0i * MathSpecial::fm_exp(-this->beta2_meam[elti] * ai) * sij;
        //rhoa3i = ro0i * MathSpecial::fm_exp(-this->beta3_meam[elti] * ai) * sij;

        delrhoa0j = ro0j * MathSpecial::fm_exp(-this->mag_beta0_meam[eltj] * aj) * sijmag * smag;
        delrhoa0mj = ro0j * MathSpecial::fm_exp(-this->mag_beta0_meam[eltj] * aj) * sijmag * smag * smag;
        delrhoa00j = ro0j * MathSpecial::fm_exp(-this->mag_beta00_meam[eltj] * aj) * sijmag;
	delrhoa1j = ro0j * MathSpecial::fm_exp(-this->mag_beta1_meam[eltj] * aj) * sijmag * smag;
	delrhoa2j = ro0j * MathSpecial::fm_exp(-this->mag_beta2_meam[eltj] * aj) * sijmag * smag;
	delrhoa3j = ro0j * MathSpecial::fm_exp(-this->mag_beta3_meam[eltj] * aj) * sijmag * smag;
	delrhoa4j = ro0j * MathSpecial::fm_exp(-this->mag_beta4_meam[eltj] * aj) * sijmag * smag;
        //delrhoa0i = ro0i * MathSpecial::fm_exp(-this->mag_beta0_meam[elti] * ai) * sijmag * smag;
	//delrhoa1i = ro0i * MathSpecial::fm_exp(-this->mag_beta1_meam[elti] * ai) * sijmag * smag;
	//delrhoa2i = ro0i * MathSpecial::fm_exp(-this->mag_beta2_meam[elti] * ai) * sijmag * smag;
	//delrhoa3i = ro0i * MathSpecial::fm_exp(-this->mag_beta3_meam[elti] * ai) * sijmag * smag;
	//delrhoa4i = ro0i * MathSpecial::fm_exp(-this->mag_beta4_meam[elti] * ai) * sijmag * smag;

	if (this->ialloy == 1) {
          rhoa1j = rhoa1j * this->t1_meam[eltj];
          rhoa2j = rhoa2j * this->t2_meam[eltj];
          rhoa3j = rhoa3j * this->t3_meam[eltj];
          //rhoa1i = rhoa1i * this->t1_meam[elti];
          //rhoa2i = rhoa2i * this->t2_meam[elti];
          //rhoa3i = rhoa3i * this->t3_meam[elti];

	  delrhoa0j = delrhoa0j * this->mag_delrho0_meam[eltj];
	  delrhoa00j = delrhoa00j * this->mag_delrho00_meam[eltj];
	  delrhoa1j = delrhoa1j * this->mag_delrho1_meam[eltj];
	  delrhoa2j = delrhoa2j * this->mag_delrho2_meam[eltj];
	  delrhoa3j = delrhoa3j * this->mag_delrho3_meam[eltj];
	  delrhoa4j = delrhoa4j * this->mag_delrho4_meam[eltj];
	  //delrhoa0i = delrhoa0i * this->mag_delrho0_meam[elti];
	  //delrhoa1i = delrhoa1i * this->mag_delrho1_meam[elti];
	  //delrhoa2i = delrhoa2i * this->mag_delrho2_meam[elti];
	  //delrhoa3i = delrhoa3i * this->mag_delrho3_meam[elti];
	  //delrhoa4i = delrhoa4i * this->mag_delrho4_meam[elti];
	}
        rho0[i] = rho0[i] + rhoa0j;
        //rho0[j] = rho0[j] + rhoa0i;
        // For ialloy = 2, use single-element value (not average)
        if (this->ialloy != 2) {
          t_ave[i][0] = t_ave[i][0] + this->t1_meam[eltj] * rhoa0j;
          t_ave[i][1] = t_ave[i][1] + this->t2_meam[eltj] * rhoa0j;
          t_ave[i][2] = t_ave[i][2] + this->t3_meam[eltj] * rhoa0j;
          t_ave[i][3] = t_ave[i][3] + this->mag_delrho0_meam[eltj] * rhoa0j;
          t_ave[i][4] = t_ave[i][4] + this->mag_delrho1_meam[eltj] * rhoa0j;
          t_ave[i][5] = t_ave[i][5] + this->mag_delrho2_meam[eltj] * rhoa0j;
          t_ave[i][6] = t_ave[i][6] + this->mag_delrho3_meam[eltj] * rhoa0j;
          t_ave[i][7] = t_ave[i][7] + this->mag_delrho4_meam[eltj] * rhoa0j;
          t_ave[i][8] = t_ave[i][8] + this->mag_delrho00_meam[eltj] * rhoa0j;
          //t_ave[j][0] = t_ave[j][0] + this->t1_meam[elti] * rhoa0i;
          //t_ave[j][1] = t_ave[j][1] + this->t2_meam[elti] * rhoa0i;
          //t_ave[j][2] = t_ave[j][2] + this->t3_meam[elti] * rhoa0i;
          //t_ave[j][3] = t_ave[j][3] + this->mag_delrho0_meam[elti] * rhoa0i;
          //t_ave[j][4] = t_ave[j][4] + this->mag_delrho1_meam[elti] * rhoa0i;
          //t_ave[j][5] = t_ave[j][5] + this->mag_delrho2_meam[elti] * rhoa0i;
          //t_ave[j][6] = t_ave[j][6] + this->mag_delrho3_meam[elti] * rhoa0i;
          //t_ave[j][7] = t_ave[j][7] + this->mag_delrho4_meam[elti] * rhoa0i;
          //t_ave[j][8] = t_ave[j][8] + this->mag_delrho00_meam[elti] * rhoa0i;
        }
        if (this->ialloy == 1) {
          tsq_ave[i][0] = tsq_ave[i][0] + this->t1_meam[eltj] * this->t1_meam[eltj] * rhoa0j;
          tsq_ave[i][1] = tsq_ave[i][1] + this->t2_meam[eltj] * this->t2_meam[eltj] * rhoa0j;
          tsq_ave[i][2] = tsq_ave[i][2] + this->t3_meam[eltj] * this->t3_meam[eltj] * rhoa0j;
          tsq_ave[i][3] = tsq_ave[i][3] + this->mag_delrho0_meam[eltj] * this->mag_delrho0_meam[eltj] * rhoa0j;
          tsq_ave[i][4] = tsq_ave[i][4] + this->mag_delrho1_meam[eltj] * this->mag_delrho1_meam[eltj] * rhoa0j;
          tsq_ave[i][5] = tsq_ave[i][5] + this->mag_delrho2_meam[eltj] * this->mag_delrho2_meam[eltj] * rhoa0j;
          tsq_ave[i][6] = tsq_ave[i][6] + this->mag_delrho3_meam[eltj] * this->mag_delrho3_meam[eltj] * rhoa0j;
          tsq_ave[i][7] = tsq_ave[i][7] + this->mag_delrho4_meam[eltj] * this->mag_delrho4_meam[eltj] * rhoa0j;
          tsq_ave[i][8] = tsq_ave[i][8] + this->mag_delrho00_meam[eltj] * this->mag_delrho00_meam[eltj] * rhoa0j;
          //tsq_ave[j][0] = tsq_ave[j][0] + this->t1_meam[elti] * this->t1_meam[elti] * rhoa0i;
          //tsq_ave[j][1] = tsq_ave[j][1] + this->t2_meam[elti] * this->t2_meam[elti] * rhoa0i;
          //tsq_ave[j][2] = tsq_ave[j][2] + this->t3_meam[elti] * this->t3_meam[elti] * rhoa0i;
          //tsq_ave[j][3] = tsq_ave[j][3] + this->mag_delrho0_meam[elti] * this->mag_delrho0_meam[elti] * rhoa0i;
          //tsq_ave[j][4] = tsq_ave[j][4] + this->mag_delrho1_meam[elti] * this->mag_delrho1_meam[elti] * rhoa0i;
          //tsq_ave[j][5] = tsq_ave[j][5] + this->mag_delrho2_meam[elti] * this->mag_delrho2_meam[elti] * rhoa0i;
          //tsq_ave[j][6] = tsq_ave[j][6] + this->mag_delrho3_meam[elti] * this->mag_delrho3_meam[elti] * rhoa0i;
          //tsq_ave[j][7] = tsq_ave[j][7] + this->mag_delrho4_meam[elti] * this->mag_delrho4_meam[elti] * rhoa0i;
          //tsq_ave[j][8] = tsq_ave[j][8] + this->mag_delrho00_meam[elti] * this->mag_delrho00_meam[elti] * rhoa0i;
        }
        arho0m[i] = arho0m[i] + delrhoa0j;
        arho00[i] = arho00[i] + delrhoa00j;
        arho00m[i] = arho00m[i] + delrhoa0mj;
        arho2b[i] = arho2b[i] + rhoa2j;
	arho2bm[i] = arho2bm[i] + delrhoa2j;
	arho4cm[i] = arho4cm[i] + delrhoa4j;
        //arho0m[j] = arho0m[j] + delrhoa0i;
        //arho2b[j] = arho2b[j] + rhoa2i;
	//arho2bm[j] = arho2bm[j] + delrhoa2i;
	//arho4cm[j] = arho4cm[j] + delrhoa4i;

        A1j = rhoa1j / rij;
        A1jm = delrhoa1j / rij;
        A2j = rhoa2j / rij2;
	A2jm = delrhoa2j / rij2;
        A3j = rhoa3j / (rij2 * rij);
	A3jm = delrhoa3j / (rij2 * rij);
	A4jm = delrhoa4j / rij2;
        //A1i = rhoa1i / rij;
        //A1im = delrhoa1i / rij;
        //A2i = rhoa2i / rij2;
	//A2im = delrhoa2i / rij2;
        //A3i = rhoa3i / (rij2 * rij);
	//A3im = delrhoa3i / (rij2 * rij);
	//A4im = delrhoa4i / rij2;
        nv2 = 0;
        nv3 = 0;
	nv4 = 0;
        for (m = 0; m < 3; m++) {
          arho1[i][m] = arho1[i][m] + A1j * delij[m];
          arho1m[i][m] = arho1m[i][m] + A1jm * delij[m];
          arho3b[i][m] = arho3b[i][m] + rhoa3j * delij[m] / rij;
          arho3bm[i][m] = arho3bm[i][m] + delrhoa3j * delij[m] / rij;
          //arho1[j][m] = arho1[j][m] + A1i * delij[m];
          //arho1m[j][m] = arho1m[j][m] + A1im * delij[m];
          //arho3b[j][m] = arho3b[j][m] + rhoa3i * delij[m] / rij;
          //arho3bm[j][m] = arho3bm[j][m] + delrhoa3i * delij[m] / rij;
          for (n = m; n < 3; n++) {
            arho2[i][nv2] = arho2[i][nv2] + A2j * delij[m] * delij[n];
            arho2m[i][nv2] = arho2m[i][nv2] + A2jm * delij[m] * delij[n];
            arho4bm[i][nv2] = arho4bm[i][nv2] + A4jm * delij[m] * delij[n];
            //arho2[j][nv2] = arho2[j][nv2] + A2i * delij[m] * delij[n];
            //arho2m[j][nv2] = arho2m[j][nv2] + A2im * delij[m] * delij[n];
            //arho4bm[j][nv2] = arho4bm[j][nv2] + A4im * delij[m] * delij[n];
            nv2 = nv2 + 1;
            for (p = n; p < 3; p++) {
              arho3[i][nv3] = arho3[i][nv3] + A3j * delij[m] * delij[n] * delij[p];
              arho3m[i][nv3] = arho3m[i][nv3] + A3jm * delij[m] * delij[n] * delij[p];
              //arho3[j][nv3] = arho3[j][nv3] + A3i * delij[m] * delij[n] * delij[p];
              //arho3m[j][nv3] = arho3m[j][nv3] + A3im * delij[m] * delij[n] * delij[p];
              nv3 = nv3 + 1;
	      for (q = p; q < 3; q++) {
		arho4m[i][nv4] = arho4m[i][nv4] + A4jm / rij2 * delij[m] * delij[n] * delij[p] * delij[q];
		//arho4m[j][nv4] = arho4m[j][nv4] + A4im / rij2 * delij[m] * delij[n] * delij[p] * delij[q];
		nv4 = nv4 + 1;
	      }
            }
          }
        }
      }
    }
  }
}
