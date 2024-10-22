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

#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
using namespace LAMMPS_NS;


void
MAGMEAM::meam_force(int i, int eflag_global, int eflag_atom, int vflag_global, int vflag_atom,
                 double* eng_vdwl, double* eatom, int /*ntype*/, int* type, int* fmap,
                 double** scale, double** x, double** sp, int numneigh, int* firstneigh, int numneigh_full,
		    int* firstneigh_full, int fnoffset, double** f, double** fm, double** fmds, double** vatom, double *virial, double hbar, double emag)
{
  int j, jn, k, kn, kk, m, n, p, q, r;
  int nv2, nv3, nv4, elti, eltj, eltk, ind;
  int eflag_either = eflag_atom || eflag_global;
  int vflag_either = vflag_atom || vflag_global;
  double xitmp, yitmp, zitmp, delij[3], rij2, rij, rij3, rij4;
  double spi[3], spj[3];
  double sdot, smag;
  double v[6], fi[3], fj[3];
  double third, sixth;
  double pp, dUdrij, dUdsij, dUdsijmag, dUdsijpair, dUdmag, dUdrijm[3], force, forcem;
  double recip, phi, phip;
  double phim, phimp;
  double sij, sijmag, sijpair;
  double a0mj, a0mi;
  double a1, a1i, a1j, a1mi, a1mj, a2, a2i, a2j, a2mi, a2mj;
  double a3, a3a, a4, a4a, a3i, a3j, a3mi, a3mj, a4mi, a4mj, a00mi, a00mj;
  double a1mag, a2mag, a3mag, a3amag, a4mag, a4amag;
  double shpi[3], shpj[3];
  double ai, aj, ro0i, ro0j, invrei, invrej;
  double rhoa0j, drhoa0j, rhoa0i, drhoa0i;
  double delrhoa0j, drhodmag0j, delrhoa0i, drhodmag0i;
  double delrhoa00j, drhodmag00j, delrhoa00i, drhodmag00i;
  double delrhoa00mj, delrhoa00mi;
  double rhoa1j, drhoa1j, rhoa1i, drhoa1i;
  double delrhoa1j, drhodmag1j, delrhoa1i, drhodmag1i;
  double rhoa2j, drhoa2j, rhoa2i, drhoa2i;
  double delrhoa2j, drhodmag2j, delrhoa2i, drhodmag2i;
  double rhoa3j, drhoa3j, rhoa3i, drhoa3i;
  double delrhoa3j, drhodmag3j, delrhoa3i, drhodmag3i;
  double delrhoa4j, drhodmag4j, delrhoa4i, drhodmag4i;
  double drhoa0mj, drhoa0mi, drhoa1mj, drhoa1mi;
  double drhoa00mj, drhoa00mi, drhoa00j, drhoa00i;
  double drhoa2mj, drhoa2mi, drhoa3mj, drhoa3mi, drhoa4mj, drhoa4mi;
  double drho0dr1, drho0dr2, drho0ds1, drho0ds2, drho0dmag1, drho0dmag2;
  double drho0mdr1, drho0mdr2, drho0mds1, drho0mds2;
  double drho00mdr1, drho00mdr2, drho00mdsa1, drho00mdsa2, drho00mdsb1, drho00mdsb2, drho00dmag1, drho00dmag2;
  double drho1dr1, drho1dr2, drho1ds1, drho1ds2, drho1dmag1, drho1dmag2;
  double drho1drm1[3], drho1drm2[3];
  double drho1mdr1, drho1mdr2, drho1mds1, drho1mds2;
  double drho1mdrm1[3], drho1mdrm2[3];
  double drho2dr1, drho2dr2, drho2ds1, drho2ds2, drho2dmag1, drho2dmag2;
  double drho2drm1[3], drho2drm2[3];
  double drho2mdr1, drho2mdr2, drho2mds1, drho2mds2;
  double drho2mdrm1[3], drho2mdrm2[3];
  double drho3dr1, drho3dr2, drho3ds1, drho3ds2, drho3dmag1, drho3dmag2;
  double drho3drm1[3], drho3drm2[3];
  double drho3mdr1, drho3mdr2, drho3mds1, drho3mds2;
  double drho3mdrm1[3], drho3mdrm2[3];
  double drho4dr1, drho4dr2, drho4ds1, drho4ds2, drho4dmag1, drho4dmag2;
  double drho4drm1[3], drho4drm2[3];
  double drho4mdr1, drho4mdr2, drho4mds1, drho4mds2;
  double drho4mdrm1[3], drho4mdrm2[3];
  double ddrho1dmag1, ddrho2dmag1, ddrho3dmag1, ddrho4dmag1;
  double ddrho1dmag2, ddrho2dmag2, ddrho3dmag2, ddrho4dmag2;
  double dt0mdr1, dt0mdr2, dt0mds1, dt0mds2;
  double dt1dr1, dt1dr2, dt1ds1, dt1ds2;
  double dt1mdr1, dt1mdr2, dt1mds1, dt1mds2;
  double dt2dr1, dt2dr2, dt2ds1, dt2ds2;
  double dt2mdr1, dt2mdr2, dt2mds1, dt2mds2;
  double dt3dr1, dt3dr2, dt3ds1, dt3ds2;
  double dt3mdr1, dt3mdr2, dt3mds1, dt3mds2;
  double dt4mdr1, dt4mdr2, dt4mds1, dt4mds2;
  double dt00mdr1, dt00mdr2, dt00mds1, dt00mds2;
  double drhodr1, drhodr2, drhods1, drhods2, drhodrm1[3], drhodrm2[3];
  double drhods1mag, drhods2mag;
  double drhodmag1, drhodmag2;
  long double fm1[3], fm2[3], fm3[3];
  double arg;
  double arg1i1, arg1j1, arg1im1, arg1jm1, arg1i2, arg1j2, arg1im2, arg1jm2, arg1i3, arg1j3, arg3i3, arg3j3, arg1im3, arg1jm3, arg3im3, arg3jm3;
  double arg2im4, arg2jm4, arg4im4, arg4jm4;
  double dsij1, dsij2, dsij1mag, dsij2mag, dsij1pair, dsij2pair, force1, force2;
  double t1i, t2i, t3i, t1j, t2j, t3j, tm0i, tm0j, tm1i, tm1j, tm2i, tm2j, tm3i, tm3j, tm4i, tm4j, tm00i, tm00j;
  double dU2dmag1, dU2dmag2, dU2dmag3, dU2dmag;
  double argd2im4, argd2im3, argd4im4, argd1im2, argd1im1, argd3im3, argd1im3;
  double argd2jm4, argd2jm3, argd4jm4, argd1jm2, argd1jm1, argd3jm3, argd1jm3;
  int countr;
  double scaleij;
    
  third = 1.0 / 3.0;
  sixth = 1.0 / 6.0;

  //     Compute forces atom i

  elti = fmap[type[i]];
  if (elti < 0) return;

  xitmp = x[i][0];
  yitmp = x[i][1];
  zitmp = x[i][2];

  spi[0] = sp[i][0];
  spi[1] = sp[i][1];
  spi[2] = sp[i][2];

  fm[i][0] = 0.0;
  fm[i][1] = 0.0;
  fm[i][2] = 0.0;

  fm1[0] = 0.0;
  fm1[1] = 0.0;
  fm1[2] = 0.0;

  fm2[0] = 0.0;
  fm2[1] = 0.0;
  fm2[2] = 0.0;

  fm3[0] = 0.0;
  fm3[1] = 0.0;
  fm3[2] = 0.0;

  emag = 0.0;

  //     Treat each pair
  for (jn = 0; jn < numneigh_full; jn++) {
    j = firstneigh_full[jn];
    eltj = fmap[type[j]];
    scaleij = scale[type[i]][type[j]];

    if ((!iszero(scrfcn[fnoffset + jn]) || !iszero(scrfcnmag[fnoffset + jn]) || !iszero(scrfcnpair[fnoffset + jn])) && eltj >= 0) {

      sij = scrfcn[fnoffset + jn] * fcpair[fnoffset + jn];
      sijmag = scrfcnmag[fnoffset + jn] * fcpair[fnoffset + jn];
      sijpair = scrfcnpair[fnoffset + jn] * fcpair[fnoffset + jn];
      //sij = 1.;//scrfcn[fnoffset + jn] * fcpair[fnoffset + jn];
      //sijmag = 1.;//scrfcnmag[fnoffset + jn] * fcpair[fnoffset + jn];
      //sijpair = 1.;//scrfcnpair[fnoffset + jn] * fcpair[fnoffset + jn];

      delij[0] = x[j][0] - xitmp;
      delij[1] = x[j][1] - yitmp;
      delij[2] = x[j][2] - zitmp;
      rij2 = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];

      spj[0] = sp[j][0];
      spj[1] = sp[j][1];
      spj[2] = sp[j][2];
      sdot = spi[0]*spj[0] + spi[1]*spj[1] + spi[2]*spj[2];
      smag = 0.5 * (1 - sdot);
      //smag = sdot;
      if (rij2 < this->cutforcesq) {
        rij = sqrt(rij2);
        recip = 1.0 / rij;
        //     Compute phi and phip
        ind = this->eltind[elti][eltj];
        pp = rij * this->rdrar;
        kk = (int)pp;
        kk = std::min(kk, this->nrar - 2);
        pp = pp - kk;
        pp = std::min(pp, 1.0);
        phi = ((this->phirar3[ind][kk] * pp + this->phirar2[ind][kk]) * pp + this->phirar1[ind][kk]) * pp + this->phirar[ind][kk];
        phip = (this->phirar6[ind][kk] * pp + this->phirar5[ind][kk]) * pp + this->phirar4[ind][kk];

	//     Compute phim and phimp
	//phim = ((this->magrar3[ind][kk] * pp + this->magrar2[ind][kk]) * pp + this->magrar1[ind][kk]) * pp + this->magrar[ind][kk];
	//phimp = (this->magrar6[ind][kk] * pp + this->magrar5[ind][kk]) * pp + this->magrar4[ind][kk];
	phim = ((this->magrar3[ind][kk] * pp + this->magrar2[ind][kk]) * pp + this->magrar1[ind][kk]) * pp + this->magrar[ind][kk];
	phimp = (this->magrar6[ind][kk] * pp + this->magrar5[ind][kk]) * pp + this->magrar4[ind][kk];
        if (eflag_either != 0) {
          double phi_sc = phi * scaleij;
          double phim_sc = phim * scaleij;
          if (eflag_global != 0)
	    *eng_vdwl = *eng_vdwl + 0.5 * phi * sij + 0.5 * phim * sijpair * smag;
          if (eflag_atom != 0) {
	    eatom[i] = eatom[i] + 0.5 * phi * sij + 0.5 * phim * sijpair * smag;
          }
        }
	emag += 0.5 * phim * sijpair * smag; 
        //     write(1,*) "force_meamf: phi: ",phi
        //     write(1,*) "force_meamf: phip: ",phip

        //     Compute pair densities and derivatives
        invrei = 1.0 / this->re_meam[elti][elti];
        ai = rij * invrei - 1.0;

        ro0i = this->rho0_meam[elti];
        rhoa0i = ro0i * MathSpecial::fm_exp(-this->beta0_meam[elti] * ai);
	delrhoa0i = ro0i * smag * MathSpecial::fm_exp(-this->mag_beta0_meam[elti] * ai);
	delrhoa00mi = ro0i * smag * smag * MathSpecial::fm_exp(-this->mag_beta0_meam[elti] * ai);
	delrhoa00i = ro0i * MathSpecial::fm_exp(-this->mag_beta00_meam[elti] * ai);
	drhoa0i = -invrei * this->beta0_meam[elti] * rhoa0i;
	drhoa0mi = -invrei * this->mag_beta0_meam[elti] * delrhoa0i;
	drhoa00mi = -invrei * this->mag_beta0_meam[elti] * delrhoa00mi;
	drhoa00i = -invrei * this->mag_beta00_meam[elti] * delrhoa00i;
        rhoa1i = ro0i *  MathSpecial::fm_exp(-this->beta1_meam[elti] * ai);
	delrhoa1i = ro0i * smag * MathSpecial::fm_exp(-this->mag_beta1_meam[elti] * ai);
        drhoa1i = -invrei * this->beta1_meam[elti] * rhoa1i;
	drhoa1mi = -invrei * this->mag_beta1_meam[elti] * delrhoa1i;
	rhoa2i = ro0i * MathSpecial::fm_exp(-this->beta2_meam[elti] * ai);
	delrhoa2i = ro0i * smag * MathSpecial::fm_exp(-this->mag_beta2_meam[elti] * ai);
	drhoa2i = -invrei * this->beta2_meam[elti] * rhoa2i;
	drhoa2mi = -invrei * this->mag_beta2_meam[elti] * delrhoa2i;
        rhoa3i = ro0i * MathSpecial::fm_exp(-this->beta3_meam[elti] * ai);
	delrhoa3i = ro0i * smag * MathSpecial::fm_exp(-this->mag_beta3_meam[elti] * ai);
        drhoa3i = -invrei * this->beta3_meam[elti] * rhoa3i;
	drhoa3mi = -invrei * this->mag_beta3_meam[elti] * delrhoa3i;
	delrhoa4i = ro0i * smag * MathSpecial::fm_exp(-this->mag_beta4_meam[elti] * ai);
	drhoa4mi = -invrei * this->mag_beta4_meam[elti] * delrhoa4i;

	//	rhoa0i += delrhoa0i;

	//      Compute spin derivatives
	drhodmag0i = -ro0i * MathSpecial::fm_exp(-this->mag_beta0_meam[elti] * ai);
	drhodmag1i = -ro0i * MathSpecial::fm_exp(-this->mag_beta1_meam[elti] * ai);
	drhodmag2i = -ro0i * MathSpecial::fm_exp(-this->mag_beta2_meam[elti] * ai);
	drhodmag3i = -ro0i * MathSpecial::fm_exp(-this->mag_beta3_meam[elti] * ai);
	drhodmag4i = -ro0i * MathSpecial::fm_exp(-this->mag_beta4_meam[elti] * ai);

        if (elti != eltj) {
          invrej = 1.0 / this->re_meam[eltj][eltj];
          aj = rij * invrej - 1.0;
          ro0j = this->rho0_meam[eltj];
          rhoa0j = ro0j * MathSpecial::fm_exp(-this->beta0_meam[eltj] * aj);
	  delrhoa0j = ro0j * smag * MathSpecial::fm_exp(-this->mag_beta0_meam[eltj] * aj);
	  delrhoa00mj = ro0j * smag * smag * MathSpecial::fm_exp(-this->mag_beta0_meam[eltj] * aj);
	  delrhoa00j = ro0j * MathSpecial::fm_exp(-this->mag_beta00_meam[eltj] * aj);
  	  drhoa0j = -invrej * this->beta0_meam[eltj] * rhoa0j;
	  drhoa0mj = -invrej * this->mag_beta0_meam[eltj] * delrhoa0j;
	  drhoa00mj = -invrej * this->mag_beta0_meam[eltj] * delrhoa00mj;
	  drhoa00j = -invrej * this->mag_beta00_meam[eltj] * delrhoa00j;
          rhoa1j = ro0j * MathSpecial::fm_exp(-this->beta1_meam[eltj] * aj);
	  delrhoa1j = ro0j * smag * MathSpecial::fm_exp(-this->mag_beta1_meam[eltj] * aj);
          drhoa1j = -invrej * this->beta1_meam[eltj] * rhoa1j;
	  drhoa1mj = -invrej * this->mag_beta1_meam[eltj] * delrhoa1j;
          rhoa2j = ro0j * MathSpecial::fm_exp(-this->beta2_meam[eltj] * aj);
	  delrhoa2j = ro0j * smag * MathSpecial::fm_exp(-this->mag_beta2_meam[eltj] * aj);
	  drhoa2j = -invrej * this->beta2_meam[eltj] * rhoa2j;
	  drhoa2mj = -invrej * this->mag_beta2_meam[eltj] * delrhoa2j;
          rhoa3j = ro0j * MathSpecial::fm_exp(-this->beta3_meam[eltj] * aj);
	  delrhoa3j = ro0j * smag * MathSpecial::fm_exp(-this->mag_beta3_meam[eltj] * aj);
          drhoa3j = -invrej * this->beta3_meam[eltj] * rhoa3j;
	  drhoa3mj = -invrej * this->mag_beta3_meam[eltj] * delrhoa3j;
	  delrhoa4j = ro0j * smag * MathSpecial::fm_exp(-this->mag_beta4_meam[eltj] * aj);
	  drhoa4mj = -invrej * this->mag_beta4_meam[eltj] * delrhoa4j;

	  //	  rhoa0j += delrhoa0j;
	  
	  drhodmag0j = -ro0j * MathSpecial::fm_exp(-this->mag_beta0_meam[eltj] * aj);
	  drhodmag1j = -ro0j * MathSpecial::fm_exp(-this->mag_beta1_meam[eltj] * aj);
	  drhodmag2j = -ro0j * MathSpecial::fm_exp(-this->mag_beta2_meam[eltj] * aj);
	  drhodmag3j = -ro0j * MathSpecial::fm_exp(-this->mag_beta3_meam[eltj] * aj);
	  drhodmag4j = -ro0j * MathSpecial::fm_exp(-this->mag_beta4_meam[eltj] * aj);

	} else {
          rhoa0j = rhoa0i;
	  delrhoa0j = delrhoa0i;
	  delrhoa00mj = delrhoa00mi;
	  delrhoa00j = delrhoa00i;
          drhoa0j = drhoa0i;
          drhoa0mj = drhoa0mi;
          drhoa00mj = drhoa00mi;
          drhoa00j = drhoa00i;
          rhoa1j = rhoa1i;
	  delrhoa1j = delrhoa1i;
          drhoa1j = drhoa1i;
          drhoa1mj = drhoa1mi;
          rhoa2j = rhoa2i;
	  delrhoa2j = delrhoa2i;
          drhoa2j = drhoa2i;
          drhoa2mj = drhoa2mi;
          rhoa3j = rhoa3i;
	  delrhoa3j = delrhoa3i;
          drhoa3j = drhoa3i;
          drhoa3mj = drhoa3mi;
	  delrhoa4j = delrhoa4i;
          drhoa4mj = drhoa4mi;

	  drhodmag0j = drhodmag0i;
	  drhodmag1j = drhodmag1i;
	  drhodmag2j = drhodmag2i;
	  drhodmag3j = drhodmag3i;
	  drhodmag4j = drhodmag4i;
        }

        const double t1mi = this->t1_meam[elti];
        const double t2mi = this->t2_meam[elti];
        const double t3mi = this->t3_meam[elti];
        const double tm0mi = this->mag_delrho0_meam[elti];
        const double tm1mi = this->mag_delrho1_meam[elti];
        const double tm2mi = this->mag_delrho2_meam[elti];
        const double tm3mi = this->mag_delrho3_meam[elti];
        const double tm4mi = this->mag_delrho4_meam[elti];
        const double tm00mi = this->mag_delrho00_meam[elti];
        const double t1mj = this->t1_meam[eltj];
        const double t2mj = this->t2_meam[eltj];
        const double t3mj = this->t3_meam[eltj];
        const double tm0mj = this->mag_delrho0_meam[eltj];
        const double tm1mj = this->mag_delrho1_meam[eltj];
        const double tm2mj = this->mag_delrho2_meam[eltj];
        const double tm3mj = this->mag_delrho3_meam[eltj];
        const double tm4mj = this->mag_delrho4_meam[eltj];
        const double tm00mj = this->mag_delrho00_meam[eltj];

        if (this->ialloy == 1) {
          rhoa1j  *= t1mj;
          rhoa2j  *= t2mj;
          rhoa3j  *= t3mj;
          delrhoa0j  *= tm0mj;
          delrhoa00mj  *= tm0mj;
          delrhoa00j  *= tm00mj;
          delrhoa1j  *= tm1mj;
          delrhoa2j  *= tm2mj;
          delrhoa3j  *= tm3mj;
          delrhoa4j  *= tm4mj;
          rhoa1i  *= t1mi;
          rhoa2i  *= t2mi;
          rhoa3i  *= t3mi;
          delrhoa0i  *= tm0mi;
          delrhoa00mi  *= tm0mi;
          delrhoa00i  *= tm00mi;
          delrhoa1i  *= tm1mi;
          delrhoa2i  *= tm2mi;
          delrhoa3i  *= tm3mi;
          delrhoa4i  *= tm4mi;
          drhoa1j *= t1mj;
          drhoa2j *= t2mj;
          drhoa3j *= t3mj;
          drhoa0mj *= tm0mj;
          drhoa00mj *= tm0mj;
          drhoa00j *= tm00mj;
          drhoa1mj *= tm1mj;
          drhoa2mj *= tm2mj;
          drhoa3mj *= tm3mj;
          drhoa4mj *= tm4mj;
          drhoa1i *= t1mi;
          drhoa2i *= t2mi;
          drhoa3i *= t3mi;
          drhoa0mi *= tm0mi;
          drhoa00mi *= tm0mi;
          drhoa00i *= tm00mi;
          drhoa1mi *= tm1mi;
          drhoa2mi *= tm2mi;
          drhoa3mi *= tm3mi;
          drhoa4mi *= tm4mi;
	  drhodmag0j *= tm0mj;
	  drhodmag1j *= tm1mj;
	  drhodmag2j *= tm2mj;
	  drhodmag3j *= tm3mj;
	  drhodmag4j *= tm4mj;
	  drhodmag0i *= tm0mi;
	  drhodmag1i *= tm1mi;
	  drhodmag2i *= tm2mi;
	  drhodmag3i *= tm3mi;
	  drhodmag4i *= tm4mi;
	  // rho00 not implemented for ialloy = 1
	}

        nv2 = 0;
        nv3 = 0;
	nv4 = 0;
        arg1i1 = 0.0;
        arg1j1 = 0.0;
        arg1im1 = 0.0;
        arg1jm1 = 0.0;
        arg1i2 = 0.0;
        arg1j2 = 0.0;
        arg1im2 = 0.0;
        arg1jm2 = 0.0;
        arg1i3 = 0.0;
        arg1j3 = 0.0;
        arg3i3 = 0.0;
        arg3j3 = 0.0;
        arg1im3 = 0.0;
        arg1jm3 = 0.0;
        arg3im3 = 0.0;
        arg3jm3 = 0.0;
        arg2im4 = 0.0;
        arg2jm4 = 0.0;
        arg4im4 = 0.0;
        arg4jm4 = 0.0;
	argd1im2 = 0.0;
	argd1im1 = 0.0;
	argd3im3 = 0.0;
	argd1im3 = 0.0;
	argd2jm4 = 0.0;
	argd2jm3 = 0.0;
	argd4jm4 = 0.0;
	argd1jm2 = 0.0;
	argd1jm1 = 0.0;
	argd3jm3 = 0.0;
	argd1jm3 = 0.0;

	for (n = 0; n < 3; n++) {
          for (p = n; p < 3; p++) {
            for (q = p; q < 3; q++) {
	      for (r = q; r < 3; r++) {
		arg = delij[n] * delij[p] * delij[q] * delij[r] * this->v4D[nv4];
		arg2im4 = arg2im4 + arho4m[i][nv4] * arg;
		arg2jm4 = arg2jm4 + arho4m[j][nv4] * arg;
		argd2im4 = argd2im4 + arg * arg;
		argd2jm4 = argd2jm4 + arg * arg;
		nv4 = nv4 + 1;
	      }
              arg = delij[n] * delij[p] * delij[q] * this->v3D[nv3];
              arg1i3 = arg1i3 + arho3[i][nv3] * arg;
              arg1j3 = arg1j3 - arho3[j][nv3] * arg;
              arg1im3 = arg1im3 + arho3m[i][nv3] * arg;
              arg1jm3 = arg1jm3 - arho3m[j][nv3] * arg;
              argd1im3 = argd1im3 + arg * arg;
              argd1jm3 = argd1jm3 + arg * arg;
              nv3 = nv3 + 1;
            }
            arg = delij[n] * delij[p] * this->v2D[nv2];
            arg1i2 = arg1i2 + arho2[i][nv2] * arg;
            arg1j2 = arg1j2 + arho2[j][nv2] * arg;
            arg1im2 = arg1im2 + arho2m[i][nv2] * arg;
            arg1jm2 = arg1jm2 + arho2m[j][nv2] * arg;
            argd1im2 = argd1im2 + arg * arg;
            argd1jm2 = argd1jm2 + arg * arg;
            arg4im4 = arg4im4 + arho4bm[i][nv2] * arg;
            arg4jm4 = arg4jm4 + arho4bm[j][nv2] * arg;
            argd4im4 = argd4im4 + arg * arg;
            argd4jm4 = argd4jm4 + arg * arg;
           nv2 = nv2 + 1;
          }
          arg1i1 = arg1i1 + arho1[i][n] * delij[n];
          arg1j1 = arg1j1 - arho1[j][n] * delij[n];
          arg3i3 = arg3i3 + arho3b[i][n] * delij[n];
          arg3j3 = arg3j3 - arho3b[j][n] * delij[n];
          arg1im1 = arg1im1 + arho1m[i][n] * delij[n];
          arg1jm1 = arg1jm1 - arho1m[j][n] * delij[n];
          argd1im1 = argd1im1 + delij[n] * delij[n];
          argd1jm1 = argd1jm1 + delij[n] * delij[n];
          arg3im3 = arg3im3 + arho3bm[i][n] * delij[n];
          arg3jm3 = arg3jm3 - arho3bm[j][n] * delij[n];
	  argd3im3 = argd3im3 + delij[n] * delij[n];
          argd3jm3 = argd3jm3 + delij[n] * delij[n];
        }

        //     rho0 terms
        drho0dr1 = drhoa0j * sij;
        drho0dr2 = drhoa0i * sij;

        drho0mdr1 = 2.0 * arho0m[i] * drhoa0mj * sijmag;
        drho0mdr2 = 2.0 * arho0m[j] * drhoa0mi * sijmag;

	drho0dmag1 = 2.0 * arho0m[i] * drhodmag0j * sijmag;
	drho0dmag2 = 2.0 * arho0m[j] * drhodmag0i * sijmag;

	//     rho00 terms
        drho00mdr1 = drhoa00j * sijmag * arho00m[i] + arho00[i] * drhoa00mj * sijmag;
        drho00mdr2 = drhoa00i * sijmag * arho00m[j] + arho00[j] * drhoa00mi * sijmag;

	drho00dmag1 = 2.0 * arho00[i] * drhodmag0j * sijmag * smag;
	drho00dmag2 = 2.0 * arho00[j] * drhodmag0i * sijmag * smag;

        //     rho1 terms
        a1 = 2 * sij / rij;
        a1mag = 2 * sijmag / rij;
        drho1dr1 = a1 * (drhoa1j - rhoa1j / rij) * arg1i1;
        drho1dr2 = a1 * (drhoa1i - rhoa1i / rij) * arg1j1;
	ddrho1dmag1 = a1mag * drhodmag1j * drhodmag1j * argd1im1 * sijmag;
	ddrho1dmag2 = a1mag * drhodmag1i * drhodmag1i * argd1jm1 * sijmag;

        drho1mdr1 = a1mag * (drhoa1mj - delrhoa1j / rij) * arg1im1;
        drho1mdr2 = a1mag * (drhoa1mi - delrhoa1i / rij) * arg1jm1;

	drho1dmag1 = a1mag * drhodmag1j * arg1im1;
	drho1dmag2 = a1mag * drhodmag1i * arg1jm1;
	
        a1 = 2.0 * sij / rij;
        a1mag = 2 * sijmag / rij;
        for (m = 0; m < 3; m++) {
          drho1drm1[m] = a1 * rhoa1j * arho1[i][m];
          drho1drm2[m] = -a1 * rhoa1i * arho1[j][m];
          drho1mdrm1[m] = a1mag * delrhoa1j * arho1m[i][m];
          drho1mdrm2[m] = -a1mag * delrhoa1i * arho1m[j][m];
        }

        //     rho2 terms
        a2 = 2 * sij / rij2;
	a2mag = 2 * sijmag /rij2;
        drho2dr1 = a2 * (drhoa2j - 2 * rhoa2j / rij) * arg1i2 - 2.0 / 3.0 * arho2b[i] * drhoa2j * sij;
        drho2dr2 = a2 * (drhoa2i - 2 * rhoa2i / rij) * arg1j2 - 2.0 / 3.0 * arho2b[j] * drhoa2i * sij;
	ddrho2dmag1 = a2mag * drhodmag2j * drhodmag2j * argd1im2 * sijmag - 2.0 / 3.0 * drhodmag2j * drhodmag2j * sijmag * sijmag;
	ddrho2dmag2 = a2mag * drhodmag2i * drhodmag2i * argd1jm2 * sijmag - 2.0 / 3.0 * drhodmag2i * drhodmag2i * sijmag * sijmag;

        drho2mdr1 = a2mag * (drhoa2mj - 2 * delrhoa2j / rij) * arg1im2 - 2.0 / 3.0 * arho2bm[i] * drhoa2mj * sijmag;
        drho2mdr2 = a2mag * (drhoa2mi - 2 * delrhoa2i / rij) * arg1jm2 - 2.0 / 3.0 * arho2bm[j] * drhoa2mi * sijmag;

	drho2dmag1 = a2mag * drhodmag2j * arg1im2 - 2.0 / 3.0 * arho2bm[i] * drhodmag2j * sijmag;
	drho2dmag2 = a2mag * drhodmag2i * arg1jm2 - 2.0 / 3.0 * arho2bm[j] * drhodmag2i * sijmag;
        a2 = 4 * sij / rij2;
	a2mag = 4 * sijmag /rij2;
        for (m = 0; m < 3; m++) {
          drho2drm1[m] = 0.0;
          drho2drm2[m] = 0.0;
          drho2mdrm1[m] = 0.0;
          drho2mdrm2[m] = 0.0;
          for (n = 0; n < 3; n++) {
            drho2drm1[m] = drho2drm1[m] + arho2[i][this->vind2D[m][n]] * delij[n];
            drho2drm2[m] = drho2drm2[m] - arho2[j][this->vind2D[m][n]] * delij[n];
            drho2mdrm1[m] = drho2mdrm1[m] + arho2m[i][this->vind2D[m][n]] * delij[n];
            drho2mdrm2[m] = drho2mdrm2[m] - arho2m[j][this->vind2D[m][n]] * delij[n];
          }
          drho2drm1[m] = a2 * rhoa2j * drho2drm1[m];
          drho2drm2[m] = -a2 * rhoa2i * drho2drm2[m];
          drho2mdrm1[m] = a2mag * delrhoa2j * drho2mdrm1[m];
          drho2mdrm2[m] = -a2mag * delrhoa2i * drho2mdrm2[m];
        }

        //     rho3 terms
        rij3 = rij * rij2;
        a3 = 2 * sij / rij3;
        a3a = 6.0 / 5.0 * sij / rij;
        a3mag = 2 * sijmag / rij3;
        a3amag = 6.0 / 5.0 * sijmag / rij;
        drho3dr1 = a3 * (drhoa3j - 3 * rhoa3j / rij) * arg1i3 - a3a * (drhoa3j - rhoa3j / rij) * arg3i3;
        drho3dr2 = a3 * (drhoa3i - 3 * rhoa3i / rij) * arg1j3 - a3a * (drhoa3i - rhoa3i / rij) * arg3j3;
        drho3mdr1 = a3mag * (drhoa3mj - 3 * delrhoa3j / rij) * arg1im3 - a3amag * (drhoa3mj - delrhoa3j / rij) * arg3im3;
        drho3mdr2 = a3mag * (drhoa3mi - 3 * delrhoa3i / rij) * arg1jm3 - a3amag * (drhoa3mi - delrhoa3i / rij) * arg3jm3;

        drho3dmag1 = a3mag * drhodmag3j * arg1im3 - a3amag * drhodmag3j * arg3im3;
        drho3dmag2 = a3mag * drhodmag3i * arg1jm3 - a3amag * drhodmag3i * arg3jm3;
        ddrho3dmag2 = a3mag * drhodmag3j * drhodmag3j * argd1im3 * sijmag - a3amag * drhodmag3j * drhodmag3j * arg3im3 * sijmag;
        ddrho3dmag2 = a3mag * drhodmag3i * drhodmag3i * argd1jm3 * sijmag - a3amag * drhodmag3i * drhodmag3i * arg3jm3 * sijmag;

	a3 = 6 * sij / rij3;
        a3a = 6 * sij / (5 * rij);
	a3mag = 6 * sijmag / rij3;
        a3amag = 6 * sijmag / (5 * rij);
        for (m = 0; m < 3; m++) {
          drho3drm1[m] = 0.0;
          drho3drm2[m] = 0.0;
          drho3mdrm1[m] = 0.0;
          drho3mdrm2[m] = 0.0;
          nv2 = 0;
          for (n = 0; n < 3; n++) {
            for (p = n; p < 3; p++) {
              arg = delij[n] * delij[p] * this->v2D[nv2];
              drho3drm1[m] = drho3drm1[m] + arho3[i][this->vind3D[m][n][p]] * arg;
              drho3drm2[m] = drho3drm2[m] + arho3[j][this->vind3D[m][n][p]] * arg;
              drho3mdrm1[m] = drho3mdrm1[m] + arho3m[i][this->vind3D[m][n][p]] * arg;
              drho3mdrm2[m] = drho3mdrm2[m] + arho3m[j][this->vind3D[m][n][p]] * arg;
              nv2 = nv2 + 1;
            }
          }
          drho3drm1[m] = (a3 * drho3drm1[m] - a3a * arho3b[i][m]) * rhoa3j;
          drho3drm2[m] = (-a3 * drho3drm2[m] + a3a * arho3b[j][m]) * rhoa3i;
          drho3mdrm1[m] = (a3mag * drho3mdrm1[m] - a3amag * arho3bm[i][m]) * delrhoa3j;
          drho3mdrm2[m] = (-a3mag * drho3mdrm2[m] + a3amag * arho3bm[j][m]) * delrhoa3i;
        }

        //     rho4 terms
        rij4 = rij2 * rij2;
	a4 = 2.0 * sij / rij4;
        a4a = 12.0 / 7.0 * sij / rij2;
        a4mag = 2.0 * sijmag / rij4;
        a4amag = 12.0 / 7.0 * sijmag / rij2;

	drho4mdr1 = a4mag * (drhoa4mj - 4 * delrhoa4j / rij) * arg2im4 - a4amag * (drhoa4mj - 2 * delrhoa4j / rij) * arg4im4 + 6.0 / 35.0 * arho4cm[i] * drhoa4mj * sijmag;
        drho4mdr2 = a4mag * (drhoa4mi - 4 * delrhoa4i / rij) * arg2jm4 - a4amag * (drhoa4mi - 2 * delrhoa4i / rij) * arg4jm4 + 6.0 / 35.0 * arho4cm[j] * drhoa4mi * sijmag;

        drho4dmag1 = a4mag * drhodmag4j * arg2im4 - a4amag * drhodmag4j * arg4im4 + 6.0 / 35.0 * arho4cm[i] * drhodmag4j * sijmag;
        drho4dmag2 = a4mag * drhodmag4i * arg2jm4 - a4amag * drhodmag4i * arg4jm4 + 6.0 / 35.0 * arho4cm[j] * drhodmag4i * sijmag;
        ddrho4dmag1 = a4mag * drhodmag4j * drhodmag4j * argd2im4 *sijmag - a4amag * drhodmag4j * drhodmag4j * argd4im4 *sijmag + 6.0 / 35.0 * drhodmag4j * drhodmag4j * sijmag * sijmag;
        ddrho4dmag2 = a4mag * drhodmag4i * drhodmag4i * argd2jm4 *sijmag - a4amag * drhodmag4i * drhodmag4i * argd4jm4 *sijmag + 6.0 / 35.0 * drhodmag4i * drhodmag4i * sijmag * sijmag;

	a4 = 8 * sij / rij4;
        a4a = 24.0 * sij / (7 * rij2);
	a4mag = 8.0 * sijmag / rij4;
        a4amag = 24.0 * sijmag / (7 * rij2);
        for (m = 0; m < 3; m++) {
          drho4mdrm1[m] = 0.0;
          drho4mdrm2[m] = 0.0;
	  nv3 = 0;
          for (n = 0; n < 3; n++) {
            for (p = n; p < 3; p++) {
	      for (q = p; q < 3; q++) {
		arg = delij[n] * delij[p] * delij[q] * this->v3D[nv3];
		drho4mdrm1[m] = drho4mdrm1[m] + a4mag * arho4m[i][this->vind4D[m][n][p][q]] * arg;
		drho4mdrm2[m] = drho4mdrm2[m] - a4mag * arho4m[j][this->vind4D[m][n][p][q]] * arg;
		nv3 = nv3 + 1;
	      }
            }
            drho4mdrm1[m] = drho4mdrm1[m] - a4amag * arho4bm[i][this->vind2D[m][n]] * delij[n];
            drho4mdrm2[m] = drho4mdrm2[m] + a4amag * arho4bm[j][this->vind2D[m][n]] * delij[n];
          }
          drho4mdrm1[m] = delrhoa4j * drho4mdrm1[m];
          drho4mdrm2[m] = -delrhoa4i * drho4mdrm2[m];
        }

        //     Compute derivatives of weighting functions t wrt rij
        t1i = t_ave[i][0];
        t2i = t_ave[i][1];
        t3i = t_ave[i][2];
        tm0i = t_ave[i][3];
        tm1i = t_ave[i][4];
        tm2i = t_ave[i][5];
        tm3i = t_ave[i][6];
        tm4i = t_ave[i][7];
        tm00i = t_ave[i][8];
        t1j = t_ave[j][0];
        t2j = t_ave[j][1];
        t3j = t_ave[j][2];
        tm0j = t_ave[j][3];
        tm1j = t_ave[j][4];
        tm2j = t_ave[j][5];
        tm3j = t_ave[j][6];
        tm4j = t_ave[j][7];
        tm00j = t_ave[j][8];

        if (this->ialloy == 1) {

          a1i = fdiv_zero(drhoa0j * sij, tsq_ave[i][0]);
          a1j = fdiv_zero(drhoa0i * sij, tsq_ave[j][0]);
          a2i = fdiv_zero(drhoa0j * sij, tsq_ave[i][1]);
          a2j = fdiv_zero(drhoa0i * sij, tsq_ave[j][1]);
          a3i = fdiv_zero(drhoa0j * sij, tsq_ave[i][2]);
          a3j = fdiv_zero(drhoa0i * sij, tsq_ave[j][2]);
          a0mi = fdiv_zero(drhoa0j * sij, tsq_ave[i][3]);
          a0mj = fdiv_zero(drhoa0i * sij, tsq_ave[j][3]);
          a1mi = fdiv_zero(drhoa0j * sij, tsq_ave[i][4]);
          a1mj = fdiv_zero(drhoa0i * sij, tsq_ave[j][4]);
          a2mi = fdiv_zero(drhoa0j * sij, tsq_ave[i][5]);
          a2mj = fdiv_zero(drhoa0i * sij, tsq_ave[j][5]);
          a3mi = fdiv_zero(drhoa0j * sij, tsq_ave[i][6]);
          a3mj = fdiv_zero(drhoa0i * sij, tsq_ave[j][6]);
          a4mi = fdiv_zero(drhoa0j * sij, tsq_ave[i][7]);
          a4mj = fdiv_zero(drhoa0i * sij, tsq_ave[j][7]);
          a00mi = fdiv_zero(drhoa0j * sij, tsq_ave[i][8]);
          a00mj = fdiv_zero(drhoa0i * sij, tsq_ave[j][8]);

          dt1dr1 = a1i * (t1mj - t1i * MathSpecial::square(t1mj));
          dt1dr2 = a1j * (t1mi - t1j * MathSpecial::square(t1mi));
          dt2dr1 = a2i * (t2mj - t2i * MathSpecial::square(t2mj));
          dt2dr2 = a2j * (t2mi - t2j * MathSpecial::square(t2mi));
          dt3dr1 = a3i * (t3mj - t3i * MathSpecial::square(t3mj));
          dt3dr2 = a3j * (t3mi - t3j * MathSpecial::square(t3mi));
          dt0mdr1 = a0mi * (tm0mj - tm0i * MathSpecial::square(tm0mj));
          dt0mdr2 = a0mj * (tm0mi - tm0j * MathSpecial::square(tm0mi));
          dt1mdr1 = a1mi * (tm1mj - tm1i * MathSpecial::square(tm1mj));
          dt1mdr2 = a1mj * (tm1mi - tm1j * MathSpecial::square(tm1mi));
          dt2mdr1 = a2mi * (tm2mj - tm2i * MathSpecial::square(tm2mj));
          dt2mdr2 = a2mj * (tm2mi - tm2j * MathSpecial::square(tm2mi));
          dt3mdr1 = a3mi * (tm3mj - tm3i * MathSpecial::square(tm3mj));
          dt3mdr2 = a3mj * (tm3mi - tm3j * MathSpecial::square(tm3mi));
          dt4mdr1 = a4mi * (tm4mj - tm4i * MathSpecial::square(tm4mj));
          dt4mdr2 = a4mj * (tm4mi - tm4j * MathSpecial::square(tm4mi));
          dt00mdr1 = a00mi * (tm00mj - tm00i * MathSpecial::square(tm4mj));
          dt00mdr2 = a00mj * (tm00mi - tm00j * MathSpecial::square(tm4mi));

        } else if (this->ialloy == 2) {

          dt1dr1 = 0.0;
          dt1dr2 = 0.0;
          dt2dr1 = 0.0;
          dt2dr2 = 0.0;
          dt3dr1 = 0.0;
          dt3dr2 = 0.0;
          dt0mdr1 = 0.0;
          dt0mdr2 = 0.0;
          dt1mdr1 = 0.0;
          dt1mdr2 = 0.0;
          dt2mdr1 = 0.0;
          dt2mdr2 = 0.0;
          dt3mdr1 = 0.0;
          dt3mdr2 = 0.0;
          dt4mdr1 = 0.0;
          dt4mdr2 = 0.0;
          dt00mdr1 = 0.0;
          dt00mdr2 = 0.0;

        } else {

          ai = 0.0;
          if (!iszero(rho0[i]))
            ai = drhoa0j * sij / rho0[i];
          aj = 0.0;
          if (!iszero(rho0[j]))
            aj = drhoa0i * sij / rho0[j];

          dt1dr1 = ai * (t1mj - t1i);
          dt1dr2 = aj * (t1mi - t1j);
          dt2dr1 = ai * (t2mj - t2i);
          dt2dr2 = aj * (t2mi - t2j);
          dt3dr1 = ai * (t3mj - t3i);
          dt3dr2 = aj * (t3mi - t3j);
          dt0mdr1 = ai * (tm0mj - tm0i);
          dt0mdr2 = aj * (tm0mi - tm0j);
          dt1mdr1 = ai * (tm1mj - tm1i);
          dt1mdr2 = aj * (tm1mi - tm1j);
          dt2mdr1 = ai * (tm2mj - tm2i);
          dt2mdr2 = aj * (tm2mi - tm2j);
          dt3mdr1 = ai * (tm3mj - tm3i);
          dt3mdr2 = aj * (tm3mi - tm3j);
          dt4mdr1 = ai * (tm4mj - tm4i);
          dt4mdr2 = aj * (tm4mi - tm4j);
          dt00mdr1 = ai * (tm00mj - tm00i);
          dt00mdr2 = aj * (tm00mi - tm00j);
        }

        //     Compute derivatives of total density wrt rij, sij and rij(3)
        get_shpfcn(this->lattce_meam[elti][elti], this->stheta_meam[elti][elti], this->ctheta_meam[elti][elti], shpi);
        get_shpfcn(this->lattce_meam[eltj][eltj], this->stheta_meam[elti][elti], this->ctheta_meam[elti][elti], shpj);

        drhodr1 = dgamma1[i] * drho0dr1 +
          dgamma2[i] * (dt0mdr1 * rho0m[i] + tm0i * drho0mdr1 + dt1dr1 * rho1[i] + t1i * drho1dr1 + dt1mdr1 * rho1m[i] + tm1i * drho1mdr1 +
			dt2dr1 * rho2[i] + t2i * drho2dr1 + dt2mdr1 * rho2m[i] + tm2i * drho2mdr1 +
                        dt3dr1 * rho3[i] + t3i * drho3dr1 + dt3mdr1 * rho3m[i] + tm3i * drho3mdr1 + dt4mdr1 * rho4m[i] + tm4i * drho4mdr1 + dt00mdr1 * rho00[i] + tm00i * drho00mdr1) -
          dgamma3[i] * (shpi[0] * dt1dr1 + shpi[1] * dt2dr1 + shpi[2] * dt3dr1);
        drhodr2 = dgamma1[j] * drho0dr2 +
          dgamma2[j] * (dt0mdr2 * rho0m[j] + tm0j * drho0mdr2 + dt1dr2 * rho1[j] + t1j * drho1dr2 + dt1mdr2 * rho1m[j] + tm1j * drho1mdr2 +
			dt2dr2 * rho2[j] + t2j * drho2dr2 + dt2mdr2 * rho2m[j] + tm2j * drho2mdr2 +
                        dt3dr2 * rho3[j] + t3j * drho3dr2 + dt3mdr2 * rho3m[j] + tm3j * drho3mdr2 + dt4mdr2 * rho4m[j] + tm4j * drho4mdr2 + dt00mdr2 * rho00[j] + tm00j * drho00mdr2) -
          dgamma3[j] * (shpj[0] * dt1dr2 + shpj[1] * dt2dr2 + shpj[2] * dt3dr2);

	drhodmag1 = dgamma2[i] * (tm0i * drho0dmag1 + tm1i * drho1dmag1 + tm2i * drho2dmag1 + tm3i * drho3dmag1 + tm4i * drho4dmag1 + tm00i * drho00dmag1);
	drhodmag2 = dgamma2[j] * (tm0j * drho0dmag2 + tm1j * drho1dmag2 + tm2j * drho2dmag2 + tm3j * drho3dmag2 + tm4j * drho4dmag2 + tm00j * drho00dmag2);
	
        for (m = 0; m < 3; m++) {
          drhodrm1[m] = 0.0;
          drhodrm2[m] = 0.0;
          drhodrm1[m] = dgamma2[i] * (t1i * drho1drm1[m] + tm1i * drho1mdrm1[m] +
				      t2i * drho2drm1[m] + tm2i * drho2mdrm1[m] + t3i * drho3drm1[m] + tm3i * drho3mdrm1[m] + tm4i * drho4mdrm1[m]);
          drhodrm2[m] = dgamma2[j] * (t1j * drho1drm2[m] + tm1j * drho1mdrm2[m] +
				      t2j * drho2drm2[m] + tm2j * drho2mdrm2[m] + t3j * drho3drm2[m] + tm3j * drho3mdrm2[m] + tm4j * drho4mdrm2[m]);
        }

        //     Compute derivatives wrt sij, but only if necessary
        if (!iszero(dscrfcn[fnoffset + jn]) || !iszero(dscrfcnmag[fnoffset + jn])) {


	  drho0ds1 = rhoa0j;
          drho0ds2 = rhoa0i;
	  drho0mds1 = 2.0 * arho0m[i] * delrhoa0j;
	  drho0mds2 = 2.0 * arho0m[j] * delrhoa0i;
	  drho00mdsa1 = arho00m[i] * delrhoa00j + arho00[i] * delrhoa00mj;
	  drho00mdsa2 = arho00m[j] * delrhoa00i + arho00[j] * delrhoa00mi;
	  //drho00mdsb1 = arho00m[i] * delrhoa0j;
	  //drho00mdsb2 = arho00m[j] * delrhoa0i;
	  a1 = 2.0 / rij;
          drho1ds1 = a1 * rhoa1j * arg1i1;
          drho1ds2 = a1 * rhoa1i * arg1j1;
          drho1mds1 = a1 * delrhoa1j * arg1im1;
          drho1mds2 = a1 * delrhoa1i * arg1jm1;
          a2 = 2.0 / rij2;
          drho2ds1 = a2 * rhoa2j * arg1i2 - 2.0 / 3.0 * arho2b[i] * rhoa2j;
          drho2ds2 = a2 * rhoa2i * arg1j2 - 2.0 / 3.0 * arho2b[j] * rhoa2i;
          drho2mds1 = a2 * delrhoa2j * arg1im2 - 2.0 / 3.0 * arho2bm[i] * delrhoa2j;
          drho2mds2 = a2 * delrhoa2i * arg1jm2 - 2.0 / 3.0 * arho2bm[j] * delrhoa2i;
          a3 = 2.0 / rij3;
          a3a = 6.0 / (5.0 * rij);
          drho3ds1 = a3 * rhoa3j * arg1i3 - a3a * rhoa3j * arg3i3;
          drho3ds2 = a3 * rhoa3i * arg1j3 - a3a * rhoa3i * arg3j3;
          drho3mds1 = a3 * delrhoa3j * arg1im3 - a3a * delrhoa3j * arg3im3;
          drho3mds2 = a3 * delrhoa3i * arg1jm3 - a3a * delrhoa3i * arg3jm3;
          a4 = 2.0 / rij4;
          a4a = 12.0 / (7.0 * rij2);
	  drho4mds1 = a4 * delrhoa4j * arg2im4 - a4a * delrhoa4j * arg4im4 + 6.0 / 35.0 * arho4cm[i] * delrhoa4j;
          drho4mds2 = a4 * delrhoa4i * arg2jm4 - a4a * delrhoa4i * arg4jm4 + 6.0 / 35.0 * arho4cm[j] * delrhoa4i;

          if (this->ialloy == 1) {
            a1i = fdiv_zero(rhoa0j, tsq_ave[i][0]);
            a1j = fdiv_zero(rhoa0i, tsq_ave[j][0]);
            a2i = fdiv_zero(rhoa0j, tsq_ave[i][1]);
            a2j = fdiv_zero(rhoa0i, tsq_ave[j][1]);
            a3i = fdiv_zero(rhoa0j, tsq_ave[i][2]);
            a3j = fdiv_zero(rhoa0i, tsq_ave[j][2]);
            a0mi = fdiv_zero(rhoa0j, tsq_ave[i][3]);
            a0mj = fdiv_zero(rhoa0i, tsq_ave[j][3]);
            a1mi = fdiv_zero(rhoa0j, tsq_ave[i][4]);
            a1mj = fdiv_zero(rhoa0i, tsq_ave[j][4]);
            a2mi = fdiv_zero(rhoa0j, tsq_ave[i][5]);
            a2mj = fdiv_zero(rhoa0i, tsq_ave[j][5]);
            a3mi = fdiv_zero(rhoa0j, tsq_ave[i][6]);
            a3mj = fdiv_zero(rhoa0i, tsq_ave[j][6]);
            a4mi = fdiv_zero(rhoa0j, tsq_ave[i][7]);
            a4mj = fdiv_zero(rhoa0i, tsq_ave[j][7]);
            a00mi = fdiv_zero(rhoa0j, tsq_ave[i][8]);
            a00mj = fdiv_zero(rhoa0i, tsq_ave[j][8]);

            dt1ds1 = a1i * (t1mj - t1i * MathSpecial::square(t1mj));
            dt1ds2 = a1j * (t1mi - t1j * MathSpecial::square(t1mi));
            dt2ds1 = a2i * (t2mj - t2i * MathSpecial::square(t2mj));
            dt2ds2 = a2j * (t2mi - t2j * MathSpecial::square(t2mi));
            dt3ds1 = a3i * (t3mj - t3i * MathSpecial::square(t3mj));
            dt3ds2 = a3j * (t3mi - t3j * MathSpecial::square(t3mi));
            dt0mds1 = a0mi * (tm0mj - tm0i * MathSpecial::square(tm0mj));
            dt0mds2 = a0mj * (tm0mi - tm0j * MathSpecial::square(tm0mi));
            dt1mds1 = a1mi * (tm1mj - tm1i * MathSpecial::square(tm1mj));
            dt1mds2 = a1mj * (tm1mi - tm1j * MathSpecial::square(tm1mi));
            dt2mds1 = a2mi * (tm2mj - tm2i * MathSpecial::square(tm2mj));
            dt2mds2 = a2mj * (tm2mi - tm2j * MathSpecial::square(tm2mi));
            dt3mds1 = a3mi * (tm3mj - tm3i * MathSpecial::square(tm3mj));
            dt3mds2 = a3mj * (tm3mi - tm3j * MathSpecial::square(tm3mi));
            dt4mds1 = a4mi * (tm4mj - tm4i * MathSpecial::square(tm4mj));
            dt4mds2 = a4mj * (tm4mi - tm4j * MathSpecial::square(tm4mi));
            dt00mds1 = a00mi * (tm00mj - tm00i * MathSpecial::square(tm00mj));
            dt00mds2 = a00mj * (tm00mi - tm00j * MathSpecial::square(tm00mi));

          } else if (this->ialloy == 2) {

            dt1ds1 = 0.0;
            dt1ds2 = 0.0;
            dt2ds1 = 0.0;
            dt2ds2 = 0.0;
            dt3ds1 = 0.0;
            dt3ds2 = 0.0;
            dt0mds1 = 0.0;
            dt0mds2 = 0.0;
            dt1mds1 = 0.0;
            dt1mds2 = 0.0;
            dt2mds1 = 0.0;
            dt2mds2 = 0.0;
            dt3mds1 = 0.0;
            dt3mds2 = 0.0;
            dt4mds1 = 0.0;
            dt4mds2 = 0.0;
            dt00mds1 = 0.0;
            dt00mds2 = 0.0;

          } else {

            ai = 0.0;
            if (!iszero(rho0[i]))
              ai = rhoa0j / rho0[i];
            aj = 0.0;
            if (!iszero(rho0[j]))
              aj = rhoa0i / rho0[j];

            dt1ds1 = ai * (t1mj - t1i);
            dt1ds2 = aj * (t1mi - t1j);
            dt2ds1 = ai * (t2mj - t2i);
            dt2ds2 = aj * (t2mi - t2j);
            dt3ds1 = ai * (t3mj - t3i);
            dt3ds2 = aj * (t3mi - t3j);
            dt0mds1 = ai * (tm0mj - tm0i);
            dt0mds2 = aj * (tm0mi - tm0j);
            dt1mds1 = ai * (tm1mj - tm1i);
            dt1mds2 = aj * (tm1mi - tm1j);
            dt2mds1 = ai * (tm2mj - tm2i);
            dt2mds2 = aj * (tm2mi - tm2j);
            dt3mds1 = ai * (tm3mj - tm3i);
            dt3mds2 = aj * (tm3mi - tm3j);
            dt4mds1 = ai * (tm4mj - tm4i);
            dt4mds2 = aj * (tm4mi - tm4j);
            dt00mds1 = ai * (tm00mj - tm00i);
            dt00mds2 = aj * (tm00mi - tm00j);
          }

          drhods1 = dgamma1[i] * drho0ds1 +
            dgamma2[i] * (dt1ds1 * rho1[i] + t1i * drho1ds1 + dt2ds1 * rho2[i] + t2i * drho2ds1 + dt3ds1 * rho3[i] + t3i * drho3ds1) -
            dgamma3[i] * (shpi[0] * dt1ds1 + shpi[1] * dt2ds1 + shpi[2] * dt3ds1);
          drhods2 = dgamma1[j] * drho0ds2 +
            dgamma2[j] * (dt1ds2 * rho1[j] + t1j * drho1ds2 + dt2ds2 * rho2[j] + t2j * drho2ds2 + dt3ds2 * rho3[j] + t3j * drho3ds2) -
            dgamma3[j] * (shpj[0] * dt1ds2 + shpj[1] * dt2ds2 + shpj[2] * dt3ds2);

          drhods1mag = dgamma2[i] * (dt0mds1 * rho0m[i] + tm0i * drho0mds1 + dt1mds1 * rho1m[i] + tm1i * drho1mds1 +
				     dt2mds1 * rho2m[i] + tm2i * drho2mds1 + dt3mds1 * rho3m[i] + tm3i * drho3mds1 + dt4mds1 * rho4m[i] + tm4i * drho4mds1 + dt00mds1 * rho00[i] + tm00i * drho00mdsa1);
          drhods2mag = dgamma2[j] * (dt0mds2 * rho0m[j] + tm0j * drho0mds2 + dt1mds2 * rho1m[j] + tm1j * drho1mds2 +
				     dt2mds2 * rho2m[j] + tm2j * drho2mds2 + dt3mds2 * rho3m[j] + tm3j * drho3mds2 + dt4mds2 * rho4m[j] + tm4j * drho4mds2 + dt00mds2 * rho00[j] + tm00j * drho00mdsa2);
        }


        //     Compute derivatives of energy wrt rij, sij and rij[3]
	dUdrij = phip * sij + phimp * sijpair * smag + frhop[i] * drhodr1 + frhop[j] * drhodr2;

	dU2dmag1 = 1.0 * frhopp[j]*drhodmag2*drhodmag2;
	dU2dmag2 = 0.5 * frhop[j] * d2gamma2[j] * (tm0j * drho0dmag2 + tm1j * drho1dmag2 + tm2j * drho2dmag2 + tm3j * drho3dmag2 + tm4j * drho4dmag2 + tm00j * drho00dmag2)*(tm0j * drho0dmag2 + tm1j * drho1dmag2 + tm2j * drho2dmag2 + tm3j * drho3dmag2 + tm4j * drho4dmag2 + tm00j * drho00dmag2);
	dU2dmag3 = 0.5 * frhop[j] * dgamma2[j] * (tm0j * drhodmag0i * drhodmag0i * sijmag * sijmag + tm1j * ddrho1dmag2 + tm2j * ddrho2dmag2 + tm3j * ddrho3dmag2 + tm4j * ddrho4dmag2);
	dU2dmag = dU2dmag1 + dU2dmag2 + dU2dmag3;

	dUdmag = frhop[i] * drhodmag1 + frhop[j] * drhodmag2;
        dUdsij = 0.0;
        dUdsijmag = 0.0;
	dUdsijpair = 0.0;
	if (!iszero(dscrfcn[fnoffset + jn])) {
	  dUdsij = phi + frhop[i] * drhods1 + frhop[j] * drhods2;
        }
        if (!iszero(dscrfcnmag[fnoffset + jn])) {
          dUdsijmag = frhop[i] * drhods1mag + frhop[j] * drhods2mag;
        }
        if (!iszero(dscrfcnpair[fnoffset + jn])) {
	  dUdsijpair = phim * smag;
	  //dUdsijpair = 0;
        }
        for (m = 0; m < 3; m++) {
          dUdrijm[m] = frhop[i] * drhodrm1[m] + frhop[j] * drhodrm2[m];
        }
        if (!isone(scaleij)) {
          dUdrij *= scaleij;
          dUdsij *= scaleij;
          dUdrijm[0] *= scaleij;
          dUdrijm[1] *= scaleij;
          dUdrijm[2] *= scaleij;
        }

        //     Add the part of the force due to dUdrij and dUdsij

        force = dUdrij * recip + dUdsij * dscrfcn[fnoffset + jn] + dUdsijmag * dscrfcnmag[fnoffset + jn] + dUdsijpair * dscrfcnpair[fnoffset + jn];
	countr = 0;
	//	if (j == 0) {
	//std::cout << "phim = " << phim << "\n";
	//std::cout << "dUdmag = " << dUdmag << "\n";
	//}	  
	for (m = 0; m < 3; m++) {
          forcem = delij[m] * force + dUdrijm[m];
          f[i][m] = f[i][m] + forcem;
	  fm[i][m] = fm[i][m] + 0.5 * spj[m] * (phim * sijpair - dUdmag);
	  fm1[m] = fm1[m] - 0.5 * spj[m] * (phim * sijpair);
	  fm2[m] = fm2[m] + 0.5 * spj[m] * (dUdmag);
	  //fm[i][m] = fm[i][m] - 0.5 * spj[m] * (phim * sijpair);// - dUdmag);
	  for (n = m; n < 3; n++) {
	    fmds[i][countr] = fmds[i][countr] + 1. * spj[m] * spj[n] * dU2dmag;
	    countr += 1;
	  }
        }
	
        //     Tabulate per-atom virial as symmetrized stress tensor

        if (vflag_either) {
          fi[0] = delij[0] * force + dUdrijm[0];
          fi[1] = delij[1] * force + dUdrijm[1];
          fi[2] = delij[2] * force + dUdrijm[2];
          v[0] = -0.5 * (delij[0] * fi[0]);
          v[1] = -0.5 * (delij[1] * fi[1]);
          v[2] = -0.5 * (delij[2] * fi[2]);
          v[3] = -0.25 * (delij[0] * fi[1] + delij[1] * fi[0]);
          v[4] = -0.25 * (delij[0] * fi[2] + delij[2] * fi[0]);
          v[5] = -0.25 * (delij[1] * fi[2] + delij[2] * fi[1]);

          if (vflag_global) {
            for (m = 0; m < 6; m++) {
              virial[m] += 1.0*v[m];
            }
          }
          if (vflag_atom) {
            for (m = 0; m < 6; m++) {
              vatom[i][m] += v[m];
            }
          }
        }

        //     Now compute forces on other atoms k due to change in sij

        if ((iszero(sij) || isone(sij)) && (iszero(sijmag) || isone(sijmag)) && (iszero(sijpair) || isone(sijpair))) continue; //: cont jn loop

        double dxik(0), dyik(0), dzik(0);
        double dxjk(0), dyjk(0), dzjk(0);

        for (kn = 0; kn < numneigh_full; kn++) {
          k = firstneigh_full[kn];
          eltk = fmap[type[k]];
          if (k != j && eltk >= 0) {
            double xik, xjk, cikj, cikjmag, cikjpair, sikj, sikjmag, sikjpair, dfc, a, amag, apair;
            double dCikj1, dCikj2;
            double delc, delcmag, delcpair, rik2, rjk2;
	    double drhods1k, drhods2k, drhods1kmag, drhods2kmag, drhods1kpair, drhods2kpair;

            sij = scrfcn[jn+fnoffset] * fcpair[jn+fnoffset];
            sijmag = scrfcnmag[jn+fnoffset] * fcpair[jn+fnoffset];
            sijpair = scrfcnpair[jn+fnoffset] * fcpair[jn+fnoffset];
            const double Cmax = this->Cmax_meam[elti][eltj][eltk];
            const double Cmin = this->Cmin_meam[elti][eltj][eltk];
            const double Cmaxmag = this->Cmax_magmeam[elti][eltj][eltk];
            const double Cminmag = this->Cmin_magmeam[elti][eltj][eltk];
            const double Cmaxpair = this->Cmax_magpair[elti][eltj][eltk];
            const double Cminpair = this->Cmin_magpair[elti][eltj][eltk];

            dsij1 = 0.0;
            dsij2 = 0.0;
            dsij1mag = 0.0;
            dsij2mag = 0.0;
            dsij1pair = 0.0;
            dsij2pair = 0.0;
            if ((!iszero(sij) && !isone(sij)) || (!iszero(sijmag) && !isone(sijmag)) || (!iszero(sijpair) && !isone(sijpair))) {
              const double rbound = rij2 * this->ebound_meam[elti][eltj];
              delc = Cmax - Cmin;
              delcmag = Cmaxmag - Cminmag;
              delcpair = Cmaxpair - Cminpair;
              dxjk = x[k][0] - x[j][0];
              dyjk = x[k][1] - x[j][1];
              dzjk = x[k][2] - x[j][2];
              rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
              if (rjk2 <= rbound) {
                dxik = x[k][0] - x[i][0];
                dyik = x[k][1] - x[i][1];
                dzik = x[k][2] - x[i][2];
                rik2 = dxik * dxik + dyik * dyik + dzik * dzik;
                if (rik2 <= rbound) {
                  xik = rik2 / rij2;
                  xjk = rjk2 / rij2;
                  a = 1 - (xik - xjk) * (xik - xjk);
                  if (!iszero(a)) {
                    cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
                    if (cikj > Cmin && cikj <= Cmax) {
                      cikj = (cikj - Cmin) / delc;
                      sikj = dfcut(cikj, dfc);
                      dCfunc2(rij2, rik2, rjk2, dCikj1, dCikj2);
                      a = sij / delc * dfc / sikj;
                      dsij1 = a * dCikj1;
                      dsij2 = a * dCikj2;
                    }
                  }
                  amag = 1 - (xik - xjk) * (xik - xjk);
                  if (!iszero(amag)) {
                    cikjmag = (2.0 * (xik + xjk) + amag - 2.0) / amag;
                    if (cikjmag > Cminmag && cikjmag <= Cmaxmag) {
                      cikjmag = (cikjmag - Cminmag) / delcmag;
                      sikjmag = dfcut(cikjmag, dfc);
                      dCfunc2(rij2, rik2, rjk2, dCikj1, dCikj2);
                      amag = sijmag / delcmag * dfc / sikjmag;
                      dsij1mag = amag * dCikj1;
                      dsij2mag = amag * dCikj2;
                    }
                  }
                  apair = 1 - (xik - xjk) * (xik - xjk);
                  if (!iszero(apair)) {
                    cikjpair = (2.0 * (xik + xjk) + apair - 2.0) / apair;
                    if (cikjpair > Cminpair && cikjpair <= Cmaxpair) {
                      cikjpair = (cikjpair - Cminpair) / delcpair;
                      sikjpair = dfcut(cikjpair, dfc);
                      dCfunc2(rij2, rik2, rjk2, dCikj1, dCikj2);
                      apair = sijpair / delcpair * dfc / sikjpair;
                      dsij1pair = apair * dCikj1;
                      dsij2pair = apair * dCikj2;
                    }
                  }
                }
              }
            }

            if (!iszero(dsij1) || !iszero(dsij2) || !iszero(dsij1mag) || !iszero(dsij2mag) || !iszero(dsij1pair) || !iszero(dsij2pair)) {
              force1 = dUdsij * dsij1 + dUdsijmag * dsij1mag + dUdsijpair * dsij1pair;
              force2 = dUdsij * dsij2 + dUdsijmag * dsij2mag + dUdsijpair * dsij2pair;
	      
              f[i][0] += force1 * dxik;
              f[i][1] += force1 * dyik;
              f[i][2] += force1 * dzik;
              f[k][0] -= force1 * dxik;
              f[k][1] -= force1 * dyik;
              f[k][2] -= force1 * dzik;

              //     Tabulate per-atom virial as symmetrized stress tensor

              if (vflag_either) {
                fi[0] = force1 * dxik;
                fi[1] = force1 * dyik;
                fi[2] = force1 * dzik;
                v[0] = -third * (dxik * fi[0]);
                v[1] = -third * (dyik * fi[1]);
                v[2] = -third * (dzik * fi[2]);
                v[3] = -sixth * (dxik * fi[1] + dyik * fi[0]);
                v[4] = -sixth * (dxik * fi[2] + dzik * fi[0]);
                v[5] = -sixth * (dyik * fi[2] + dzik * fi[1]);

                if (vflag_global) {
                  for (m = 0; m < 6; m++) {
                    virial[m] += 3.0*v[m];
                  }
                }

                if (vflag_atom) {
                  for (m = 0; m < 6; m++) {
                    vatom[i][m] += v[m];
                    vatom[k][m] += 0.5 * v[m];
                  }
                }
              }
            }
          }
          //     end of k loop
        }
      }
    }
    //     end of j loop
  }
  //  if (i==0) outdata << std::setprecision(12) << fmds[0][0] << " " << fmds[0][1] << " " << fmds[0][2] << " " << fmds[0][3] << " " << fmds[0][4] << " " << fmds[0][5] << "\n";
}

void
MAGMEAM::meam_spin(int i, int /*ntype*/, int* type, int* fmap, double** scale, double** x,
		   double** sp, int numneigh, int* firstneigh, int fnoffset, double* fmi, double* fmdsi, double hbar)
{
  int j, jn, k, kn, kk, m, mm, n, p, q, r;
  int nv2, nv3, nv4, elti, eltj, eltk, ind;
  double xitmp, yitmp, zitmp, delij[3], rij2, rij, rij3, rij4;
  double spi[3], spj[3];
  double sdot, smag;
  double v[6], fi[3], fj[3];
  double third, sixth;
  double pp, dUdrij, dUdsij, dUdsijmag, dUdmag, dUdrijm[3], force, forcem;
  double recip, phi, phim;
  double dUdmag1, dUdmag2;
  double sij, sijmag, sijpair;
  double a1, a1i, a1j, a1mi, a1mj, a2, a2i, a2j, a2mi, a2mj;
  double a3, a3a, a4, a4a, a3i, a3j, a3mi, a3mj, a4mi, a4mj;
  double a1mag, a2mag, a3mag, a3amag, a4mag, a4amag;
  double shpi[3], shpj[3];
  double ai, aj, ro0i, ro0j, invrei, invrej;
  double rhoa0j, drhoa0j, rhoa0i, drhoa0i;
  double delrhoa0j, drhodmag0j, delrhoa0i, drhodmag0i;
  double delrhoa00j, drhodmag00j, delrhoa00i, drhodmag00i;
  double rhoa1j, drhoa1j, rhoa1i, drhoa1i;
  double delrhoa1j, drhodmag1j, delrhoa1i, drhodmag1i;
  double rhoa2j, drhoa2j, rhoa2i, drhoa2i;
  double delrhoa2j, drhodmag2j, delrhoa2i, drhodmag2i;
  double rhoa3j, drhoa3j, rhoa3i, drhoa3i;
  double delrhoa3j, drhodmag3j, delrhoa3i, drhodmag3i;
  double rhoa4j, drhoa4j, rhoa4i, drhoa4i;
  double delrhoa4j, drhodmag4j, delrhoa4i, drhodmag4i;
  double drhoa0mj, drhoa0mi, drhoa1mj, drhoa1mi;
  double drhoa2mj, drhoa2mi, drhoa3mj, drhoa3mi, drhoa4mj, drhoa4mi;
  double drho0dr1, drho0dr2, drho0ds1, drho0ds2, drho0dmag1, drho0dmag2;
  double drho1dr1, drho1dr2, drho1ds1, drho1ds2, drho1dmag1, drho1dmag2;
  double drho1drm1[3], drho1drm2[3];
  double drho1mdr1, drho1mdr2, drho1mds1, drho1mds2;
  double drho1mdrm1[3], drho1mdrm2[3];
  double drho2dr1, drho2dr2, drho2ds1, drho2ds2, drho2dmag1, drho2dmag2;
  double drho2drm1[3], drho2drm2[3];
  double drho2mdr1, drho2mdr2, drho2mds1, drho2mds2;
  double drho2mdrm1[3], drho2mdrm2[3];
  double drho3dr1, drho3dr2, drho3ds1, drho3ds2, drho3dmag1, drho3dmag2;
  double drho3drm1[3], drho3drm2[3];
  double drho3mdr1, drho3mdr2, drho3mds1, drho3mds2;
  double drho3mdrm1[3], drho3mdrm2[3];
  double drho4dr1, drho4dr2, drho4ds1, drho4ds2, drho4dmag1, drho4dmag2;
  double drho00dmag1, drho00dmag2;
  double drho4drm1[3], drho4drm2[3];
  double drho4mdr1, drho4mdr2, drho4mds1, drho4mds2;
  double drho4mdrm1[3], drho4mdrm2[3];
  double drhods1mag, drhods2mag;
  double drhodmag1, drhodmag2;
  double arg;
  double fm1[3],fm2[3];
  double arg1i1, arg1j1, arg1im1, arg1jm1, arg1i2, arg1j2, arg1im2, arg1jm2, arg1i3, arg1j3, arg3i3, arg3j3, arg1im3, arg1jm3, arg3im3, arg3jm3;
  double arg2im4, arg2jm4, arg4im4, arg4jm4;
  double dsij1, dsij2, dsij1mag, dsij2mag, force1, force2;
  double t1i, t2i, t3i, t1j, t2j, t3j, tm0i, tm0j, tm1i, tm1j, tm2i, tm2j, tm3i, tm3j, tm4i, tm4j, tm00i, tm00j;
  double scaleij;
  double eatm;
  double dU2dmag1, dU2dmag2, dU2dmag3, dU2dmag, dU2dmag00;
  double dU2dmagi[3], dU2dmag0i[3], dU2dmag1i[3], dU2dmag2i[3], dU2dmag3i[3], dU2dmag4i[3];
  double ddrho1mag[3][3], ddrho2mag[3][6], ddrho2bmag[3], ddrho3mag[3][10], ddrho3bmag[3][3], ddrho4mag[3][15], ddrho4bmag[3][6], ddrho4cmag[3];
  double ddrho1dmag1[3], ddrho3dmag1b[3], ddrho2dmag1[6], ddrho4dmag1b[6], ddrho3dmag1[10], ddrho4dmag1[15];
  double ddrho1dmag2, ddrho2dmag2, ddrho3dmag2, ddrho4dmag2, ddrho00dmag2, ddrho00dmag1;
  double ddrho2dmag1b, ddrho4dmag1c;
  double argd2im4, argd2im3, argd4im4, argd1im2, argd1im1, argd3im3, argd1im3;
  double argd2jm4, argd2jm3, argd4jm4, argd1jm2, argd1jm1, argd3jm3, argd1jm3;
  double dd1[6], dd2[6], dd3[6], dd3b[6], dd4[6], dd4b[6];
  int countr;
  third = 1.0 / 3.0;
  sixth = 1.0 / 6.0;

  //     Compute forces atom i

  elti = fmap[type[i]];
  if (elti < 0) return;

  xitmp = x[i][0];
  yitmp = x[i][1];
  zitmp = x[i][2];

  spi[0] = sp[i][0];
  spi[1] = sp[i][1];
  spi[2] = sp[i][2];

  fmi[0] = 0.0;
  fmi[1] = 0.0;
  fmi[2] = 0.0;

  fm1[0] = 0.0;
  fm1[1] = 0.0;
  fm1[2] = 0.0;

  fm2[0] = 0.0;
  fm2[1] = 0.0;
  fm2[2] = 0.0;

  countr = 0;
  for (m = 0; m < 3; m++) {
    dU2dmagi[m] = 0.0;
    dU2dmag0i[m] = 0.0;
    dU2dmag1i[m] = 0.0;
    dU2dmag2i[m] = 0.0;
    dU2dmag3i[m] = 0.0;
    dU2dmag4i[m] = 0.0;
    for (mm = m; mm < 3; mm++) {
      dd4[countr] = 0.;
      dd4b[countr] = 0.;
      dd3[countr] = 0.;
      dd3b[countr] = 0.;
      dd2[countr] = 0.;
      dd1[countr] = 0.;
      countr += 1;
    }
    nv2 = 0;
    nv3 = 0;
    nv4 = 0;
    for (n = 0; n < 3; n++) {
      for (p = n; p < 3; p++) {
	for (q = p; q < 3; q++) {
	  for (r = q; r < 3; r++) {
	    ddrho4mag[m][nv4] = 0.0;
	    nv4 += 1;
	  }
	  ddrho3mag[m][nv3] = 0.0;\
	  nv3 += 1;
	}
	ddrho2mag[m][nv2] = 0.0;
	ddrho4bmag[m][nv2] = 0.0;
	nv2 += 1;
      }
      ddrho3bmag[m][n] = 0.0;
      ddrho1mag[m][n] = 0.0;
    }
    ddrho2bmag[m] = 0.0;
    ddrho4cmag[m] = 0.0;
  }
  
  eatm=0.0;
  //     Treat each pair
  for (jn = 0; jn < numneigh; jn++) {
    j = firstneigh[jn];
    eltj = fmap[type[j]];
    scaleij = scale[type[i]][type[j]];

    if ((!iszero(scrfcn[fnoffset + jn]) || !iszero(scrfcnmag[fnoffset + jn]) || !iszero(scrfcnpair[fnoffset + jn])) && eltj >= 0) {

      sij = scrfcn[fnoffset + jn] * fcpair[fnoffset + jn];
      sijmag = scrfcnmag[fnoffset + jn] * fcpair[fnoffset + jn];
      sijpair = scrfcnpair[fnoffset + jn] * fcpair[fnoffset + jn];
      //sij = 1.0 ;//scrfcn[fnoffset + jn] * fcpair[fnoffset + jn];
      //sijmag = 1. ;//scrfcnmag[fnoffset + jn] * fcpair[fnoffset + jn];
      //sijpair = 1. ;//scrfcnpair[fnoffset + jn] * fcpair[fnoffset + jn];

      delij[0] = x[j][0] - xitmp;
      delij[1] = x[j][1] - yitmp;
      delij[2] = x[j][2] - zitmp;
      rij2 = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];

      spj[0] = sp[j][0];
      spj[1] = sp[j][1];
      spj[2] = sp[j][2];
      sdot = spi[0]*spj[0] + spi[1]*spj[1] + spi[2]*spj[2];
      smag = 0.5 * (1 - sdot);
      //smag = sdot;
      if (rij2 < this->cutforcesq) {
        rij = sqrt(rij2);
        recip = 1.0 / rij;
	//     Compute phim
        ind = this->eltind[elti][eltj];
        pp = rij * this->rdrar;
        kk = (int)pp;
        kk = std::min(kk, this->nrar - 2);
        pp = pp - kk;
        pp = std::min(pp, 1.0);
        phi = ((this->phirar3[ind][kk] * pp + this->phirar2[ind][kk]) * pp + this->phirar1[ind][kk]) * pp + this->phirar[ind][kk];
	//phim = ((this->magrar3[ind][kk] * pp + this->magrar2[ind][kk]) * pp + this->magrar1[ind][kk]) * pp + this->magrar[ind][kk];
	phim = ((this->magrar3[ind][kk] * pp + this->magrar2[ind][kk]) * pp + this->magrar1[ind][kk]) * pp + this->magrar[ind][kk];
	eatm = eatm + 0.5 * phi * sij + 0.5 * phim * sijpair * smag;
	//     write(1,*) "force_meamf: phi: ",phi
        //     write(1,*) "force_meamf: phip: ",phip

        //     Compute pair densities and derivatives
	invrei = 1.0 / this->re_meam[elti][elti];
        ai = rij * invrei - 1.0;

        ro0i = this->rho0_meam[elti];

	//	rhoa0i += delrhoa0i;

	//      Compute spin derivatives
	drhodmag0i = -ro0i * MathSpecial::fm_exp(-this->mag_beta0_meam[elti] * ai);
	drhodmag1i = -ro0i * MathSpecial::fm_exp(-this->mag_beta1_meam[elti] * ai);
	drhodmag2i = -ro0i * MathSpecial::fm_exp(-this->mag_beta2_meam[elti] * ai);
	drhodmag3i = -ro0i * MathSpecial::fm_exp(-this->mag_beta3_meam[elti] * ai);
	drhodmag4i = -ro0i * MathSpecial::fm_exp(-this->mag_beta4_meam[elti] * ai);

        if (elti != eltj) {
          invrej = 1.0 / this->re_meam[eltj][eltj];
          aj = rij * invrej - 1.0;
          ro0j = this->rho0_meam[eltj];

	  //	  rhoa0j += delrhoa0j;
	  
	  drhodmag0j = -ro0j * MathSpecial::fm_exp(-this->mag_beta0_meam[eltj] * aj);
	  drhodmag1j = -ro0j * MathSpecial::fm_exp(-this->mag_beta1_meam[eltj] * aj);
	  drhodmag2j = -ro0j * MathSpecial::fm_exp(-this->mag_beta2_meam[eltj] * aj);
	  drhodmag3j = -ro0j * MathSpecial::fm_exp(-this->mag_beta3_meam[eltj] * aj);
	  drhodmag4j = -ro0j * MathSpecial::fm_exp(-this->mag_beta4_meam[eltj] * aj);

	} else {
	  drhodmag0j = drhodmag0i;
	  drhodmag1j = drhodmag1i;
	  drhodmag2j = drhodmag2i;
	  drhodmag3j = drhodmag3i;
	  drhodmag4j = drhodmag4i;
        }

        const double t1mi = this->t1_meam[elti];
        const double t2mi = this->t2_meam[elti];
        const double t3mi = this->t3_meam[elti];
        const double tm0mi = this->mag_delrho0_meam[elti];
        const double tm1mi = this->mag_delrho1_meam[elti];
        const double tm2mi = this->mag_delrho2_meam[elti];
        const double tm3mi = this->mag_delrho3_meam[elti];
        const double tm4mi = this->mag_delrho4_meam[elti];
        const double t1mj = this->t1_meam[eltj];
        const double t2mj = this->t2_meam[eltj];
        const double t3mj = this->t3_meam[eltj];
        const double tm0mj = this->mag_delrho0_meam[eltj];
        const double tm1mj = this->mag_delrho1_meam[eltj];
        const double tm2mj = this->mag_delrho2_meam[eltj];
        const double tm3mj = this->mag_delrho3_meam[eltj];
        const double tm4mj = this->mag_delrho4_meam[eltj];

        if (this->ialloy == 1) {
          rhoa1j  *= t1mj;
          rhoa2j  *= t2mj;
          rhoa3j  *= t3mj;
          delrhoa0j  *= tm0mj;
          delrhoa1j  *= tm1mj;
          delrhoa2j  *= tm2mj;
          delrhoa3j  *= tm3mj;
          delrhoa4j  *= tm4mj;
          rhoa1i  *= t1mi;
          rhoa2i  *= t2mi;
          rhoa3i  *= t3mi;
          delrhoa0i  *= tm0mi;
          delrhoa1i  *= tm1mi;
          delrhoa2i  *= tm2mi;
          delrhoa3i  *= tm3mi;
          delrhoa4i  *= tm4mi;
          drhoa1j *= t1mj;
          drhoa2j *= t2mj;
          drhoa3j *= t3mj;
          drhoa0mj *= tm0mj;
          drhoa1mj *= tm1mj;
          drhoa2mj *= tm2mj;
          drhoa3mj *= tm3mj;
          drhoa4mj *= tm4mj;
          drhoa1i *= t1mi;
          drhoa2i *= t2mi;
          drhoa3i *= t3mi;
          drhoa0mi *= tm0mi;
          drhoa1mi *= tm1mi;
          drhoa2mi *= tm2mi;
          drhoa3mi *= tm3mi;
          drhoa4mi *= tm4mi;
	  //ADD dmag terms?
        }

        nv2 = 0;
        nv3 = 0;
	nv4 = 0;
        arg1i1 = 0.0;
        arg1j1 = 0.0;
        arg1im1 = 0.0;
        arg1jm1 = 0.0;
        arg1i2 = 0.0;
        arg1j2 = 0.0;
        arg1im2 = 0.0;
        arg1jm2 = 0.0;
        arg1i3 = 0.0;
        arg1j3 = 0.0;
        arg3i3 = 0.0;
        arg3j3 = 0.0;
        arg1im3 = 0.0;
        arg1jm3 = 0.0;
        arg3im3 = 0.0;
        arg3jm3 = 0.0;
        arg2im4 = 0.0;
        arg2jm4 = 0.0;
        arg4im4 = 0.0;
        arg4jm4 = 0.0;
	argd2im4 = 0.0;
	argd2im3 = 0.0;
	argd4im4 = 0.0;
	argd1im2 = 0.0;
	argd1im1 = 0.0;
	argd3im3 = 0.0;
	argd1im3 = 0.0;
	argd2jm4 = 0.0;
	argd2jm3 = 0.0;
	argd4jm4 = 0.0;
	argd1jm2 = 0.0;
	argd1jm1 = 0.0;
	argd3jm3 = 0.0;
	argd1jm3 = 0.0;
	
        for (n = 0; n < 3; n++) {
	  for (p = n; p < 3; p++) {
            for (q = p; q < 3; q++) {
	      for (r = q; r < 3; r++) {
		arg = delij[n] * delij[p] * delij[q] * delij[r] * this->v4D[nv4];
		arg2im4 = arg2im4 + arho4m[i][nv4] * arg;
		arg2jm4 = arg2jm4 + arho4m[j][nv4] * arg;
		argd2im4 = argd2im4 + arg * delij[n] * delij[p] * delij[q] * delij[r];
		argd2jm4 = argd2jm4 + arg * delij[n] * delij[p] * delij[q] * delij[r];
		nv4 = nv4 + 1;
	      }
              arg = delij[n] * delij[p] * delij[q] * this->v3D[nv3];
              arg1i3 = arg1i3 + arho3[i][nv3] * arg;
              arg1j3 = arg1j3 - arho3[j][nv3] * arg;
              arg1im3 = arg1im3 + arho3m[i][nv3] * arg;
              arg1jm3 = arg1jm3 - arho3m[j][nv3] * arg;
              argd1im3 = argd1im3 + arg * delij[n] * delij[p] * delij[q];
              argd1jm3 = argd1jm3 + arg * delij[n] * delij[p] * delij[q];
              nv3 = nv3 + 1;
            }
            arg = delij[n] * delij[p] * this->v2D[nv2];
            arg1i2 = arg1i2 + arho2[i][nv2] * arg;
            arg1j2 = arg1j2 + arho2[j][nv2] * arg;
            arg1im2 = arg1im2 + arho2m[i][nv2] * arg;
            arg1jm2 = arg1jm2 + arho2m[j][nv2] * arg;
            argd1im2 = argd1im2 + arg * delij[n] * delij[p];
            argd1jm2 = argd1jm2 + arg * delij[n] * delij[p];
            arg4im4 = arg4im4 + arho4bm[i][nv2] * arg;
            arg4jm4 = arg4jm4 + arho4bm[j][nv2] * arg;
            argd4im4 = argd4im4 + arg * delij[n] * delij[p];
            argd4jm4 = argd4jm4 + arg * delij[n] * delij[p];
            nv2 = nv2 + 1;
          }
          arg1i1 = arg1i1 + arho1[i][n] * delij[n];
          arg1j1 = arg1j1 - arho1[j][n] * delij[n];
          arg3i3 = arg3i3 + arho3b[i][n] * delij[n];
          arg3j3 = arg3j3 - arho3b[j][n] * delij[n];
          arg1im1 = arg1im1 + arho1m[i][n] * delij[n];
          arg1jm1 = arg1jm1 - arho1m[j][n] * delij[n];
          argd1im1 = argd1im1 + delij[n] * delij[n];
          argd1jm1 = argd1jm1 + delij[n] * delij[n];
          arg3im3 = arg3im3 + arho3bm[i][n] * delij[n];
          arg3jm3 = arg3jm3 - arho3bm[j][n] * delij[n];
          argd3im3 = argd3im3 + delij[n] * delij[n];
          argd3jm3 = argd3jm3 + delij[n] * delij[n];
        }

        nv2 = 0;
        nv3 = 0;
	nv4 = 0;

        //     rho0 terms
	drho0dmag1 = 2.0 * arho0m[i] * drhodmag0j * sijmag;
	drho0dmag2 = 2.0 * arho0m[j] * drhodmag0i * sijmag;

	//     rho00 terms
	drho00dmag1 = 2.0 * arho00[i] * drhodmag0j * sijmag * smag;
	drho00dmag2 = 2.0 * arho00[j] * drhodmag0i * sijmag * smag;

	ddrho00dmag1 = -1.0 * arho00[i] * drhodmag0j * sijmag;
	ddrho00dmag2 = -1.0 * arho00[j] * drhodmag0i * sijmag;

	
        //     rho1 terms
	a1mag = 2.0 * sijmag / rij;
	drho1dmag1 = a1mag * drhodmag1j * arg1im1;
	drho1dmag2 = a1mag * drhodmag1i * arg1jm1;
        for (n = 0; n < 3; n++) {
	  ddrho1dmag1[n] = drhodmag1j * sijmag *delij[n]/ rij;
	}
	ddrho1dmag2 = sijmag * sijmag * drhodmag1i * drhodmag1i * argd1jm1 / rij2;
	
        //     rho2 terms
	a2mag = 2. * sijmag /rij2;
	drho2dmag1 = a2mag * drhodmag2j * arg1im2 - 2.0 / 3.0 * arho2bm[i] * drhodmag2j * sijmag;
	drho2dmag2 = a2mag * drhodmag2i * arg1jm2 - 2.0 / 3.0 * arho2bm[j] * drhodmag2i * sijmag;
        for (n = 0; n < 3; n++) {
	  for (p = n; p < 3; p++) {
	    ddrho2dmag1[nv2] = drhodmag2j * sijmag * delij[n] * delij[p] / rij2;
	    nv2 = nv2 + 1;
	  }
	}
	ddrho2dmag1b =  drhodmag2j * sijmag;
	ddrho2dmag2 = sijmag * sijmag * drhodmag2i * drhodmag2i * (argd1jm2 / rij2 / rij2 - 1.0 / 3.0);
	

        //     rho3 terms
        rij3 = rij * rij2;
        a3mag = 2. * sijmag / rij3;
        a3amag = 6.0 / 5.0 * sijmag / rij;

        drho3dmag1 = a3mag * drhodmag3j * arg1im3 - a3amag * drhodmag3j * arg3im3;
        drho3dmag2 = a3mag * drhodmag3i * arg1jm3 - a3amag * drhodmag3i * arg3jm3;
        for (n = 0; n < 3; n++) {
	  for (p = n; p < 3; p++) {
	    for (q = p; q < 3; q++) {
	      ddrho3dmag1[nv3] = drhodmag3j * sijmag * delij[n] * delij[p] *delij[q] / rij3;
	      nv3 = nv3 + 1;
	    }
	  }
	  ddrho3dmag1b[n] =  drhodmag3j * sijmag * delij[n] / rij;
	}
        ddrho3dmag2 = sijmag * sijmag * drhodmag3i * drhodmag3i * (argd1jm3 / rij3 / rij3 - 3.0 / 5.0 * argd3jm3 / rij2);

        //     rho4 terms
        rij4 = rij2 * rij2;
        a4mag = 2.0 * sijmag / rij4;
        a4amag = 12.0 / 7.0 * sijmag / rij2;

        drho4dmag1 = a4mag * drhodmag4j * arg2im4 - a4amag * drhodmag4j * arg4im4 + 6.0 / 35.0 * arho4cm[i] * drhodmag4j * sijmag;
        drho4dmag2 = a4mag * drhodmag4i * arg2jm4 - a4amag * drhodmag4i * arg4jm4 + 6.0 / 35.0 * arho4cm[j] * drhodmag4i * sijmag;

	nv2 = 0;
        for (n = 0; n < 3; n++) {
	  for (p = n; p < 3; p++) {
	    for (q = p; q < 3; q++) {
	      for (r = q; r <3; r++) {
		ddrho4dmag1[nv4] = drhodmag4j * sijmag * delij[n] * delij[p] * delij[q] * delij[r] / rij4;
		nv4 = nv4 + 1;
	      }
	    }
	    ddrho4dmag1b[nv2] = drhodmag4j * sijmag * delij[n] * delij[p] / rij2;
	    nv2 = nv2 + 1;
	  }
	}
	ddrho4dmag1c = drhodmag4j * sijmag;

        ddrho4dmag2 = sijmag * sijmag * drhodmag4i * drhodmag4i * (argd2jm4 / rij4 / rij4 - 6.0 / 7.0 * argd4jm4 / rij4 + 3.0 / 35.0);

        //     Compute derivatives of weighting functions t wrt rij
        t1i = t_ave[i][0];
        t2i = t_ave[i][1];
        t3i = t_ave[i][2];
        tm0i = t_ave[i][3];
        tm1i = t_ave[i][4];
        tm2i = t_ave[i][5];
        tm3i = t_ave[i][6];
        tm4i = t_ave[i][7];
        tm00i = t_ave[i][8];
        t1j = t_ave[j][0];
        t2j = t_ave[j][1];
        t3j = t_ave[j][2];
        tm0j = t_ave[j][3];
        tm1j = t_ave[j][4];
        tm2j = t_ave[j][5];
        tm3j = t_ave[j][6];
        tm4j = t_ave[j][7];
        tm00j = t_ave[j][8];

	drhodmag1 = (tm0i * drho0dmag1 + tm1i * drho1dmag1 + tm2i * drho2dmag1 + tm3i * drho3dmag1 + tm4i * drho4dmag1 + tm00i * drho00dmag1);
	drhodmag2 = (tm0j * drho0dmag2 + tm1j * drho1dmag2 + tm2j * drho2dmag2 + tm3j * drho3dmag2 + tm4j * drho4dmag2 + tm00j * drho00dmag2);
        
	dU2dmag1 = 0.25 * frhopp[j] * dgamma2[j] * dgamma2[j] * drhodmag2 * drhodmag2;
	dU2dmag2 = 0.25 * frhop[j] * d2gamma2[j] * drhodmag2 * drhodmag2;
	dU2dmag3 = 0.5 * frhop[j] * dgamma2[j] * (tm0j * drhodmag0i * drhodmag0i * sijmag * sijmag + tm1j * ddrho1dmag2 + tm2j * ddrho2dmag2 + tm3j * ddrho3dmag2 + tm4j * ddrho4dmag2 + tm00j * ddrho00dmag2);
	dU2dmag00 = 0.5 * frhop[i] * dgamma2[i] * tm00i * ddrho00dmag1;
	//	if (i==0) std::cout << "dU2d1 = " << dU2dmag1 << " dU2d2 = " << dU2dmag2 << " dU2d3 = " << dU2dmag3 << "\n";
	dU2dmag = dU2dmag1 + dU2dmag2 + dU2dmag3 + dU2dmag00;
	dUdmag1 = frhop[i] * dgamma2[i] * drhodmag1;// + frhop[j] * drhodmag2;
	dUdmag2 = frhop[j] * dgamma2[j] * drhodmag2;
	dUdmag = dUdmag1 + dUdmag2;
	countr = 0;
	for (m = 0; m < 3; m++) {
	  fm1[m] = fm1[m] - 0.5 * spj[m] * (phim * sijpair);
	  fm2[m] = fm2[m] + 0.5 * spj[m] * (dUdmag);
	  fmi[m] = fmi[m] + 0.5 * spj[m] * (phim * sijpair - dUdmag);	  
	  dU2dmagi[m] += drhodmag1 * spj[m];
	  dU2dmag0i[m] += (drhodmag0j * sijmag * spj[m]);
	  dU2dmag1i[m] += (drhodmag1j * sijmag * spj[m]);
	  dU2dmag2i[m] += (drhodmag2j * sijmag * spj[m]);
	  dU2dmag3i[m] += (drhodmag3j * sijmag * spj[m]);
	  dU2dmag4i[m] += (drhodmag4j * sijmag * spj[m]);
	  nv2 = 0;
	  nv3 = 0;
	  nv4 = 0;
	  for (n = 0; n < 3; n++) {
	    for (p = n; p < 3; p++) {
	      for (q = p; q < 3; q++) {
		for (r = q; r < 3; r++) {
		  ddrho4mag[m][nv4] += ddrho4dmag1[nv4] * spj[m];
		  nv4 += 1;
		}
		ddrho3mag[m][nv3] += ddrho3dmag1[nv3] * spj[m];
		nv3 += 1;
	      }
	      ddrho2mag[m][nv2] += ddrho2dmag1[nv2] * spj[m];
	      ddrho4bmag[m][nv2] += ddrho4dmag1b[nv2] * spj[m];
	      nv2 += 1;
	    }
	    ddrho1mag[m][n] += ddrho1dmag1[n] * spj[m];
	    ddrho3bmag[m][n] += ddrho3dmag1b[n] * spj[m];
	  }
	  ddrho2bmag[m] += ddrho2dmag1b * spj[m];
	  ddrho4cmag[m] += ddrho4dmag1c * spj[m];
	  for (n = m; n < 3; n++) {
	    fmdsi[countr] = fmdsi[countr] - 1.0 * spj[m] * spj[n] * dU2dmag;
	    countr += 1;
	  }
	}
      }
    }
  }
  countr = 0;
  for (m = 0; m < 3; m++) {
    for (mm = m; mm < 3; mm++) {
      nv2 = 0;
      nv3 = 0;
      nv4 = 0;
      for (n = 0; n < 3; n++) {
	for (p = n; p < 3; p++) {
	  for (q = p; q < 3; q++) {
	    for (r = q; r < 3; r++) {
	      dd4[countr] += ddrho4mag[m][nv4]*ddrho4mag[mm][nv4] * this->v4D[nv4];
	      nv4 += 1;
	    }
	    dd3[countr] += ddrho3mag[m][nv3]*ddrho3mag[mm][nv3] * this->v3D[nv3];
	    nv3 += 1;
	  }
	  dd2[countr] += ddrho2mag[m][nv2]*ddrho2mag[mm][nv2] * this->v2D[nv2];
	  dd4b[countr] += ddrho4bmag[m][nv2]*ddrho4bmag[mm][nv2] * this->v2D[nv2];
	  nv2 += 1;
	}
	dd1[countr] += ddrho1mag[m][n]*ddrho1mag[mm][n];
	dd3b[countr] += ddrho3bmag[m][n]*ddrho3bmag[mm][n];
      }
      fmdsi[countr] = fmdsi[countr] - 0.25 * frhopp[i] * dgamma2[i] * dgamma2[i] *dU2dmagi[m]*dU2dmagi[mm];
      fmdsi[countr] = fmdsi[countr] - 0.25 * frhop[i] * d2gamma2[i] * dU2dmagi[m] * dU2dmagi[mm];
      //      fmdsi[countr] = fmdsi[countr] + 0.5 * frhop[i] * dgamma2[i] * (tm0i * dU2dmag0i[m] * dU2dmag0i[n] + tm1i * dU2dmag1i[m] * dU2dmag1i[n] + tm2i * dU2dmag2i[m] * dU2dmag2i[n] + tm3i * dU2dmag3i[m] * dU2dmag3i[n] + tm4i * dU2dmag4i[m] * dU2dmag4i[n]);
      fmdsi[countr] -= 0.5 * frhop[i] * dgamma2[i] * (tm0i * dU2dmag0i[m] * dU2dmag0i[mm] + tm1i * dd1[countr] + tm2i * (dd2[countr] - 1.0 /3.0 * ddrho2bmag[m] * ddrho2bmag[mm]));
      fmdsi[countr] -= 0.5 * frhop[i] * dgamma2[i] * (tm3i * (dd3[countr] - 3.0 / 5.0 * dd3b[countr]) + tm4i * (dd4[countr] - 6.0 / 7.0 * dd4b[countr] + 3.0 / 35.0 * ddrho4cmag[m] * ddrho4cmag[mm]));
      countr += 1;
    }
  }
}
