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

#include "math_const.h"

#include <algorithm>
#include <cmath>

using namespace LAMMPS_NS;
using MathConst::MY_PI;

//     do a sanity check on index parameters
void
MAGMEAM::meam_checkindex(int num, int lim, int nidx, int* idx /*idx(3)*/, int* ierr)
{
  //: idx[0..2]
  *ierr = 0;
  if (nidx < num) {
    *ierr = 2;
    return;
  }

  for (int i = 0; i < num; i++) {
    if ((idx[i] < 0) || (idx[i] >= lim)) {
      *ierr = 3;
      return;
    }
  }
}

//     The "which" argument corresponds to the index of the "keyword" array
//     in pair_meam.cpp:
//
//     0 = Ec_meam
//     1 = alpha_meam
//     2 = rho0_meam
//     3 = delta_meam
//     4 = lattce_meam
//     5 = attrac_meam
//     6 = repuls_meam
//     7 = nn2_meam
//     8 = Cmin_meam
//     9 = Cmax_meam
//     10 = rc_meam
//     11 = delr_meam
//     12 = augt1
//     13 = gsmooth_factor
//     14 = re_meam
//     15 = ialloy
//     16 = mixture_ref_t
//     17 = erose_form
//     18 = zbl_meam
//     19 = emb_lin_neg
//     20 = bkgd_dyn
//     21 = theta
//     22 = imag
//     23 = mag_Ec
//     24 = mag_alpha
//     25 = mag_re
//     26 = mag_lattce
//     27 = mag_attrac
//     28 = mag_repuls
//     29 = mag_B
//     30 = mag_delrho0
//     31 = mag_delrho1
//     32 = mag_delrho2
//     33 = mag_delrho3
//     34 = mag_delrho4
//     35 = mag_delrho00
//     36 = mag_beta0
//     37 = mag_beta1
//     38 = mag_beta2
//     39 = mag_beta3
//     40 = mag_beta4
//     41 = mag_beta00
//     42 = Cmin_magmeam
//     43 = Cmax_magmeam
//     44 = Cmin_magpair
//     45 = Cmax_magpair

//     The returned errorflag has the following meanings:

//     0 = no error
//     1 = "which" out of range / invalid keyword
//     2 = not enough indices given
//     3 = an element index is out of range

void
MAGMEAM::meam_setup_param(int which, double value, int nindex, int* index /*index(3)*/, int* errorflag)
{
  //: index[0..2]
  int i1, i2;
  lattice_t vlat;
  *errorflag = 0;

  switch (which) {
    //     0 = Ec_meam
    case 0:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->Ec_meam[index[0]][index[1]] = value;
      break;

    //     1 = alpha_meam
    case 1:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->alpha_meam[index[0]][index[1]] = value;
      break;

    //     2 = rho0_meam
    case 2:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->rho0_meam[index[0]] = value;
      break;

    //     3 = delta_meam
    case 3:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->delta_meam[index[0]][index[1]] = value;
      break;

    //     4 = lattce_meam
    case 4:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      vlat = (lattice_t)value;

      this->lattce_meam[index[0]][index[1]] = vlat;
      break;

    //     5 = attrac_meam
    case 5:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->attrac_meam[index[0]][index[1]] = value;
      break;

    //     6 = repuls_meam
    case 6:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->repuls_meam[index[0]][index[1]] = value;
      break;

    //     7 = nn2_meam
    case 7:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      i1 = std::min(index[0], index[1]);
      i2 = std::max(index[0], index[1]);
      this->nn2_meam[i1][i2] = (int)value;
      break;

    //     8 = Cmin_meam
    case 8:
      meam_checkindex(3, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->Cmin_meam[index[0]][index[1]][index[2]] = value;
      break;

    //     9 = Cmax_meam
    case 9:
      meam_checkindex(3, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->Cmax_meam[index[0]][index[1]][index[2]] = value;
      break;

    //     10 = rc_meam
    case 10:
      this->rc_meam = value;
      break;

    //     11 = delr_meam
    case 11:
      this->delr_meam = value;
      break;

    //     12 = augt1
    case 12:
      this->augt1 = (int)value;
      break;

    //     13 = gsmooth
    case 13:
      this->gsmooth_factor = value;
      break;

    //     14 = re_meam
    case 14:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->re_meam[index[0]][index[1]] = value;
      break;

    //     15 = ialloy
    case 15:
      this->ialloy = (int)value;
      break;

    //     16 = mixture_ref_t
    case 16:
      this->mix_ref_t = (int)value;
      break;

    //     17 = erose_form
    case 17:
      this->erose_form = (int)value;
      break;

    //     18 = zbl_meam
    case 18:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      i1 = std::min(index[0], index[1]);
      i2 = std::max(index[0], index[1]);
      this->zbl_meam[i1][i2] = (int)value;
      break;

    //     19 = emb_lin_neg
    case 19:
      this->emb_lin_neg = (int)value;
      break;

    //     20 = bkgd_dyn
    case 20:
      this->bkgd_dyn = (int)value;
      break;

    //     21 = theta
    // see alloyparams(void) in meam_setup_done.cpp
    case 21:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      i1 = std::min(index[0], index[1]);
      i2 = std::max(index[0], index[1]);
      // we don't use theta, instead stheta and ctheta
      this->stheta_meam[i1][i2] = sin(value/2*MY_PI/180.0);
      this->ctheta_meam[i1][i2] = cos(value/2*MY_PI/180.0);
      break;

    //     22 = imag
    case 22:
      this->imag = (int)value;
      break;

    //     23 = mag_Ec
    case 23:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->mag_Ec[index[0]][index[1]] = value;
      break;

    //     24 = mag_alpha
    case 24:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->mag_alpha[index[0]][index[1]] = value;
      break;

    //     25 = mag_re
    case 25:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->mag_re[index[0]][index[1]] = value;
      break;

    //     26 = mag_lattce
    case 26:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      vlat = (lattice_t)value;
      this->mag_lattce[index[0]][index[1]] = vlat;
      break;

    //     27 = mag_attrac
    case 27:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->mag_attrac[index[0]][index[1]] = value;
      break;

    //     28 = repuls_meam
    case 28:
      meam_checkindex(2, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->mag_repuls[index[0]][index[1]] = value;
      break;

    //     29 = mag_B
    case 29:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_B_meam[index[0]] = value;
      break;

    //     30 = mag_delrho0
    case 30:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_delrho0_meam[index[0]] = value;
      break;

    //     31 = mag_delrho1
    case 31:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_delrho1_meam[index[0]] = value;
      break;

    //     32 = mag_delrho2
    case 32:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_delrho2_meam[index[0]] = value;
      break;

    //     33 = mag_delrho3
    case 33:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_delrho3_meam[index[0]] = value;
      break;

    //     34 = mag_delrho4
    case 34:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_delrho4_meam[index[0]] = value;
      break;

    //     35 = mag_delrho00
    case 35:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_delrho00_meam[index[0]] = value;
      break;

    //     36 = mag_beta0
    case 36:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_beta0_meam[index[0]] = value;
      break;

    //     37 = mag_beta1
    case 37:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_beta1_meam[index[0]] = value;
      break;

    //     38 = mag_beta2
    case 38:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_beta2_meam[index[0]] = value;
      break;
      
    //     39 = mag_beta3
    case 39:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_beta3_meam[index[0]] = value;
      break;

    //     40 = mag_beta4
    case 40:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_beta4_meam[index[0]] = value;
      break;

    //     41 = mag_beta00
    case 41:
      meam_checkindex(1, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
	return;
      this->mag_beta00_meam[index[0]] = value;
      break;

    //     42 = Cmin_magmeam
    case 42:
      meam_checkindex(3, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->Cmin_magmeam[index[0]][index[1]][index[2]] = value;
      break;

    //     43 = Cmax_magmeam
    case 43:
      meam_checkindex(3, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->Cmax_magmeam[index[0]][index[1]][index[2]] = value;
      break;

    //     44 = Cmin_magpair
    case 44:
      meam_checkindex(3, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->Cmin_magpair[index[0]][index[1]][index[2]] = value;
      break;

    //     45 = Cmax_magpair
    case 45:
      meam_checkindex(3, neltypes, nindex, index, errorflag);
      if (*errorflag != 0)
        return;
      this->Cmax_magpair[index[0]][index[1]][index[2]] = value;
      break;

    default:
      *errorflag = 1;
  }
}
