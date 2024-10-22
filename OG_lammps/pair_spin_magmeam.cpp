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
   Contributing author: Greg Wagner (SNL)
------------------------------------------------------------------------- */

#include "pair_spin_magmeam.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "spin_magmeam.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <iostream>
using namespace LAMMPS_NS;

#define MAXLINE 1024

static const int nkeywords = 46;
static const char *keywords[] = {
  "Ec","alpha","rho0","delta","lattce",
  "attrac","repuls","nn2","Cmin","Cmax","rc","delr",
  "augt1","gsmooth_factor","re","ialloy",
  "mixture_ref_t","erose_form","zbl",
  "emb_lin_neg","bkgd_dyn", "theta","imag","mag_Ec","mag_alpha","mag_re","mag_lattce","mag_attrac","mag_repuls",
  "mag_B","mag_delrho0","mag_delrho1","mag_delrho2","mag_delrho3","mag_delrho4","mag_delrho00",
  "mag_beta0","mag_beta1","mag_beta2","mag_beta3","mag_beta4","mag_beta00","Cminmag","Cmaxmag","Cminpair","Cmaxpair"};

/* ---------------------------------------------------------------------- */

PairMAGMEAM::PairMAGMEAM(LAMMPS *lmp) : PairSpin(lmp)
{
  hbar = force->hplanck/6.283185307179586476925286766559;
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;

  allocated = 0;

  nlibelements = 0;
  meam_inst = new MAGMEAM(memory);
  scale = nullptr;
  // set comm size needed by this Pair

  comm_forward = 90;
  comm_reverse = 75;
}

/* ----------------------------------------------------------------------
   free all arrays
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairMAGMEAM::~PairMAGMEAM()
{
  delete meam_inst;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
    memory->destroy(emag);
  }
}

/* ---------------------------------------------------------------------- */

void PairMAGMEAM::compute(int eflag, int vflag)
{
  int i,ii,n,inum_half,inum_full,errorflag;
  int *ilist_half,*ilist_full,*numneigh_half,**firstneigh_half;
  int *numneigh_full,**firstneigh_full;
  
  ev_init(eflag,vflag);

  // neighbor list info
  
  inum_half = listhalf->inum;
  ilist_half = listhalf->ilist;
  numneigh_half = listhalf->numneigh;
  firstneigh_half = listhalf->firstneigh;
  inum_full = listfull->inum;
  ilist_full = listfull->ilist;
  numneigh_full = listfull->numneigh;
  firstneigh_full = listfull->firstneigh;

  // strip neighbor lists of any special bond flags before using with MEAM
  // necessary before doing neigh_f2c and neigh_c2f conversions each step

  if (neighbor->ago == 0) {
    neigh_strip(inum_full,ilist_full,numneigh_half,firstneigh_half);
    neigh_strip(inum_full,ilist_full,numneigh_full,firstneigh_full);
  }

  // check size of scrfcn based on half neighbor list

  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;

  if (nlocal_max < nlocal) {    // grow emag lists if necessary
    nlocal_max = nlocal;
    memory->grow(emag,nlocal_max,"pair/spin:emag");
  }

  n = 0;
  for (ii = 0; ii < inum_half; ii++) n += numneigh_full[ilist_full[ii]];

  meam_inst->meam_dens_setup(atom->nmax, nall, n);

  double **x = atom->x;
  double **f = atom->f;
  double **fm = atom->fm;
  double **fmds = atom->fmds;
  double **sp = atom->sp;
  int *type = atom->type;
  int ntype = atom->ntypes;

  // 3 stages of MEAM calculation
  // loop over my atoms followed by communication

  memory->destroy(offinv);
  memory->create(offinv,inum_full, "pair:offinv");

  int offset = 0;
  errorflag = 0;
  

  for (ii = 0; ii < inum_full; ii++) {
    offinv[ii] = offset;
    i = ilist_full[ii];
    meam_inst->meam_dens_init(i,ntype,type,map,x,sp,
                    numneigh_half[i],firstneigh_half[i],
                    numneigh_full[i],firstneigh_full[i],
                    offset);
    offset += numneigh_full[i];
  }

  comm->reverse_comm(this);

  int engflag = 0;
  meam_inst->meam_dens_final(nlocal,eflag_either,eflag_global,eflag_atom,
			     &eng_vdwl,eatom,ntype,type,map,scale,errorflag,engflag);

  if (errorflag)
    error->one(FLERR,"MEAM library error {}",errorflag);

  comm->forward_comm(this);

  offset = 0;

  // vptr is first value in vatom if it will be used by meam_force()
  // else vatom may not exist, so pass dummy ptr

  double **vptr = nullptr;
  if (vflag_atom) vptr = vatom;

  vflag_global = 1;
  for (ii = 0; ii < inum_full; ii++) {
    i = ilist_full[ii];
    meam_inst->meam_force(i,eflag_global,eflag_atom,vflag_global,
                          vflag_atom,&eng_vdwl,eatom,ntype,type,map,scale,x,sp,
                          numneigh_half[i],firstneigh_half[i],
                          numneigh_full[i],firstneigh_full[i],
                          offset,f,fm,fmds,vptr,virial,hbar,emag[i]);
    offset += numneigh_full[i];
    fm[i][0] = fm[i][0] / hbar;
    fm[i][1] = fm[i][1] / hbar;
    fm[i][2] = fm[i][2] / hbar;
    fmds[i][0] = fmds[i][0] / hbar;
    fmds[i][1] = fmds[i][1] / hbar;
    fmds[i][2] = fmds[i][2] / hbar;
    fmds[i][3] = fmds[i][3] / hbar;
    fmds[i][4] = fmds[i][4] / hbar;
    fmds[i][5] = fmds[i][5] / hbar;
  }
  //if (vflag_fdotr) virial_fdotr_compute();

}

/* ---------------------------------------------------------------------- */

void PairMAGMEAM::compute_single_pair(int ii, double fmi[3], double fmdsi[6])
{

  int i,ix,n,inum_half,inum_full,errorflag;
  int *ilist_half,*ilist_full,*numneigh_half,**firstneigh_half;
  int *numneigh_full,**firstneigh_full;

  inum_half = listhalf->inum;
  ilist_half = listhalf->ilist;
  numneigh_half = listhalf->numneigh;
  firstneigh_half = listhalf->firstneigh;
  inum_full = listfull->inum;
  ilist_full = listfull->ilist;
  numneigh_full = listfull->numneigh;
  firstneigh_full = listfull->firstneigh;

  
  // strip neighbor lists of any special bond flags before using with MEAM
  // necessary before doing neigh_f2c and neigh_c2f conversions each step

  if (neighbor->ago == 0) {
    neigh_strip(inum_full,ilist_full,numneigh_half,firstneigh_half);
    neigh_strip(inum_full,ilist_full,numneigh_full,firstneigh_full);
  }

  // check size of scrfcn based on half neighbor list

  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;

  n = 0;
  for (i = 0; i < inum_half; i++) n += numneigh_full[ilist_full[i]];

  meam_inst->meam_dens_setup(atom->nmax, nall, n);

  double **x = atom->x;
  double **f = atom->f;
  double **sp = atom->sp;
  int *type = atom->type;
  int ntype = atom->ntypes;

  // 3 stages of MEAM calculation
  // loop over my atoms followed by communication

  int offset = 0;
  errorflag = 0;
  

  for (i = 0; i < inum_full; i++) {
    ix = ilist_full[i];
    meam_inst->meam_dens_update(ix,ntype,type,map,x,sp,
                    numneigh_half[ix],firstneigh_half[ix],
                    numneigh_full[ix],firstneigh_full[ix],
                    offset);
    offset += numneigh_full[ix];
  }

  comm->reverse_comm(this);


  int engflag = 1;
  for (i = 0; i < inum_full; i++) {
    ix = ilist_full[i];
    meam_inst->update_dens_final(ix,ntype,type,map,scale,errorflag);
  }
  if (errorflag)
    error->one(FLERR,"MEAM library error {}",errorflag);

  comm->forward_comm(this);

  offset = 0;
  
  i = ilist_full[ii];

  meam_inst->meam_spin(i,ntype,type,map,scale,x,sp,numneigh_full[i],
		       firstneigh_full[i],offinv[i],fmi,fmdsi,hbar);
  fmi[0] = fmi[0] / hbar;
  fmi[1] = fmi[1] / hbar;
  fmi[2] = fmi[2] / hbar;

  fmdsi[0] = fmdsi[0] / hbar;
  fmdsi[1] = fmdsi[1] / hbar;
  fmdsi[2] = fmdsi[2] / hbar;
  fmdsi[3] = fmdsi[3] / hbar;
  fmdsi[4] = fmdsi[4] / hbar;
  fmdsi[5] = fmdsi[5] / hbar;
}

void PairMAGMEAM::allocate()
{

  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(scale,n+1,n+1,"pair:scale");

  map = new int[n+1];

}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMAGMEAM::settings(int narg, char ** /*arg*/)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMAGMEAM::coeff(int narg, char **arg)
{
  int m,n;

  if (!allocated) allocate();

  if (narg < 6) error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // check for presence of first meam file

  std::string lib_file = utils::get_potential_file_path(arg[2]);
  if (lib_file.empty())
    error->all(FLERR,"Cannot open MEAM library file {}",lib_file);

  // find meam parameter file in arguments:
  // first word that is a file or "NULL" after the MEAM library file
  // we need to extract at least one element, so start from index 4

  int paridx=-1;
  std::string par_file;
  for (int i = 4; i < narg; ++i) {
    if (strcmp(arg[i],"NULL") == 0) {
      par_file = "NULL";
      paridx = i;
      break;
    }
    par_file = utils::get_potential_file_path(arg[i]);
    if (!par_file.empty()) {
      paridx=i;
      break;
    }
  }
  if (paridx < 0) error->all(FLERR,"No MEAM parameter file in pair coefficients");
  if ((narg - paridx - 1) != atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // MEAM element names between 2 filenames
  // nlibelements = # of MEAM elements
  // elements = list of unique element names

  if (nlibelements) {
    libelements.clear();
    mass.clear();
  }

  nlibelements = paridx - 3;
  if (nlibelements < 1) error->all(FLERR,"Incorrect args for pair coefficients");
  if (nlibelements > maxelt)
    error->all(FLERR,"Too many elements extracted from MEAM library (current limit: {}). "
               "Increase 'maxelt' in meam.h and recompile.", maxelt);

  for (int i = 0; i < nlibelements; i++) {
    if (std::any_of(libelements.begin(), libelements.end(),
                    [&](const std::string &elem) { return elem == arg[i+3]; }))
      error->all(FLERR,"Must not extract the same element ({}) from MEAM library twice. ", arg[i+3]);

    libelements.emplace_back(arg[i+3]);
    mass.push_back(0.0);
  }

  // read MEAM library and parameter files
  // pass all parameters to MEAM package
  // tell MEAM package that setup is done
  read_files(lib_file,par_file);
  meam_inst->meam_setup_done(&cutmax);

  // read args that map atom types to MEAM elements
  // map[i] = which element the Ith atom type is, -1 if not mapped

  for (int i = 4 + nlibelements; i < narg; i++) {
    m = i - (4+nlibelements) + 1;
    int j;
    for (j = 0; j < nlibelements; j++)
      if (libelements[j] == arg[i]) break;
    if (j < nlibelements) map[m] = j;
    else if (strcmp(arg[i],"NULL") == 0) map[m] = -1;
    else error->all(FLERR,"Incorrect args for pair coefficients");
  }

  // clear setflag since coeff() called once with I,J = * *

  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements
  // set mass for i,i in atom class

  int count = 0;
  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) {
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        if (i == j) atom->set_mass(FLERR,i,mass[map[i]]);
        count++;
      }
      scale[i][j] = 1.0;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");

}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairMAGMEAM::init_style()
{
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style MEAM requires newton pair on");

  // need a full and a half neighbor list

  neighbor->add_request(this, NeighConst::REQ_FULL)->set_id(1);
  neighbor->add_request(this)->set_id(2);

  nlocal_max = atom->nlocal;
  memory->grow(emag,nlocal_max,"pair/spin:emag");
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   half or full
------------------------------------------------------------------------- */

void PairMAGMEAM::init_list(int id, NeighList *ptr)
{
  if (id == 1) listfull = ptr;
  else if (id == 2) listhalf = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMAGMEAM::init_one(int i, int j)
{
  if (setflag[i][j] == 0) scale[i][j] = 1.0;
  scale[j][i] = scale[i][j];
  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairMAGMEAM::read_files(const std::string &globalfile,
                           const std::string &userfile)
{
  read_global_meam_file(globalfile);
  read_user_meam_file(userfile);
}

/* ---------------------------------------------------------------------- */

void PairMAGMEAM::read_global_meam_file(const std::string &globalfile)
{
  // allocate parameter arrays
  std::vector<lattice_t> lat(nlibelements);
  std::vector<int> ielement(nlibelements);
  std::vector<int> ibar(nlibelements);
  std::vector<double> z(nlibelements);
  std::vector<double> atwt(nlibelements);
  std::vector<double> alpha(nlibelements);
  std::vector<double> b0(nlibelements);
  std::vector<double> b1(nlibelements);
  std::vector<double> b2(nlibelements);
  std::vector<double> b3(nlibelements);
  std::vector<double> alat(nlibelements);
  std::vector<double> esub(nlibelements);
  std::vector<double> asub(nlibelements);
  std::vector<double> t0(nlibelements);
  std::vector<double> t1(nlibelements);
  std::vector<double> t2(nlibelements);
  std::vector<double> t3(nlibelements);
  std::vector<double> rozero(nlibelements);
  std::vector<bool> found(nlibelements, false);

  // open global meamf file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, globalfile, "MAGMEAM", " library");
    char * line;

    constexpr int params_per_line = 19;
    int nset = 0;

    while ((line = reader.next_line(params_per_line))) {
      try {
        ValueTokenizer values(line, "' \t\n\r\f");

        // read each set of params from global MEAM file
        // one set of params can span multiple lines
        // store params if element name is in element list
        // if element name appears multiple times, only store 1st entry
        std::string element = values.next_string();

        // skip if element name isn't in element list

        int index;
        for (index = 0; index < nlibelements; index++)
          if (libelements[index] == element) break;
        if (index == nlibelements) continue;

        // skip if element already appeared (technically error in library file, but always ignored)

        if (found[index]) continue;
        found[index] = true;

        // map lat string to an integer
        std::string lattice_type = values.next_string();

        if (!MAGMEAM::str_to_lat(lattice_type, true, lat[index]))
          error->one(FLERR,"Unrecognized lattice type in MEAM "
                                       "library file: {}", lattice_type);

        // store parameters

        z[index] = values.next_double();
        ielement[index] = values.next_int();
        atwt[index] = values.next_double();
        alpha[index] = values.next_double();
        b0[index] = values.next_double();
        b1[index] = values.next_double();
        b2[index] = values.next_double();
        b3[index] = values.next_double();
        alat[index] = values.next_double();
        esub[index] = values.next_double();
        asub[index] = values.next_double();
        t0[index] = values.next_double();
        t1[index] = values.next_double();
        t2[index] = values.next_double();
        t3[index] = values.next_double();
        rozero[index] = values.next_double();
        ibar[index] = values.next_int();

        if (!isone(t0[index]))
          error->one(FLERR,"Unsupported parameter in MEAM library file: t0!=1");

        // z given is ignored: if this is mismatched, we definitely won't do what the user said -> fatal error
        if (z[index] != MAGMEAM::get_Zij(lat[index]))
          error->one(FLERR,"Mismatched parameter in MEAM library file: z!=lat");

        nset++;
      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }
    }

    // error if didn't find all elements in file

    if (nset != nlibelements) {
      std::string msg = "Did not find all elements in MEAM library file, missing:";
      for (int i = 0; i < nlibelements; i++)
        if (!found[i]) {
          msg += " ";
          msg += libelements[i];
        }
      error->one(FLERR,msg);
    }
  }

  // distribute complete parameter sets
  MPI_Bcast(lat.data(), nlibelements, MPI_INT, 0, world);
  MPI_Bcast(ielement.data(), nlibelements, MPI_INT, 0, world);
  MPI_Bcast(ibar.data(), nlibelements, MPI_INT, 0, world);
  MPI_Bcast(z.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(atwt.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(alpha.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(b0.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(b1.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(b2.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(b3.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(alat.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(esub.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(asub.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(t0.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(t1.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(t2.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(t3.data(), nlibelements, MPI_DOUBLE, 0, world);
  MPI_Bcast(rozero.data(), nlibelements, MPI_DOUBLE, 0, world);

  // pass element parameters to MEAM package
  
  meam_inst->meam_setup_global(nlibelements, lat.data(), ielement.data(), atwt.data(),
                               alpha.data(), b0.data(), b1.data(), b2.data(), b3.data(),
                               alat.data(), esub.data(), asub.data(), t0.data(), t1.data(),
                               t2.data(), t3.data(), rozero.data(), ibar.data());

  // set element masses
  
  for (int i = 0; i < nlibelements; i++) mass[i] = atwt[i];
}

/* ---------------------------------------------------------------------- */

void PairMAGMEAM::read_user_meam_file(const std::string &userfile)
{
  // done if user param file is "NULL"

  if (userfile == "NULL") return;

  // open user param file on proc 0

  std::shared_ptr<PotentialFileReader> reader;

  if (comm->me == 0) {
    reader = std::make_shared<PotentialFileReader>(lmp, userfile, "MAGMEAM");
  }

  // read settings
  // pass them one at a time to MEAM package
  // match strings to list of corresponding ints
  char * line = nullptr;
  char buffer[MAXLINE];

  while (true) {
    int which;
    int nindex, index[3];
    double value;
    int nline;
    if (comm->me == 0) {
      line = reader->next_line();
      if (line == nullptr) {
        nline = -1;
      } else nline = strlen(line) + 1;
    } else {
      line = buffer;
    }

    MPI_Bcast(&nline,1,MPI_INT,0,world);
    if (nline<0) break;
    MPI_Bcast(line,nline,MPI_CHAR,0,world);

    ValueTokenizer values(line, "=(), '\t\n\r\f");
    int nparams = values.count();
    std::string keyword = values.next_string();

    for (which = 0; which < nkeywords; which++)
      if (keyword == keywords[which]) break;
    if (which == nkeywords)
      error->all(FLERR,"Keyword {} in MEAM parameter file not recognized", keyword);

    nindex = nparams - 2;
    for (int i = 0; i < nindex; i++) index[i] = values.next_int() - 1;

    // map lattce_meam value to an integer
    if (which == 4) {
      std::string lattice_type = values.next_string();
      lattice_t latt;
      if (!MAGMEAM::str_to_lat(lattice_type, false, latt))
        error->all(FLERR, "Unrecognized lattice type in MEAM parameter file: {}", lattice_type);
      value = latt;
    } else if (which == 26) {
      std::string mag_lattice_type = values.next_string();
      lattice_t mag_latt;
      if (!MAGMEAM::str_to_lat(mag_lattice_type, false, mag_latt))
        error->all(FLERR, "Unrecognized lattice type in MEAM parameter file: {}", mag_lattice_type);
      value = mag_latt;
    }
    else value = values.next_double();

    // pass single setting to MEAM package

    int errorflag = 0;
    meam_inst->meam_setup_param(which,value,nindex,index,&errorflag);
    if (errorflag) {
      const char *descr[] = { "has an unknown error",
              "is out of range (please report a bug)",
              "expected more indices",
              "has out of range element index"};
      if ((errorflag < 0) || (errorflag > 3)) errorflag = 0;
      error->all(FLERR,"Error in MEAM parameter file: keyword {} {}", keyword, descr[errorflag]);
    }
  }
}

/* ---------------------------------------------------------------------- */

int PairMAGMEAM::pack_forward_comm(int n, int *list, double *buf,
                                int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = meam_inst->rho0[j];
    buf[m++] = meam_inst->rho1[j];
    buf[m++] = meam_inst->rho2[j];
    buf[m++] = meam_inst->rho3[j];
    buf[m++] = meam_inst->rho0m[j];
    buf[m++] = meam_inst->rho1m[j];
    buf[m++] = meam_inst->rho2m[j];
    buf[m++] = meam_inst->rho3m[j];
    buf[m++] = meam_inst->rho4m[j];
    buf[m++] = meam_inst->rho00[j];
    buf[m++] = meam_inst->frhop[j];
    buf[m++] = meam_inst->frhopp[j];
    buf[m++] = meam_inst->gamma[j];
    buf[m++] = meam_inst->dgamma1[j];
    buf[m++] = meam_inst->dgamma2[j];
    buf[m++] = meam_inst->dgamma3[j];
    buf[m++] = meam_inst->d2gamma2[j];
    buf[m++] = meam_inst->arho0m[j];
    buf[m++] = meam_inst->arho00[j];
    buf[m++] = meam_inst->arho00m[j];
    buf[m++] = meam_inst->arho2b[j];
    buf[m++] = meam_inst->arho2bm[j];
    buf[m++] = meam_inst->arho1[j][0];
    buf[m++] = meam_inst->arho1[j][1];
    buf[m++] = meam_inst->arho1[j][2];
    buf[m++] = meam_inst->arho1m[j][0];
    buf[m++] = meam_inst->arho1m[j][1];
    buf[m++] = meam_inst->arho1m[j][2];
    buf[m++] = meam_inst->arho2[j][0];
    buf[m++] = meam_inst->arho2[j][1];
    buf[m++] = meam_inst->arho2[j][2];
    buf[m++] = meam_inst->arho2[j][3];
    buf[m++] = meam_inst->arho2[j][4];
    buf[m++] = meam_inst->arho2[j][5];
    buf[m++] = meam_inst->arho2m[j][0];
    buf[m++] = meam_inst->arho2m[j][1];
    buf[m++] = meam_inst->arho2m[j][2];
    buf[m++] = meam_inst->arho2m[j][3];
    buf[m++] = meam_inst->arho2m[j][4];
    buf[m++] = meam_inst->arho2m[j][5];
    for (k = 0; k < 10; k++) buf[m++] = meam_inst->arho3[j][k];
    for (k = 0; k < 10; k++) buf[m++] = meam_inst->arho3m[j][k];
    buf[m++] = meam_inst->arho3b[j][0];
    buf[m++] = meam_inst->arho3b[j][1];
    buf[m++] = meam_inst->arho3b[j][2];
    buf[m++] = meam_inst->arho3bm[j][0];
    buf[m++] = meam_inst->arho3bm[j][1];
    buf[m++] = meam_inst->arho3bm[j][2];
    for (k = 0; k < 15; k++) buf[m++] = meam_inst->arho4m[j][k];
    buf[m++] = meam_inst->arho4bm[j][0];
    buf[m++] = meam_inst->arho4bm[j][1];
    buf[m++] = meam_inst->arho4bm[j][2];
    buf[m++] = meam_inst->arho4bm[j][3];
    buf[m++] = meam_inst->arho4bm[j][4];
    buf[m++] = meam_inst->arho4bm[j][5];
    buf[m++] = meam_inst->arho4cm[j];
    buf[m++] = meam_inst->t_ave[j][0];
    buf[m++] = meam_inst->t_ave[j][1];
    buf[m++] = meam_inst->t_ave[j][2];
    buf[m++] = meam_inst->t_ave[j][3];
    buf[m++] = meam_inst->t_ave[j][4];
    buf[m++] = meam_inst->t_ave[j][5];
    buf[m++] = meam_inst->t_ave[j][6];
    buf[m++] = meam_inst->t_ave[j][7];
    buf[m++] = meam_inst->t_ave[j][8];
    buf[m++] = meam_inst->tsq_ave[j][0];
    buf[m++] = meam_inst->tsq_ave[j][1];
    buf[m++] = meam_inst->tsq_ave[j][2];
    buf[m++] = meam_inst->tsq_ave[j][3];
    buf[m++] = meam_inst->tsq_ave[j][4];
    buf[m++] = meam_inst->tsq_ave[j][5];
    buf[m++] = meam_inst->tsq_ave[j][6];
    buf[m++] = meam_inst->tsq_ave[j][7];
    buf[m++] = meam_inst->tsq_ave[j][8];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void PairMAGMEAM::unpack_forward_comm(int n, int first, double *buf)
{
  int i,k,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    meam_inst->rho0[i] = buf[m++];
    meam_inst->rho1[i] = buf[m++];
    meam_inst->rho2[i] = buf[m++];
    meam_inst->rho3[i] = buf[m++];
    meam_inst->rho0m[i] = buf[m++];
    meam_inst->rho1m[i] = buf[m++];
    meam_inst->rho2m[i] = buf[m++];
    meam_inst->rho3m[i] = buf[m++];
    meam_inst->rho4m[i] = buf[m++];
    meam_inst->rho00[i] = buf[m++];
    meam_inst->frhop[i] = buf[m++];
    meam_inst->frhopp[i] = buf[m++];
    meam_inst->gamma[i] = buf[m++];
    meam_inst->dgamma1[i] = buf[m++];
    meam_inst->dgamma2[i] = buf[m++];
    meam_inst->dgamma3[i] = buf[m++];
    meam_inst->d2gamma2[i] = buf[m++];
    meam_inst->arho0m[i] = buf[m++];
    meam_inst->arho00[i] = buf[m++];
    meam_inst->arho00m[i] = buf[m++];
    meam_inst->arho2b[i] = buf[m++];
    meam_inst->arho2bm[i] = buf[m++];
    meam_inst->arho1[i][0] = buf[m++];
    meam_inst->arho1[i][1] = buf[m++];
    meam_inst->arho1[i][2] = buf[m++];
    meam_inst->arho1m[i][0] = buf[m++];
    meam_inst->arho1m[i][1] = buf[m++];
    meam_inst->arho1m[i][2] = buf[m++];
    meam_inst->arho2[i][0] = buf[m++];
    meam_inst->arho2[i][1] = buf[m++];
    meam_inst->arho2[i][2] = buf[m++];
    meam_inst->arho2[i][3] = buf[m++];
    meam_inst->arho2[i][4] = buf[m++];
    meam_inst->arho2[i][5] = buf[m++];
    meam_inst->arho2m[i][0] = buf[m++];
    meam_inst->arho2m[i][1] = buf[m++];
    meam_inst->arho2m[i][2] = buf[m++];
    meam_inst->arho2m[i][3] = buf[m++];
    meam_inst->arho2m[i][4] = buf[m++];
    meam_inst->arho2m[i][5] = buf[m++];
    for (k = 0; k < 10; k++) meam_inst->arho3[i][k] = buf[m++];
    for (k = 0; k < 10; k++) meam_inst->arho3m[i][k] = buf[m++];
    meam_inst->arho3b[i][0] = buf[m++];
    meam_inst->arho3b[i][1] = buf[m++];
    meam_inst->arho3b[i][2] = buf[m++];
    meam_inst->arho3bm[i][0] = buf[m++];
    meam_inst->arho3bm[i][1] = buf[m++];
    meam_inst->arho3bm[i][2] = buf[m++];
    for (k = 0; k < 15; k++) meam_inst->arho4m[i][k] = buf[m++];
    meam_inst->arho4bm[i][0] = buf[m++];
    meam_inst->arho4bm[i][1] = buf[m++];
    meam_inst->arho4bm[i][2] = buf[m++];
    meam_inst->arho4bm[i][3] = buf[m++];
    meam_inst->arho4bm[i][4] = buf[m++];
    meam_inst->arho4bm[i][5] = buf[m++];
    meam_inst->arho4cm[i] = buf[m++];
    meam_inst->t_ave[i][0] = buf[m++];
    meam_inst->t_ave[i][1] = buf[m++];
    meam_inst->t_ave[i][2] = buf[m++];
    meam_inst->t_ave[i][3] = buf[m++];
    meam_inst->t_ave[i][4] = buf[m++];
    meam_inst->t_ave[i][5] = buf[m++];
    meam_inst->t_ave[i][6] = buf[m++];
    meam_inst->t_ave[i][7] = buf[m++];
    meam_inst->t_ave[i][8] = buf[m++];
    meam_inst->tsq_ave[i][0] = buf[m++];
    meam_inst->tsq_ave[i][1] = buf[m++];
    meam_inst->tsq_ave[i][2] = buf[m++];
    meam_inst->tsq_ave[i][3] = buf[m++];
    meam_inst->tsq_ave[i][4] = buf[m++];
    meam_inst->tsq_ave[i][5] = buf[m++];
    meam_inst->tsq_ave[i][6] = buf[m++];
    meam_inst->tsq_ave[i][7] = buf[m++];
    meam_inst->tsq_ave[i][8] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int PairMAGMEAM::pack_reverse_comm(int n, int first, double *buf)
{
  int i,k,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = meam_inst->rho0[i];
    buf[m++] = meam_inst->arho0m[i];
    buf[m++] = meam_inst->arho00[i];
    buf[m++] = meam_inst->arho00m[i];
    buf[m++] = meam_inst->arho2b[i];
    buf[m++] = meam_inst->arho2bm[i];
    buf[m++] = meam_inst->arho1[i][0];
    buf[m++] = meam_inst->arho1[i][1];
    buf[m++] = meam_inst->arho1[i][2];
    buf[m++] = meam_inst->arho1m[i][0];
    buf[m++] = meam_inst->arho1m[i][1];
    buf[m++] = meam_inst->arho1m[i][2];
    buf[m++] = meam_inst->arho2[i][0];
    buf[m++] = meam_inst->arho2[i][1];
    buf[m++] = meam_inst->arho2[i][2];
    buf[m++] = meam_inst->arho2[i][3];
    buf[m++] = meam_inst->arho2[i][4];
    buf[m++] = meam_inst->arho2[i][5];
    buf[m++] = meam_inst->arho2m[i][0];
    buf[m++] = meam_inst->arho2m[i][1];
    buf[m++] = meam_inst->arho2m[i][2];
    buf[m++] = meam_inst->arho2m[i][3];
    buf[m++] = meam_inst->arho2m[i][4];
    buf[m++] = meam_inst->arho2m[i][5];
    for (k = 0; k < 10; k++) buf[m++] = meam_inst->arho3[i][k];
    for (k = 0; k < 10; k++) buf[m++] = meam_inst->arho3m[i][k];
    buf[m++] = meam_inst->arho3b[i][0];
    buf[m++] = meam_inst->arho3b[i][1];
    buf[m++] = meam_inst->arho3b[i][2];
    buf[m++] = meam_inst->arho3bm[i][0];
    buf[m++] = meam_inst->arho3bm[i][1];
    buf[m++] = meam_inst->arho3bm[i][2];
    for (k = 0; k < 15; k++) buf[m++] = meam_inst->arho4m[i][k];
    buf[m++] = meam_inst->arho4bm[i][0];
    buf[m++] = meam_inst->arho4bm[i][1];
    buf[m++] = meam_inst->arho4bm[i][2];
    buf[m++] = meam_inst->arho4bm[i][3];
    buf[m++] = meam_inst->arho4bm[i][4];
    buf[m++] = meam_inst->arho4bm[i][5];
    buf[m++] = meam_inst->arho4cm[i];
    buf[m++] = meam_inst->t_ave[i][0];
    buf[m++] = meam_inst->t_ave[i][1];
    buf[m++] = meam_inst->t_ave[i][2];
    buf[m++] = meam_inst->t_ave[i][3];
    buf[m++] = meam_inst->t_ave[i][4];
    buf[m++] = meam_inst->t_ave[i][5];
    buf[m++] = meam_inst->t_ave[i][6];
    buf[m++] = meam_inst->t_ave[i][7];
    buf[m++] = meam_inst->t_ave[i][8];
    buf[m++] = meam_inst->tsq_ave[i][0];
    buf[m++] = meam_inst->tsq_ave[i][1];
    buf[m++] = meam_inst->tsq_ave[i][2];
    buf[m++] = meam_inst->tsq_ave[i][3];
    buf[m++] = meam_inst->tsq_ave[i][4];
    buf[m++] = meam_inst->tsq_ave[i][5];
    buf[m++] = meam_inst->tsq_ave[i][6];
    buf[m++] = meam_inst->tsq_ave[i][7];
    buf[m++] = meam_inst->tsq_ave[i][8];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void PairMAGMEAM::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,k,m;

 m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    meam_inst->rho0[j] += buf[m++];
    meam_inst->arho0m[j] += buf[m++];
    meam_inst->arho00[j] += buf[m++];
    meam_inst->arho00m[j] += buf[m++];
    meam_inst->arho2b[j] += buf[m++];
    meam_inst->arho2bm[j] += buf[m++];
    meam_inst->arho1[j][0] += buf[m++];
    meam_inst->arho1[j][1] += buf[m++];
    meam_inst->arho1[j][2] += buf[m++];
    meam_inst->arho1m[j][0] += buf[m++];
    meam_inst->arho1m[j][1] += buf[m++];
    meam_inst->arho1m[j][2] += buf[m++];
    meam_inst->arho2[j][0] += buf[m++];
    meam_inst->arho2[j][1] += buf[m++];
    meam_inst->arho2[j][2] += buf[m++];
    meam_inst->arho2[j][3] += buf[m++];
    meam_inst->arho2[j][4] += buf[m++];
    meam_inst->arho2[j][5] += buf[m++];
    meam_inst->arho2m[j][0] += buf[m++];
    meam_inst->arho2m[j][1] += buf[m++];
    meam_inst->arho2m[j][2] += buf[m++];
    meam_inst->arho2m[j][3] += buf[m++];
    meam_inst->arho2m[j][4] += buf[m++];
    meam_inst->arho2m[j][5] += buf[m++];
    for (k = 0; k < 10; k++) meam_inst->arho3[j][k] += buf[m++];
    for (k = 0; k < 10; k++) meam_inst->arho3m[j][k] += buf[m++];
    meam_inst->arho3b[j][0] += buf[m++];
    meam_inst->arho3b[j][1] += buf[m++];
    meam_inst->arho3b[j][2] += buf[m++];
    meam_inst->arho3bm[j][0] += buf[m++];
    meam_inst->arho3bm[j][1] += buf[m++];
    meam_inst->arho3bm[j][2] += buf[m++];
    for (k = 0; k < 15; k++) meam_inst->arho4m[j][k] += buf[m++];
    meam_inst->arho4bm[j][0] += buf[m++];
    meam_inst->arho4bm[j][1] += buf[m++];
    meam_inst->arho4bm[j][2] += buf[m++];
    meam_inst->arho4bm[j][3] += buf[m++];
    meam_inst->arho4bm[j][4] += buf[m++];
    meam_inst->arho4bm[j][5] += buf[m++];
    meam_inst->arho4cm[j] += buf[m++];
    meam_inst->t_ave[j][0] += buf[m++];
    meam_inst->t_ave[j][1] += buf[m++];
    meam_inst->t_ave[j][2] += buf[m++];
    meam_inst->t_ave[j][3] += buf[m++];
    meam_inst->t_ave[j][4] += buf[m++];
    meam_inst->t_ave[j][5] += buf[m++];
    meam_inst->t_ave[j][6] += buf[m++];
    meam_inst->t_ave[j][7] += buf[m++];
    meam_inst->t_ave[j][8] += buf[m++];
    meam_inst->tsq_ave[j][0] += buf[m++];
    meam_inst->tsq_ave[j][1] += buf[m++];
    meam_inst->tsq_ave[j][2] += buf[m++];
    meam_inst->tsq_ave[j][3] += buf[m++];
    meam_inst->tsq_ave[j][4] += buf[m++];
    meam_inst->tsq_ave[j][5] += buf[m++];
    meam_inst->tsq_ave[j][6] += buf[m++];
    meam_inst->tsq_ave[j][7] += buf[m++];
    meam_inst->tsq_ave[j][8] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairMAGMEAM::memory_usage()
{
    
  double bytes = 17 * meam_inst->nmax * sizeof(double);
  bytes += (double)(3 + 4 + 6 + 6 + 10 + 10 + 15 + 15 + 3 + 3 + 8 + 8 + 2) * meam_inst->nmax * sizeof(double);
  bytes += (double)3 * meam_inst->maxneigh * sizeof(double);

  return bytes;
}

/* ----------------------------------------------------------------------
   strip special bond flags from neighbor list entries
   are not used with MEAM
   need to do here so Fortran lib doesn't see them
   done once per reneighbor so that neigh_f2c and neigh_c2f don't see them
------------------------------------------------------------------------- */

void PairMAGMEAM::neigh_strip(int inum, int *ilist,
                           int *numneigh, int **firstneigh)
{
  int i,j,ii,jnum;
  int *jlist;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (j = 0; j < jnum; j++) jlist[j] &= NEIGHMASK;
  }
}

/* ---------------------------------------------------------------------- */

void *PairMAGMEAM::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"scale") == 0) return (void *) scale;
  return nullptr;
}
