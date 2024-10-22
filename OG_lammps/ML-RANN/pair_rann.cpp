// clang-format off
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

#include "pair_rann.h"

#include "atom.h"
#include "citeme.h"
#include "error.h"
#include "math_special.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "tokenizer.h"
#include "update.h"
#include "force.h"

#include <cmath>
#include <cstring>

#include "rann_activation.h"
#include "rann_fingerprint.h"
#include "rann_stateequation.h"

#define MAXLINE 1024

using namespace LAMMPS_NS;

static const char cite_ml_rann_package[] =
  "ML-RANN package:\n\n"
  "@Article{Nitol2021,\n"
  " author = {Nitol, Mashroor S and Dickel, Doyl E and Barrett, Christopher D},\n"
  " title = {Artificial neural network potential for pure zinc},\n"
  " journal = {Computational Materials Science},\n"
  " year =    2021,\n"
  " volume =  188,\n"
  " pages =   {110207}\n"
  "}\n\n";


PairRANN::PairRANN(LAMMPS *lmp) : Pair(lmp)
{
  if (lmp->citeme) lmp->citeme->add(cite_ml_rann_package);

  //initialize ints and bools
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  allocated = 0;
  nelements = -1;
  nelementsp = -1;
  comm_forward = 0;
  comm_reverse = 0;
  res = 10000;
  cutmax = 0;
  dospin = false;
  memguess = 0;
  nmax1 = 0;
  nmax2 = 0;
  fmax = 0;
  fnmax = 0;
  //at least one of the following two flags will change during fingerprint definition:
  doscreen = false;
  allscreen = true;

  //null init for arrays with sizes not yet determined.
  elements = nullptr;
  mass = nullptr;
  elementsp = nullptr;
  map  = nullptr;
  fingerprintcount = nullptr;
  fingerprintlength = nullptr;
  fingerprintperelement = nullptr;
  stateequationcount = nullptr;
  stateequationperelement = nullptr;
  screening_min = nullptr;
  screening_max = nullptr;
  weightdefined = nullptr;
  biasdefined = nullptr;
  dimensiondefined = nullptr;
  bundle_inputdefined = nullptr;
  bundle_outputdefined = nullptr;
  xn = nullptr;
  yn = nullptr;
  zn = nullptr;
  Sik = nullptr;
  dSikx = nullptr;
  dSiky = nullptr;
  dSikz = nullptr;
  dSijkx = nullptr;
  dSijky = nullptr;
  dSijkz = nullptr;
  sx = nullptr;
  sy = nullptr;
  sz = nullptr;
  dSijkxc = nullptr;
  dSijkyc = nullptr;
  dSijkzc = nullptr;
  dfeaturesx = nullptr;
  dfeaturesy = nullptr;
  dfeaturesz = nullptr;
  features = nullptr;
  layer = nullptr;
  sum = nullptr;
  dsum1 = nullptr;
  dlayerx = nullptr;
  dlayery = nullptr;
  dlayerz = nullptr;
  dlayersumx = nullptr;
  dlayersumy = nullptr;
  dlayersumz = nullptr;
  dsx = nullptr;
  dsy = nullptr;
  dsz = nullptr;
  dssumx = nullptr;
  dssumy = nullptr;
  dssumz = nullptr;
  tn = nullptr;
  jl = nullptr;
  Bij = nullptr;
  sims = nullptr;
  net = nullptr;
  activation = nullptr;
  fingerprints = nullptr;
  state = nullptr;
  hbar = force->hplanck/6.283185307179586476925286766559;
}

PairRANN::~PairRANN()
{
  deallocate();
}

void PairRANN::deallocate()
{
  //clear memory
  delete[] mass;
  for (int i=0;i<nelements;i++) {delete [] elements[i];}
  delete[] elements;
  for (int i=0;i<nelementsp;i++) {delete [] elementsp[i];}
  delete[] elementsp;
  for (int i=0;i<=nelements;i++) {
    if (net[i].layers>0) {
      for (int j=0;j<net[i].layers-1;j++) {
        delete activation[i][j];
        delete [] net[i].bundleinputsize[j];
        delete [] net[i].bundleoutputsize[j];
        for (int k=0;k<net[i].bundles[j];k++){
          delete [] net[i].bundleinput[j][k];
          delete [] net[i].bundleoutput[j][k];
          delete [] net[i].bundleW[j][k];
          delete [] net[i].bundleB[j][k];
          delete [] net[i].freezeW[j][k];
          delete [] net[i].freezeB[j][k];
        }
        delete [] net[i].bundleinput[j];
        delete [] net[i].bundleoutput[j];
        delete [] net[i].bundleW[j];
        delete [] net[i].bundleB[j];
        delete [] net[i].freezeW[j];
        delete [] net[i].freezeB[j];
      }
      delete [] activation[i];
      delete [] net[i].dimensions;
      delete [] net[i].startI;
      delete [] net[i].bundleinput;
      delete [] net[i].bundleoutput;
      delete [] net[i].bundleinputsize;
      delete [] net[i].bundleoutputsize;
      delete [] net[i].bundleW;
      delete [] net[i].bundleB;
      delete [] net[i].freezeW;
      delete [] net[i].freezeB;
      delete [] net[i].bundles;
      delete [] dimensiondefined[i];
      delete [] weightdefined[i];
      delete [] biasdefined[i];
      delete [] bundle_inputdefined[i];
      delete [] bundle_outputdefined[i];
    }
  }
  delete[] net;
  delete[] map;
  for (int i=0;i<nelementsp;i++) {
    if (fingerprintlength[i]>0) {
      for (int j=0;j<fingerprintperelement[i];j++) {
        delete fingerprints[i][j];
      }
      delete[] fingerprints[i];
    }
    if (stateequationperelement[i]>0) {
      for (int j=0;j<stateequationperelement[i];j++) {
        delete state[i][j];
      }
      delete[] state[i];
    }
  }
  delete[] fingerprints;
  delete[] activation;
  delete[] state;
  delete[] weightdefined;
  delete[] biasdefined;
  delete[] dimensiondefined;
  delete[] bundle_inputdefined;
  delete[] bundle_outputdefined;
  delete[] fingerprintcount;
  delete[] fingerprintperelement;
  delete[] fingerprintlength;
  delete[] screening_min;
  delete[] screening_max;
  memory->destroy(xn);
  memory->destroy(yn);
  memory->destroy(zn);
  memory->destroy(tn);
  memory->destroy(jl);
  memory->destroy(features);
  memory->destroy(dfeaturesx);
  memory->destroy(dfeaturesy);
  memory->destroy(dfeaturesz);
  memory->destroy(layer);
  memory->destroy(sum);
  memory->destroy(dsum1);
  memory->destroy(dlayerx);
  memory->destroy(dlayery);
  memory->destroy(dlayerz);
  memory->destroy(dlayersumx);
  memory->destroy(dlayersumy);
  memory->destroy(dlayersumz);
  memory->destroy(Sik);
  memory->destroy(Bij);
  memory->destroy(dSikx);
  memory->destroy(dSiky);
  memory->destroy(dSikz);
  memory->destroy(dSijkx);
  memory->destroy(dSijky);
  memory->destroy(dSijkz);
  memory->destroy(dSijkxc);
  memory->destroy(dSijkyc);
  memory->destroy(dSijkzc);
  memory->destroy(sx);
  memory->destroy(sy);
  memory->destroy(sz);
  memory->destroy(dsx);
  memory->destroy(dsy);
  memory->destroy(dsz);
  memory->destroy(dssumx);
  memory->destroy(dssumy);
  memory->destroy(dssumz);
  memory->destroy(setflag);
  memory->destroy(cutsq);
}

void PairRANN::allocate(const std::vector<std::string> &elementwords)
{
  int i,n;
  n = atom->ntypes;
  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  cutmax = 0;
  nmax1 = 100;
  nmax2 = 20;
  fmax = 0;
  fnmax = 0;
  nelementsp=nelements+1;
  //initialize arrays
  elements = new char *[nelements];
  elementsp = new char *[nelementsp];//elements + 'all'
  mass = new double[nelements];
  net = new NNarchitecture[nelementsp];
  weightdefined = new bool**[nelementsp];
  biasdefined = new bool **[nelementsp];
  dimensiondefined = new bool*[nelements];
  bundle_inputdefined = new bool**[nelements];
  bundle_outputdefined = new bool**[nelements];
  activation = new RANN::Activation***[nelementsp];
  fingerprints = new RANN::Fingerprint**[nelementsp];
  state = new RANN::State**[nelementsp];
  fingerprintlength = new int[nelementsp];
  fingerprintperelement = new int[nelementsp];
  fingerprintcount = new int[nelementsp];
  stateequationperelement = new int [nelementsp];
  stateequationcount = new int [nelementsp];
  screening_min = new double[nelements*nelements*nelements];
  screening_max = new double[nelements*nelements*nelements];
  for (i=0;i<nelements;i++) {
    for (int j =0;j<nelements;j++) {
      for (int k=0;k<nelements;k++) {
        screening_min[i*nelements*nelements+j*nelements+k] = 0.8;//default values. Custom values may be read from potential file later.
        screening_max[i*nelements*nelements+j*nelements+k] = 2.8;//default values. Custom values may be read from potential file later.
      }
    }
  }
  for (i=0;i<=nelements;i++) {
    fingerprintlength[i]=0;
    fingerprintperelement[i] = -1;
    fingerprintcount[i] = 0;
    stateequationcount[i] = 0;
    stateequationperelement[i] = 0;
    if (i<nelements) {
      mass[i]=-1.0;
      elements[i]= utils::strdup(elementwords[i]);
    }
    elementsp[i] = utils::strdup(elementwords[i]);
    net[i].layers = 0;
    net[i].dimensions = new int[1];
    net[i].dimensions[0]=0;
  }
}

void PairRANN::settings(int narg, char ** /*arg*/)
{
  //read pair_style command in input file
  if (narg > 0) error->one(FLERR,"Illegal pair_style command");
}

void PairRANN::coeff(int narg, char **arg)
{
  int i,j;
  deallocate();//clear allocation from any previous coeff
  map = new int[atom->ntypes+1];
  if (narg != 3 + atom->ntypes) error->one(FLERR,"Incorrect args for pair coefficients");
  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0) error->one(FLERR,"Incorrect args for pair coefficients");
  nelements = -1;
  read_file(arg[2]);
  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++) {
      if (strcmp(arg[i],elements[j]) == 0) break;
    }
    if (j < nelements) map[i-2] = j;
    else error->one(FLERR,"No matching element in NN potential file");
  }
  // clear setflag since coeff() called once with I,J = * *
  int n = atom->ntypes;
  for (i = 1; i <= n; i++) {
    for (j = i; j <= n; j++) {
      setflag[i][j] = 0;
    }
  }
  // set setflag i,j for type pairs where both are mapped to elements
  // set mass of atom type if i = j
  int count = 0;
  for (i = 1; i <= n; i++) {
    for (j = i; j <= n; j++) {
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        if (i == j) atom->set_mass(FLERR,i,mass[map[i]]);
        count++;
      }
    }
  }
  if (count == 0) error->one(FLERR,"Incorrect args for pair coefficients");
  for (i=0;i<nelementsp;i++) {
    for (j=0;j<fingerprintperelement[i];j++) {
      fingerprints[i][j]->allocate();
    }
    for (j=0;j<stateequationperelement[i];j++) {
      state[i][j]->allocate();
    }
  }
  create_compute_arrays();
  allocated=1;
}

void PairRANN::create_compute_arrays(){
  memory->create(xn,nmax1,"pair:xn");
  memory->create(yn,nmax1,"pair:yn");
  memory->create(zn,nmax1,"pair:zn");
  memory->create(tn,nmax1,"pair:tn");
  memory->create(jl,nmax1,"pair:jl");
  memory->create(features,fmax,"pair:features");
  memory->create(dfeaturesx,fmax*nmax2,"pair:dfeaturesx");
  memory->create(dfeaturesy,fmax*nmax2,"pair:dfeaturesy");
  memory->create(dfeaturesz,fmax*nmax2,"pair:dfeaturesz");
  memory->create(layer,fnmax,"pair:layer");
  memory->create(sum,fnmax,"pair:sum");
  memory->create(dsum1,fnmax,"pair:dsum1");
  memory->create(dlayerx,nmax2,fnmax,"pair:dlayerx");
  memory->create(dlayery,nmax2,fnmax,"pair:dlayery");
  memory->create(dlayerz,nmax2,fnmax,"pair:dlayerz");
  memory->create(dlayersumx,nmax2,fnmax,"pair:dlayersumx");
  memory->create(dlayersumy,nmax2,fnmax,"pair:dlayersumy");
  memory->create(dlayersumz,nmax2,fnmax,"pair:dlayersumz");
  if (doscreen) {
    memory->create(Sik,nmax2,"pair:Sik");
    memory->create(Bij,nmax2,"pair:Bij");
    memory->create(dSikx,nmax2,"pair:dSikx");
    memory->create(dSiky,nmax2,"pair:dSiky");
    memory->create(dSikz,nmax2,"pair:dSikz");
    memory->create(dSijkx,nmax2*nmax2,"pair:dSijkx");
    memory->create(dSijky,nmax2*nmax2,"pair:dSijky");
    memory->create(dSijkz,nmax2*nmax2,"pair:dSijkz");
    memory->create(dSijkxc,nmax2,nmax2,"pair:dSijkxc");
    memory->create(dSijkyc,nmax2,nmax2,"pair:dSijkyc");
    memory->create(dSijkzc,nmax2,nmax2,"pair:dSijkzc");
  }
  if (dospin) {
    memory->create(sx,fmax*nmax2,"pair:sx");
    memory->create(sy,fmax*nmax2,"pair:sy");
    memory->create(sz,fmax*nmax2,"pair:sz");
    memory->create(dsx,nmax2,fnmax,"pair:dsx");
    memory->create(dsy,nmax2,fnmax,"pair:dsy");
    memory->create(dsz,nmax2,fnmax,"pair:dsz");
    memory->create(dssumx,nmax2,fnmax,"pair:dssumx");
    memory->create(dssumy,nmax2,fnmax,"pair:dssumy");
    memory->create(dssumz,nmax2,fnmax,"pair:dssumz");
  }
}

void PairRANN::compute(int eflag, int vflag)
{
  //perform force/energy computation_
  if (dospin) {
    if (strcmp(update->unit_style,"metal") != 0)
      error->one(FLERR,"Spin pair styles require metal units");
    if (!atom->sp_flag)
        error->one(FLERR,"Spin pair styles requires atom/spin style");
  }
  ev_init(eflag,vflag);

  // only global virial via fdotr is supported by this pair style

  if (vflag_atom)
    error->all(FLERR,"Pair style rann does not support computing per-atom stress");
  if (vflag && !vflag_fdotr)
    error->all(FLERR,"Pair style rann does not support 'pair_modify nofdotr'");

  int ii,i,j;
  int nn = 0;
  sims = new Simulation[1];
  sims->inum = listfull->inum;
  sims->gnum = listfull->gnum;
  sims->ilist=listfull->ilist;
  //sims->id = new int[sims->inum+sims->gnum];
  //for (ii=0;ii<sims->inum+sims->gnum;ii++){
  //  sims->id[ii]=ii;
  //}
  sims->id = listfull->ilist;
  sims->type = atom->type;
  sims->x = atom->x;
  sims->numneigh = listfull->numneigh;
  sims->firstneigh = listfull->firstneigh;
  if (dospin) {
    sims->s = atom->sp;
  }
  int itype,f,jnum,len;
  if (eflag_global) {eng_vdwl=0;eng_coul=0;}
  double energy;
  double **force = atom->f;
  double **fm = atom->fm;
  //loop over atoms
  for (ii=0;ii<sims->inum;ii++) {
      energy = 0;
      i = sims->ilist[ii];
      itype = map[sims->type[i]];
      f = net[itype].dimensions[0];
      jnum = sims->numneigh[i];
      if (jnum>nmax1) {
        nmax1 = jnum;
        memory->grow(xn,nmax1,"pair:xn");
        memory->grow(yn,nmax1,"pair:yn");
        memory->grow(zn,nmax1,"pair:zn");
        memory->grow(tn,nmax1,"pair:tn");
        memory->grow(jl,nmax1,"pair:jl");
      }
      cull_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,0,cutmax);
      if (jnum>nmax2) {
        nmax2=jnum;
        memory->grow(dfeaturesx,fmax*nmax2,"pair:dfeaturesx");
        memory->grow(dfeaturesy,fmax*nmax2,"pair:dfeaturesy");
        memory->grow(dfeaturesz,fmax*nmax2,"pair:dfeaturesz");
        memory->grow(layer,fnmax,"pair:layer");
        memory->grow(sum,fnmax,"pair:sum");
        memory->grow(dsum1,fnmax,"pair:dsum1");
        memory->grow(dlayerx,nmax2,fnmax,"pair:dlayerx");
        memory->grow(dlayery,nmax2,fnmax,"pair:dlayery");
        memory->grow(dlayerz,nmax2,fnmax,"pair:dlayerz");
        memory->grow(dlayersumx,nmax2,fnmax,"pair:dlayersumx");
        memory->grow(dlayersumy,nmax2,fnmax,"pair:dlayersumy");
        memory->grow(dlayersumz,nmax2,fnmax,"pair:dlayersumz");
        if (doscreen) {
          memory->grow(Sik,nmax2,"pair:Sik");
          memory->grow(Bij,nmax2,"pair:Bij");
          memory->grow(dSikx,nmax2,"pair:dSikx");
          memory->grow(dSiky,nmax2,"pair:dSiky");
          memory->grow(dSikz,nmax2,"pair:dSikz");
          memory->grow(dSijkx,nmax2*nmax2,"pair:dSijkx");
          memory->grow(dSijky,nmax2*nmax2,"pair:dSijky");
          memory->grow(dSijkz,nmax2*nmax2,"pair:dSijkz");
          memory->destroy(dSijkxc);
          memory->destroy(dSijkyc);
          memory->destroy(dSijkzc);
          memory->create(dSijkxc,nmax2,nmax2,"pair:dSijkxc");
          memory->create(dSijkyc,nmax2,nmax2,"pair:dSijkyc");
          memory->create(dSijkzc,nmax2,nmax2,"pair:dSijkzc");
        }
        if (dospin) {
          memory->grow(sx,fmax*nmax2,"pair:sx");
          memory->grow(sy,fmax*nmax2,"pair:sy");
          memory->grow(sz,fmax*nmax2,"pair:sz");
          memory->grow(dsx,nmax2,fnmax,"pair:dsx");
          memory->grow(dsy,nmax2,fnmax,"pair:dsy");
          memory->grow(dsz,nmax2,fnmax,"pair:dsz");
          memory->grow(dssumx,nmax2,fnmax,"pair:dssumx");
          memory->grow(dssumy,nmax2,fnmax,"pair:dssumy");
          memory->grow(dssumz,nmax2,fnmax,"pair:dssumz");
        }
      }
      for (j=0;j<f;j++) {
        features[j]=0;
      }
      for (j=0;j<f*jnum;j++) {
        dfeaturesx[j]=dfeaturesy[j]=dfeaturesz[j]=0;
      }
      //screening is calculated once for all atoms if any fingerprint uses it.
      if (dospin) {
        for (j=0;j<f*jnum;j++) {
          sx[j]=sy[j]=sz[j]=0;
        }
      }
      if (doscreen) screening(ii,0,jnum-1);
      if (allscreen) screen_neighbor_list(&jnum);
      //do fingerprints for atom type
      len = fingerprintperelement[itype];
      for (j=0;j<len;j++) {
        if      (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
        else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
        else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
        else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
      }
      itype = nelements;
      //do fingerprints for type "all"
      len = fingerprintperelement[itype];
      for (j=0;j<len;j++) {
        if      (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
        else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
        else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
        else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
      }
      //run fingerprints through network
      if (dospin) {
        propagateforwardspin(&energy,force,fm,ii,jnum);
      } else {
        propagateforward(&energy,force,ii,jnum);
      }
      itype = map[sims->type[i]];

      len = stateequationperelement[itype];
      for (j=0;j<len;j++){
        if      (state[itype][j]->screen==false && state[itype][j]->spin==false){state[itype][j]->eos_function(&energy,force,i,nn,xn,yn,zn,tn,jnum-1,jl);}
        else if (state[itype][j]->screen==true  && state[itype][j]->spin==false){state[itype][j]->eos_function(&energy,force,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
        else if (state[itype][j]->screen==false && state[itype][j]->spin==true ){state[itype][j]->eos_function(&energy,force,fm,i,nn,xn,yn,zn,tn,jnum-1,jl);}
        else if (state[itype][j]->screen==true  && state[itype][j]->spin==true ){state[itype][j]->eos_function(&energy,force,fm,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
      }
      itype = nelements;
      len = stateequationperelement[itype];
      for (j=0;j<len;j++){
        if      (state[itype][j]->screen==false && state[itype][j]->spin==false){state[itype][j]->eos_function(&energy,force,i,nn,xn,yn,zn,tn,jnum-1,jl);}
        else if (state[itype][j]->screen==true  && state[itype][j]->spin==false){state[itype][j]->eos_function(&energy,force,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
        else if (state[itype][j]->screen==false && state[itype][j]->spin==true ){state[itype][j]->eos_function(&energy,force,fm,i,nn,xn,yn,zn,tn,jnum-1,jl);}
        else if (state[itype][j]->screen==true  && state[itype][j]->spin==true ){state[itype][j]->eos_function(&energy,force,fm,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
      }
      if (dospin){
        for (int jj=0;jj<3;jj++){
          fm[ii][jj]/=hbar;
        }
      }
      if (eflag_atom) eatom[i]=energy;
      if (eflag_global) eng_vdwl +=energy;

  }
  if (vflag_fdotr) virial_fdotr_compute();
  //delete []sims->id;
  delete[] sims;
}

void PairRANN::compute_single_pair(int ii, double *fmi)
{
  int i,j;
  int nn = 0;
  sims = new Simulation[1];
  sims->inum = listfull->inum;
  sims->ilist=listfull->ilist;
  sims->id = listfull->ilist;
  sims->type = atom->type;
  sims->x = atom->x;
  sims->numneigh = listfull->numneigh;
  sims->firstneigh = listfull->firstneigh;
  if (dospin) {
    sims->s = atom->sp;
  }
  int itype,f,jnum,len;
  double energy;
  double **force = atom->f;
  double **fm = atom->fm;
  energy = 0;
  i = sims->ilist[ii];
  itype = map[sims->type[i]];
  f = net[itype].dimensions[0];
  jnum = sims->numneigh[i];
  if (jnum>nmax1) {
    nmax1 = jnum;
    memory->grow(xn,nmax1,"pair:xn");
    memory->grow(yn,nmax1,"pair:yn");
    memory->grow(zn,nmax1,"pair:zn");
    memory->grow(tn,nmax1,"pair:tn");
    memory->grow(jl,nmax1,"pair:jl");
  }
  cull_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,0,cutmax);
  if (jnum>nmax2) {
    nmax2=jnum;
    memory->grow(dfeaturesx,fmax*nmax2,"pair:dfeaturesx");
    memory->grow(dfeaturesy,fmax*nmax2,"pair:dfeaturesy");
    memory->grow(dfeaturesz,fmax*nmax2,"pair:dfeaturesz");
    memory->grow(layer,fnmax,"pair:layer");
    memory->grow(sum,fnmax,"pair:sum");
    memory->grow(dsum1,fnmax,"pair:dsum1");
    memory->grow(dlayerx,nmax2,fnmax,"pair:dlayerx");
    memory->grow(dlayery,nmax2,fnmax,"pair:dlayery");
    memory->grow(dlayerz,nmax2,fnmax,"pair:dlayerz");
    memory->grow(dlayersumx,nmax2,fnmax,"pair:dlayersumx");
    memory->grow(dlayersumy,nmax2,fnmax,"pair:dlayersumy");
    memory->grow(dlayersumz,nmax2,fnmax,"pair:dlayersumz");
    if (doscreen) {
      memory->grow(Sik,nmax2,"pair:Sik");
      memory->grow(Bij,nmax2,"pair:Bij");
      memory->grow(dSikx,nmax2,"pair:dSikx");
      memory->grow(dSiky,nmax2,"pair:dSiky");
      memory->grow(dSikz,nmax2,"pair:dSikz");
      memory->grow(dSijkx,nmax2*nmax2,"pair:dSijkx");
      memory->grow(dSijky,nmax2*nmax2,"pair:dSijky");
      memory->grow(dSijkz,nmax2*nmax2,"pair:dSijkz");
      memory->destroy(dSijkxc);
      memory->destroy(dSijkyc);
      memory->destroy(dSijkzc);
      memory->create(dSijkxc,nmax2,nmax2,"pair:dSijkxc");
      memory->create(dSijkyc,nmax2,nmax2,"pair:dSijkyc");
      memory->create(dSijkzc,nmax2,nmax2,"pair:dSijkzc");
    }
    if (dospin) {
      memory->grow(sx,fmax*nmax2,"pair:sx");
      memory->grow(sy,fmax*nmax2,"pair:sy");
      memory->grow(sz,fmax*nmax2,"pair:sz");
      memory->grow(dsx,nmax2,fnmax,"pair:dsx");
      memory->grow(dsy,nmax2,fnmax,"pair:dsy");
      memory->grow(dsz,nmax2,fnmax,"pair:dsz");
      memory->grow(dssumx,nmax2,fnmax,"pair:dssumx");
      memory->grow(dssumy,nmax2,fnmax,"pair:dssumy");
      memory->grow(dssumz,nmax2,fnmax,"pair:dssumz");
    }
  }
  for (j=0;j<f;j++) {
    features[j]=0;
  }
  for (j=0;j<f*jnum;j++) {
    dfeaturesx[j]=dfeaturesy[j]=dfeaturesz[j]=0;
  }
  //screening is calculated once for all atoms if any fingerprint uses it.
  if (dospin) {
    for (j=0;j<f*jnum;j++) {
      sx[j]=sy[j]=sz[j]=0;
    }
  }
  if (doscreen) screening(ii,0,jnum-1);
  if (allscreen) screen_neighbor_list(&jnum);
  //do fingerprints for atom type
  len = fingerprintperelement[itype];
  for (j=0;j<len;j++) {
    if      (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
    else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
    else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
    else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
  }
  itype = nelements;
  //do fingerprints for type "all"
  len = fingerprintperelement[itype];
  for (j=0;j<len;j++) {
    if      (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
    else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
    else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
    else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
  }
  //run fingerprints through network
  if (dospin) {
    propagateforwardspin(&energy,force,fm,ii,jnum);
  } else {
    propagateforward(&energy,force,ii,jnum);
  }
  itype = map[sims->type[i]];

  len = stateequationperelement[itype];
  for (j=0;j<len;j++){
    if      (state[itype][j]->screen==false && state[itype][j]->spin==false){state[itype][j]->eos_function(&energy,force,i,nn,xn,yn,zn,tn,jnum-1,jl);}
    else if (state[itype][j]->screen==true  && state[itype][j]->spin==false){state[itype][j]->eos_function(&energy,force,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
    else if (state[itype][j]->screen==false && state[itype][j]->spin==true ){state[itype][j]->eos_function(&energy,force,fm,i,nn,xn,yn,zn,tn,jnum-1,jl);}
    else if (state[itype][j]->screen==true  && state[itype][j]->spin==true ){state[itype][j]->eos_function(&energy,force,fm,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
  }
  itype = nelements;
  len = stateequationperelement[itype];
  for (j=0;j<len;j++){
    if      (state[itype][j]->screen==false && state[itype][j]->spin==false){state[itype][j]->eos_function(&energy,force,i,nn,xn,yn,zn,tn,jnum-1,jl);}
    else if (state[itype][j]->screen==true  && state[itype][j]->spin==false){state[itype][j]->eos_function(&energy,force,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
    else if (state[itype][j]->screen==false && state[itype][j]->spin==true ){state[itype][j]->eos_function(&energy,force,fm,i,nn,xn,yn,zn,tn,jnum-1,jl);}
    else if (state[itype][j]->screen==true  && state[itype][j]->spin==true ){state[itype][j]->eos_function(&energy,force,fm,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
  }
  if (dospin){
    for (int jj=0;jj<3;jj++){
      fm[ii][jj]/=hbar;
    }
  }
  delete[] sims;
}

void PairRANN::cull_neighbor_list(double *xn,double *yn, double *zn,int *tn, int* jnum,int *jl,int i,int sn,double cutmax){
	int *jlist,j,count,jj,*type,jtype;
	double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
	double **x = sims[sn].x;
	xtmp = x[i][0];
	ytmp = x[i][1];
	ztmp = x[i][2];
	type = sims[sn].type;
	jlist = sims[sn].firstneigh[i];
	count = 0;
	for (jj=0;jj<jnum[0];jj++){
		j = jlist[jj];
		j &= NEIGHMASK;
		jtype = map[type[j]];
		delx = xtmp - x[j][0];
		dely = ytmp - x[j][1];
		delz = ztmp - x[j][2];
		rsq = delx*delx + dely*dely + delz*delz;
		if (rsq>cutmax*cutmax){
			continue;
		}
		xn[count]=delx;
		yn[count]=dely;
		zn[count]=delz;
		tn[count]=jtype;
		//jl[count]=sims[sn].id[j];
		jl[count]=j;
		//jl is currently only used to calculate spin dot products.
		//j includes ghost atoms. id maps back to atoms in the box across periodic boundaries.
		//lammps code uses id instead of j because spin spirals are not supported.
		count++;
	}
	jnum[0]=count+1;
}

void PairRANN::screen_neighbor_list(int *jnum) {
  int jj,kk,count,count1;
  count = 0;
  for (jj=0;jj<jnum[0]-1;jj++) {
    if (Bij[jj]) {
      count1 = 0;
      if (jj!=count) {
      xn[count]=xn[jj];
      yn[count]=yn[jj];
      zn[count]=zn[jj];
      tn[count]=tn[jj];
      jl[count]=jl[jj];
      Sik[count]=Sik[jj];
      dSikx[count]=dSikx[jj];
      dSiky[count]=dSiky[jj];
      dSikz[count]=dSikz[jj];
      }
      for (kk=0;kk<jnum[0]-1;kk++) {
        if (Bij[kk]) {
          dSijkxc[count][count1] = dSijkx[jj*(jnum[0]-1)+kk];
          dSijkyc[count][count1] = dSijky[jj*(jnum[0]-1)+kk];
          dSijkzc[count][count1] = dSijkz[jj*(jnum[0]-1)+kk];
          count1++;
        }
      }
      count++;
    }
  }
  jnum[0]=count+1;
  for (jj=0;jj<count;jj++) {
    Bij[jj]=true;
    for (kk=0;kk<count;kk++) {
      dSijkx[jj*count+kk] = dSijkxc[jj][kk];
      dSijky[jj*count+kk] = dSijkyc[jj][kk];
      dSijkz[jj*count+kk] = dSijkzc[jj][kk];
    }
  }
}


void PairRANN::screening(int ii,int sid,int jnum)
{
 //see Baskes, Materials Chemistry and Physics 50 (1997) 152-1.58
  int i,*jlist,jj,j,kk,k,itype,jtype,ktype;
  double Sijk,Cijk,Cn,Cd,Dij,Dik,Djk,C,dfc,dC,**x;
  PairRANN::Simulation *sim = &sims[sid];
  double xtmp,ytmp,ztmp,delx,dely,delz,rij,delx2,dely2,delz2,rik,delx3,dely3,delz3,rjk;
  i = sim->ilist[ii];
  itype = map[sim->type[i]];
  for (int jj=0;jj<jnum;jj++){
    Sik[jj]=1;
    Bij[jj]=true;
    dSikx[jj]=0;
    dSiky[jj]=0;
    dSikz[jj]=0;
    }
    for (int jj=0;jj<jnum;jj++)
    for (kk=0;kk<jnum;kk++)
      dSijkx[jj*jnum+kk]=0;
    for (int jj=0;jj<jnum;jj++)
    for (kk=0;kk<jnum;kk++)
      dSijky[jj*jnum+kk]=0;
    for (int jj=0;jj<jnum;jj++)
    for (kk=0;kk<jnum;kk++)
      dSijkz[jj*jnum+kk]=0;
    for (kk=0;kk<jnum;kk++){//outer sum over k in accordance with source, some others reorder to outer sum over jj
    //if (Bij[kk]==false){continue;}
    ktype = tn[kk];
    delx2 = xn[kk];
    dely2 = yn[kk];
    delz2 = zn[kk];
    rik = delx2*delx2+dely2*dely2+delz2*delz2;
    if (rik>cutmax*cutmax){
      //Bij[kk]= false;
      continue;
    }
    for (jj=0;jj<jnum;jj++){
      if (jj==kk){continue;}
      //if (Bij[jj]==false){continue;}
      jtype = tn[jj];
      delx = xn[jj];
      dely = yn[jj];
      delz = zn[jj];
      rij = delx*delx+dely*dely+delz*delz;
      if (rij>cutmax*cutmax){
        //Bij[jj] = false;
        continue;
      }
      delx3 = delx2-delx;
      dely3 = dely2-dely;
      delz3 = delz2-delz;
      rjk = delx3*delx3+dely3*dely3+delz3*delz3;
      if (rik+rjk-rij<1e-13){continue;}//bond angle > 90 degrees
      if (rik+rij-rjk<1e-13){continue;}//bond angle > 90 degrees
      double Cmax = screening_max[itype*nelements*nelements+jtype*nelements+ktype];
      double Cmin = screening_min[itype*nelements*nelements+jtype*nelements+ktype];
      double temp1 = rij-rik+rjk;
      Cn = temp1*temp1-4*rij*rjk;
      temp1 = rij-rjk;
      Cd = temp1*temp1-rik*rik;
      Cijk = Cn/Cd;
      C = (Cijk-Cmin)/(Cmax-Cmin);
      if (C>=1){continue;}
      else if (C<=0){
        //Bij[kk]=false;
        Sik[kk]=0.0;
        dSikx[kk]=0.0;
        dSiky[kk]=0.0;
        dSikz[kk]=0.0;
        break;
      }
      dC = Cmax-Cmin;
      dC *= dC;
      dC *= dC;
      temp1 = 1-C;
      temp1 *= temp1;
      temp1 *= temp1;
      Sijk = 1-temp1;
      Sijk *= Sijk;
      Dij = 4*rik*(Cn+4*rjk*(rij+rik-rjk))/Cd/Cd;
      Dik = -4*(rij*Cn+rjk*Cn+8*rij*rik*rjk)/Cd/Cd;
      Djk = 4*rik*(Cn+4*rij*(rik-rij+rjk))/Cd/Cd;
      temp1 = Cijk-Cmax;
      double temp2 = temp1*temp1;
      dfc = 8*temp1*temp2/(temp2*temp2-dC);
      Sik[kk] *= Sijk;
      dSijkx[kk*jnum+jj] = dfc*(delx*Dij-delx3*Djk);
      dSikx[kk] += dfc*(delx2*Dik+delx3*Djk);
      dSijky[kk*jnum+jj] = dfc*(dely*Dij-dely3*Djk);
      dSiky[kk] += dfc*(dely2*Dik+dely3*Djk);
      dSijkz[kk*jnum+jj] = dfc*(delz*Dij-delz3*Djk);
      dSikz[kk] += dfc*(delz2*Dik+delz3*Djk);
    }
  }
}


//Called by getproperties. Propagate features and dfeatures through network. Updates force and energy
void PairRANN::propagateforward(double *energy,double **force,int ii,int jnum) {
  int i,j,k,jj,j1,itype,i1;
  int *ilist;
  ilist = listfull->ilist;
  int *type = atom->type;
  i1=ilist[ii];
  itype = map[type[i1]];
  NNarchitecture net1 = net[itype];
  int L = net1.layers-1;
  //energy output with forces from analytical derivatives
  int f = net1.dimensions[0];
  for (k=0;k<net1.dimensions[0];k++){
    layer[k]=features[k];
    for (jj=0;jj<jnum;jj++){
      dlayerx[jj][k]=dfeaturesx[jj*f+k];
      dlayery[jj][k]=dfeaturesy[jj*f+k];
      dlayerz[jj][k]=dfeaturesz[jj*f+k];
    }
  }
  for (i=0;i<net1.layers-1;i++) {
    for (j=0;j<net1.dimensions[i+1];j++){
      sum[j]=0;
    }
    for (i1=0;i1<net1.bundles[i];i1++){
      int s1=net1.bundleoutputsize[i][i1];
      int s2=net1.bundleinputsize[i][i1];
      for (j=0;j<s1;j++){
        int j1 = net1.bundleoutput[i][i1][j];
        for (k=0;k<s2;k++){
          int k1 = net1.bundleinput[i][i1][k];
          sum[j1] += net1.bundleW[i][i1][j*s2+k]*layer[k1];
        }
        sum[j1]+= net1.bundleB[i][i1][j];
      }
    }
    for (j=0;j<net1.dimensions[i+1];j++) {
      dsum1[j] = activation[itype][i][j]->dactivation_function(sum[j]);
      sum[j] = activation[itype][i][j]->activation_function(sum[j]);
      if (i==L-1) {
        energy[j] = sum[j];
      }
      //force propagation
      for (jj=0;jj<jnum;jj++) {
        dlayersumx[jj][j]=0;
        dlayersumy[jj][j]=0;
        dlayersumz[jj][j]=0;
      }
    }
    for (i1=0;i1<net1.bundles[i];i1++){
      int s1 = net1.bundleoutputsize[i][i1];
      int s2 = net1.bundleinputsize[i][i1];
      for (j=0;j<s1;j++){
        int j1 = net1.bundleoutput[i][i1][j];
        for (jj=0;jj<jnum;jj++){
          for (k=0;k<s2;k++){
            int k1= net1.bundleinput[i][i1][k];
            double w1 = net1.bundleW[i][i1][j*s2+k];
            dlayersumx[jj][j1] += w1*dlayerx[jj][k1];
            dlayersumy[jj][j1] += w1*dlayery[jj][k1];
            dlayersumz[jj][j1] += w1*dlayerz[jj][k1];
          }
        }
      }
    }
    for (j=0;j<net1.dimensions[i+1];j++){
      for (jj=0;jj<jnum;jj++){
        dlayersumx[jj][j]*= dsum1[j];
        dlayersumy[jj][j]*= dsum1[j];
        dlayersumz[jj][j]*= dsum1[j];
      }
    }
    if (i==L-1) {
      for (j=0;j<net1.dimensions[i+1];j++){
        for (jj=0;jj<jnum-1;jj++){
          int j2 = jl[jj];
          force[j2][0]+=dlayersumx[jj][j];
          force[j2][1]+=dlayersumy[jj][j];
          force[j2][2]+=dlayersumz[jj][j];
        }
        int j2 = sims->ilist[ii];
        jj = jnum-1;
        force[j2][0]+=dlayersumx[jj][j];
        force[j2][1]+=dlayersumy[jj][j];
        force[j2][2]+=dlayersumz[jj][j];
      }
    }
    //update values for next iteration
    for (j=0;j<net1.dimensions[i+1];j++) {
      layer[j]=sum[j];
      for (jj=0;jj<jnum;jj++) {
        dlayerx[jj][j] = dlayersumx[jj][j];
        dlayery[jj][j] = dlayersumy[jj][j];
        dlayerz[jj][j] = dlayersumz[jj][j];
      }
    }
  }
}

//Called by getproperties. Propagate features and dfeatures through network. Updates force and energy
void PairRANN::propagateforwardspin(double * energy,double **force,double **fm,int ii,int jnum) {
  int i,j,k,jj,j1,itype,i1;
  int *ilist;
  ilist = listfull->ilist;
  int *type = atom->type;
  i1=ilist[ii];
  itype = map[type[i1]];
  NNarchitecture net1 = net[itype];
  int L = net1.layers-1;
  //energy output with forces from analytical derivatives
  int f = net1.dimensions[0];
  for (k=0;k<net1.dimensions[0];k++) {
    layer[k]=features[k];
    for (jj=0;jj<jnum;jj++) {
      dlayerx[jj][k]=dfeaturesx[jj*f+k];
      dlayery[jj][k]=dfeaturesy[jj*f+k];
      dlayerz[jj][k]=dfeaturesz[jj*f+k];
      dsx[jj][k]=-sx[jj*f+k];
      dsy[jj][k]=-sy[jj*f+k];
      dsz[jj][k]=-sz[jj*f+k];
    }
  }
  for (i=0;i<net1.layers-1;i++) {
    for (j=0;j<net1.dimensions[i+1];j++) {
      sum[j]=0;
    }
    for (i1=0;i1<net1.bundles[i];i1++){
      int s1=net1.bundleoutputsize[i][i1];
      int s2=net1.bundleinputsize[i][i1];
      for (j=0;j<s1;j++){
        int j1 = net1.bundleoutput[i][i1][j];
        for (k=0;k<s2;k++){
          int k1 = net1.bundleinput[i][i1][k];
          sum[j1] += net1.bundleW[i][i1][j*s2+k]*layer[k1];
        }
        sum[j1]+= net1.bundleB[i][i1][j];
       }
    }
    for (j=0;j<net1.dimensions[i+1];j++) {
      dsum1[j] = activation[itype][i][j]->dactivation_function(sum[j]);
      sum[j] = activation[itype][i][j]->activation_function(sum[j]);
      if (i==L-1) {
        energy[j] = sum[j];
      }
      //force propagation
      for (jj=0;jj<jnum;jj++) {
        dlayersumx[jj][j]=0;
        dlayersumy[jj][j]=0;
        dlayersumz[jj][j]=0;
        dssumx[jj][j]=0;
        dssumy[jj][j]=0;
        dssumz[jj][j]=0;
      }
    }
    for (i1=0;i1<net1.bundles[i];i1++){
      int s1 = net1.bundleoutputsize[i][i1];
      int s2 = net1.bundleinputsize[i][i1];
      for (j=0;j<s1;j++){
        int j1 = net1.bundleoutput[i][i1][j];
        for (jj=0;jj<jnum;jj++){
          for (k=0;k<s2;k++){
            int k1= net1.bundleinput[i][i1][k];
            double w1 = net1.bundleW[i][i1][j*s2+k];
            dlayersumx[jj][j1] += w1*dlayerx[jj][k1];
            dlayersumy[jj][j1] += w1*dlayery[jj][k1];
            dlayersumz[jj][j1] += w1*dlayerz[jj][k1];
            dssumx[jj][j] += w1*dsx[jj][k];
            dssumy[jj][j] += w1*dsy[jj][k];
            dssumz[jj][j] += w1*dsz[jj][k];
          }
        }
      }
    }
    for (j=0;j<net1.dimensions[i+1];j++){
      for (jj=0;jj<jnum;jj++){
        dlayersumx[jj][j]*= dsum1[j];
        dlayersumy[jj][j]*= dsum1[j];
        dlayersumz[jj][j]*= dsum1[j];
        dssumx[jj][j] *= dsum1[j];
        dssumy[jj][j] *= dsum1[j];
        dssumz[jj][j] *= dsum1[j];
      }
    }
    if (i==L-1) {
      for (j=0;j<net1.dimensions[i+1];j++){
        for (jj=0;jj<jnum-1;jj++){
          int j2 = jl[jj];
          force[j2][0]+=dlayersumx[jj][j];
          force[j2][1]+=dlayersumy[jj][j];
          force[j2][2]+=dlayersumz[jj][j];
          fm[j1][0]+=dssumx[jj][j];
          fm[j1][1]+=dssumy[jj][j];
          fm[j1][2]+=dssumz[jj][j];
        }
        int j2 = sims->ilist[ii];
        jj = jnum-1;
        force[j2][0]+=dlayersumx[jj][j];
        force[j2][1]+=dlayersumy[jj][j];
        force[j2][2]+=dlayersumz[jj][j];
        fm[j1][0]+=dssumx[jj][j];
        fm[j1][1]+=dssumy[jj][j];
        fm[j1][2]+=dssumz[jj][j];
      }
    }
    //update values for next iteration
    for (j=0;j<net1.dimensions[i+1];j++) {
      layer[j]=sum[j];
      for (jj=0;jj<jnum;jj++) {
        dlayerx[jj][j] = dlayersumx[jj][j];
        dlayery[jj][j] = dlayersumy[jj][j];
        dlayerz[jj][j] = dlayersumz[jj][j];
        dsx[jj][j] = dssumx[jj][j];
        dsy[jj][j] = dssumy[jj][j];
        dsz[jj][j] = dssumz[jj][j];
      }
    }
  }
}

void PairRANN::create_random_weights(int rows,int columns,int itype,int layer,int bundle){
  net[itype].bundleW[layer][bundle] = new double [rows*columns];
  net[itype].freezeW[layer][bundle] = new bool [rows*columns];
  double r;
  for (int i=0;i<rows;i++){
    for (int j=0;j<columns;j++){
      r = (double)rand()/RAND_MAX*2-1;//flat distribution from -1 to 1
      net[itype].bundleW[layer][bundle][i*columns+j] = r;
      net[itype].freezeW[layer][bundle][i*columns+j] = 0;
    }
  }
  weightdefined[itype][layer][bundle]=true;
}

void PairRANN::create_random_biases(int rows,int itype, int layer,int bundle){
  net[itype].bundleB[layer][bundle] = new double [rows];
  net[itype].freezeB[layer][bundle] = new bool [rows];
  double r;
  for (int i=0;i<rows;i++){
    r = (double) rand()/RAND_MAX*2-1;
    net[itype].bundleB[layer][bundle][i] = r;
    net[itype].freezeB[layer][bundle][i] = 0;
  }
  biasdefined[itype][layer][bundle]=true;
}

void PairRANN::read_parameters(std::vector<std::string>, std::vector<std::string>,FILE *, char *, int *, char *){

}

void PairRANN::init_list(int /*which*/, NeighList *ptr)
{
  listfull = ptr;
}

void PairRANN::init_style()
{
  neighbor->add_request(this, NeighConst::REQ_FULL);
}


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairRANN::init_one(int /*i*/, int /*j*/)
{
  return cutmax;
}

void PairRANN::errorf(const char *file, int line, const char * message) {
  error->one(file,line,message);
}

int PairRANN::factorial(int n) {
  return round(MathSpecial::factorial(n));
}

std::vector<std::string> PairRANN::tokenmaker(std::string line,std::string delimiter) {
  return Tokenizer(line,delimiter).as_vector();
}
