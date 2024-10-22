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
   Contributing author: Mark Sears (SNL)
------------------------------------------------------------------------- */

#include "replica_atom_swap_npt.h"

#include "atom.h"
#include "compute.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "finish.h"
#include "fix.h"
#include "force.h"
#include "integrate.h"
#include "modify.h"
#include "random_park.h"
#include "timer.h"
#include "universe.h"
#include "update.h"
#include "random_park.h"
#include "group.h"
#include "neighbor.h"
#include "pair.h"
#include "memory.h"
#include "improper.h"
#include "kspace.h"
#include "dihedral.h"
#include "bond.h"
#include "angle.h"
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

// #define TEMPER_DEBUG 1

/* ---------------------------------------------------------------------- */

ReplicaAtomSwapNPT::ReplicaAtomSwapNPT(LAMMPS *lmp) : Command(lmp) {
    type_list = new int[2];
}

/* ---------------------------------------------------------------------- */

ReplicaAtomSwapNPT::~ReplicaAtomSwapNPT()
{
  MPI_Comm_free(&roots);
  if (ranswap) delete ranswap;
  delete ranboltz;
  delete [] set_atom_types;
  delete [] set_temp;
  delete [] index2world;
  delete [] world2index;
  delete [] world2root;
  delete [] type_list;
}

/* ----------------------------------------------------------------------
   perform tempering with inter-world swaps
------------------------------------------------------------------------- */

void ReplicaAtomSwapNPT::command(int narg, char **arg)
{
  if (universe->nworlds == 1)
    error->all(FLERR,"Must have more than one processor partition to swap atoms between replicas");
  if (domain->box_exist == 0)
    error->all(FLERR,"Replica atom swap command before simulation box is defined");
  if (narg != 10)
    error->universe_all(FLERR,"Illegal replica atom swap command");

  int nsteps = utils::inumeric(FLERR,arg[0],false,lmp);
  nevery = utils::inumeric(FLERR,arg[1],false,lmp);
  temp = utils::numeric(FLERR,arg[2],false,lmp);
  press_set = utils::numeric(FLERR,arg[3],false,lmp);

  // ignore command, if walltime limit was already reached

  if (timer->is_timeout()) return;

  for (whichfix = 0; whichfix < modify->nfix; whichfix++)
    if (strcmp(arg[4],modify->fix[whichfix]->id) == 0) break;
  if (whichfix == modify->nfix)
    error->universe_all(FLERR,"Tempering fix ID is not defined");

  igroup = group->find(arg[5]);
  if (igroup == -1) error->all(FLERR,"Could not find fix group ID");
  groupbit = group->bitmask[igroup];
  seed_swap = utils::inumeric(FLERR,arg[6],false,lmp);
  seed_boltz = utils::inumeric(FLERR,arg[7],false,lmp);
  type_list[0] = utils::numeric(FLERR, arg[8], false, lmp);
  type_list[1] = utils::numeric(FLERR, arg[9], false, lmp);
  nswaptypes = 2;
  random_equal = new RanPark(lmp, seed_swap);
  my_set_index = universe->iworld;

  // swap frequency must evenly divide total # of timesteps

  if (nevery <= 0)
    error->universe_all(FLERR,"Invalid frequency in replica atom swap command");
  nswaps = nsteps/nevery;
  if (nswaps*nevery != nsteps)
    error->universe_all(FLERR,"Non integer # of swaps in replica atom swap command");

  // fix style must be appropriate for temperature control, i.e. it needs
  // to provide a working Fix::reset_target() and must not change the volume.

  if ( (!utils::strmatch(modify->fix[whichfix]->style,"^npt")) &&
      (!utils::strmatch(modify->fix[whichfix]->style,"^rigid/npt")) )
    error->universe_all(FLERR,"Tempering temperature and pressure fix is not supported");
  // setup for long tempering run

  update->whichflag = 1;
  timer->init_timeout();

  update->nsteps = nsteps;
  update->beginstep = update->firststep = update->ntimestep;
  update->endstep = update->laststep = update->firststep + nsteps;
  if (update->laststep < 0)
    error->all(FLERR,"Too many timesteps");

  lmp->init();

  atom_swap_nmax = 0;
  local_swap_atom_list = nullptr;
  local_swap_iatom_list = nullptr;
  local_swap_jatom_list = nullptr;

  // local storage

  me_universe = universe->me;
  MPI_Comm_rank(world,&me);
  nworlds = universe->nworlds;
  iworld = universe->iworld;
  boltz = force->boltz;
  nktv2p = force->nktv2p;
  // pe_compute = ptr to thermo_pe compute
  // notify compute it will be called at first swap

  int id = modify->find_compute("thermo_pe");
  if (id < 0) error->all(FLERR,"Replica atom swap could not find thermo_pe compute");
  c_pe = modify->compute[id];
  c_pe->addstep(update->ntimestep + nevery);

  // create MPI communicator for root proc from each world

  int color;
  if (me == 0) color = 0;
  else color = 1;
  MPI_Comm_split(universe->uworld,color,0,&roots);

  // RNGs for swaps and Boltzmann test
  // warm up Boltzmann RNG

  if (seed_swap) ranswap = new RanPark(lmp,seed_swap);
  else ranswap = nullptr;
  ranboltz = new RanPark(lmp,seed_boltz + me_universe);
  for (int i = 0; i < 100; i++) ranboltz->uniform();

  // world2root[i] = global proc that is root proc of world i

  world2root = new int[nworlds];
  if (me == 0)
    MPI_Allgather(&me_universe,1,MPI_INT,world2root,1,MPI_INT,roots);
  MPI_Bcast(world2root,nworlds,MPI_INT,0,world);

  set_temp = new double[nworlds];
  if (me == 0) MPI_Allgather(&temp,1,MPI_DOUBLE,set_temp,1,MPI_DOUBLE,roots);
  MPI_Bcast(set_temp,nworlds,MPI_DOUBLE,0,world);
  //printf("c0 %d %d %d\n",me_universe,nworlds,universe->iworld);
  // create static list of set types
  // allgather tempering arg "type" across root procs
  // bcast from each root to other procs in world

  set_atom_types = new int*[nworlds];
  if (me == 0) MPI_Allgather(&atom->type,1,MPI_INT,set_atom_types,1,MPI_INT,roots);
  MPI_Bcast(set_atom_types,nworlds,MPI_INT,0,world);

  // create world2index only on root procs from my_set_atom_types
  // create index2world on root procs from world2index,
  //   then bcast to all procs within world

  world2index = new int[nworlds];
  index2world = new int[nworlds];
  if (me == 0) {
    MPI_Allgather(&my_set_index,1,MPI_INT,world2index,1,MPI_INT,roots);
    for (int i = 0; i < nworlds; i++) index2world[world2index[i]] = i;
  }
  MPI_Bcast(index2world,nworlds,MPI_INT,0,world);
  //printf("c1 %d\n",me_universe);
  //reset temp target of Fix to current my_set_temp
    modify->fix[whichfix]->reset_target(temp);
    beta = 1.0 / (force->boltz * temp);

  memory->create(sqrt_mass_ratio, atom->ntypes + 1, atom->ntypes + 1, "atom/swap:sqrt_mass_ratio");
  for (int itype = 1; itype <= atom->ntypes; itype++)
    for (int jtype = 1; jtype <= atom->ntypes; jtype++)
      sqrt_mass_ratio[itype][jtype] = sqrt(atom->mass[itype] / atom->mass[jtype]);

  // check to see if itype and jtype cutoffs are the same
  // if not, reneighboring will be needed between swaps

  double **cutsq = force->pair->cutsq;
  unequal_cutoffs = false;
  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
    for (int jswaptype = 0; jswaptype < nswaptypes; jswaptype++)
      for (int ktype = 1; ktype <= atom->ntypes; ktype++)
        if (cutsq[type_list[iswaptype]][ktype] != cutsq[type_list[jswaptype]][ktype])
          unequal_cutoffs = true;

  // setup tempering runs

  int i,which,swap;
  double pe,pe_partner,boltz_factor,*new_atom_types;

  if (me_universe == 0 && universe->uscreen)
    fprintf(universe->uscreen,"Setting up tempering ...\n");
  //printf("c2 %d\n",me_universe);
  update->integrate->setup(1);
  //printf("c3 %d\n",me_universe);
  if (me_universe == 0) {
    if (universe->uscreen) {
      fprintf(universe->uscreen,"Step");
      for (int i = 0; i < nworlds; i++)
        fprintf(universe->uscreen," T%d",i);
      fprintf(universe->uscreen,"\n");
    }
    if (universe->ulogfile) {
      fprintf(universe->ulogfile,"Step");
      for (int i = 0; i < nworlds; i++)
        fprintf(universe->ulogfile," T%d",i);
      fprintf(universe->ulogfile,"\n");
    }
    print_status();
  }

  timer->init();
  timer->barrier_start();

  for (int iswap = 0; iswap < nswaps; iswap++) {

    // run for nevery timesteps
    //printf("l0 %d\n",me_universe);
    timer->init_timeout();
    update->integrate->run(nevery);
    //printf("l1 %d\n",me_universe);
    // check for timeout across all procs

    int my_timeout=0;
    int any_timeout=0;
    if (timer->is_timeout()) my_timeout=1;
    MPI_Allreduce(&my_timeout, &any_timeout, 1, MPI_INT, MPI_SUM, universe->uworld);
    if (any_timeout) {
      timer->force_timeout();
      break;
    }

    // compute PE
    // notify compute it will be called at next swap

    pe = c_pe->compute_scalar();
    c_pe->addstep(update->ntimestep + nevery);

    // which = which of 2 kinds of swaps to do (0,1)

    if (!ranswap) which = iswap % 2;
    else if (ranswap->uniform() < 0.5) which = 0;
    else which = 1;

    // If seed1 is 0, then the swap attempts will alternate between odd and 
    // even pairings. If seed1 is non-zero then it is used as a seed in a 
    // random number generator to randomly choose an odd or even pairing each time.

    if (which == 0) {
      if (my_set_index % 2 == 0) partner_set_index = my_set_index + 1;
      else partner_set_index = my_set_index - 1;
    } else {
      if (my_set_index % 2 == 1) partner_set_index = my_set_index + 1;
      else partner_set_index = my_set_index - 1;
    }
    if (partner_set_index >= nworlds){
      partner_set_index-=nworlds;
    }
    if (partner_set_index < 0){
      partner_set_index+=nworlds;
    }
    //printf("set ind: %d, partner ind: %d, me_un: %d\n",my_set_index,partner_set_index,me_universe);
    // partner = proc ID to swap with
    // if partner = -1, then I am not a proc that swaps
    partner = -1;
    partner_root = -1;
    if (me == 0 && partner_set_index >= 0 && partner_set_index < nworlds) {
      partner_world = index2world[partner_set_index];
      partner = world2root[partner_world];
      //partner_root = partner;
    }
    //MPI_Bcast(&partner_root,1,MPI_INT,universe->iworld,world);
    partner_root = partner_set_index;
    // swap with a partner, only root procs in each world participate
    // hi proc sends PE to low proc
    // lo proc make Boltzmann decision on whether to swap
    // lo proc communicates decision back to hi proc
     //printf("l2 %d\n",me_universe);

     //reneighbor to make sure atoms are in correct processor domains
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
    energy_stored = energy_full();
    //printf("l3 %d\n",me_universe);
    swap = 0;
    if (partner != -1) {
      if (me_universe > partner)
        MPI_Send(&energy_stored,1,MPI_DOUBLE,partner,4,universe->uworld);
      else
        MPI_Recv(&energy_stored_partner,1,MPI_DOUBLE,partner,4,universe->uworld,MPI_STATUS_IGNORE);
      if (me_universe > partner) {
        MPI_Send(&vol_stored,1, MPI_DOUBLE,partner,0,universe->uworld);
      }
      else {
        MPI_Recv(&vol_stored_partner,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);
      }
    }
    int nsuccess = 0;
    //printf("l4 %d\n",me_universe);
    update_swap_atoms_list();
   // printf("l5 %d\n",me_universe);
    for (int i = 0; i < 1; i++) nsuccess += attempt_swap();
    //printf("l6 %d\n",me_universe);
    if (me==0){
        if (me_universe==0){
          if (random_equal->uniform()>0.5){
            int temp = type_list[0];
            type_list[0]=type_list[1];
            type_list[1]=temp;
          }
          MPI_Send(type_list,2,MPI_INT,partner,14,universe->uworld);
        }
        else {
          MPI_Recv(type_list,2,MPI_INT,partner,14,universe->uworld,MPI_STATUS_IGNORE);
        }
        // if (me_universe<partner){
        //   MPI_Send(&type_list[1],1,MPI_INT,partner,5,universe->uworld);
        // }
        // else {
        //   MPI_Recv(&type_list[1],1,MPI_INT,partner,5,universe->uworld,MPI_STATUS_IGNORE);
        // }
      }
      MPI_Bcast(type_list,2,MPI_INT,0,world);
      // if (me_universe < partner)
      //   MPI_Send(&swap,1,MPI_INT,partner,0,universe->uworld);
      // else
      //   MPI_Recv(&swap,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);

#ifdef TEMPER_DEBUG
      if (me_universe < partner)
        printf("SWAP %d & %d: yes = %d,Ts = %d %d, PEs = %g %g, Bz = %g %g\n",
               me_universe,partner,swap,my_set_temp,partner_set_temp,
               pe,pe_partner,boltz_factor,exp(boltz_factor));
#endif

      
    
    // bcast swap result to other procs in my world

    // MPI_Bcast(&swap,1,MPI_INT,0,world);

    // update my_set_temp and temp2world on every proc
    // root procs update their value if swap took place
    // allgather across root procs
    // bcast within my world



    // print out current swap status
    
    if (me_universe == 0) print_status();
  }

  timer->barrier_stop();

  update->integrate->cleanup();

  Finish finish(lmp);
  finish.end(1);

  update->whichflag = 0;
  update->firststep = update->laststep = 0;
  update->beginstep = update->endstep = 0;
}



/* ----------------------------------------------------------------------
   proc 0 prints current tempering status
------------------------------------------------------------------------- */

void ReplicaAtomSwapNPT::print_status()
{
  std::string status = std::to_string(update->ntimestep);
  for (int i = 0; i < nworlds; i++)
    status += " " + std::to_string(world2index[i]);

  status += "\n";

  if (universe->uscreen) fputs(status.c_str(), universe->uscreen);
  if (universe->ulogfile) {
    fputs(status.c_str(), universe->ulogfile);
    fflush(universe->ulogfile);
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int ReplicaAtomSwapNPT::pick_i_swap_atom(int iwhichglobal)
{
  int i = -1;
  //int iwhichglobal = static_cast<int>(niswap * random_equal->uniform());
  //printf("iwhich: %d, i: %d, niswap_before: %d, niswap_local: %d\n",iwhichglobal,i,niswap_before,niswap_local);
  if ((iwhichglobal >= niswap_before) && (iwhichglobal < niswap_before + niswap_local)) {
    int iwhichlocal = iwhichglobal - niswap_before;
    i = local_swap_iatom_list[iwhichlocal];
    if (atom->type[i] == type_list[0])return i;
  }
  //printf("iwhich: %d, i: %d\n",iwhichglobal,i);
  return -1;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int ReplicaAtomSwapNPT::pick_j_swap_atom(int jwhichglobal)
{
  int j = -1;
  //int jwhichglobal = static_cast<int>(njswap * random_equal->uniform());
  //printf("jwhich: %d, j: %d, njswap_before: %d, njswap_local: %d\n",jwhichglobal,j,njswap_before,njswap_local);
  if ((jwhichglobal >= njswap_before) && (jwhichglobal < njswap_before + njswap_local)) {
    int jwhichlocal = jwhichglobal - njswap_before;
    j = local_swap_jatom_list[jwhichlocal];
    if (atom->type[j] == type_list[1]) return j;
  }
 // printf("jwhich: %d, j: %d\n",jwhichglobal,j);
  return -1;
}

double ReplicaAtomSwapNPT::energy_full()
{
  int eflag = 1;
  int vflag = 0;

  if (modify->n_pre_force) modify->pre_force(vflag);

  if (force->pair) force->pair->compute(eflag, vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (force->kspace) force->kspace->compute(eflag, vflag);

  if (modify->n_post_force_any) modify->post_force(vflag);

  update->eflag_global = update->ntimestep;
  double total_energy = c_pe->compute_scalar();

  return total_energy;
}


int ReplicaAtomSwapNPT::attempt_swap()
{
  int swap=0;
  //printf("a0 %d\n",me_universe);
  //
  //printf("partner_root: %d, me_universe: %d\n",partner_root,me_universe);
  int iglobal,jglobal;
  if (me==0){
    iglobal = static_cast<int>(niswap * random_equal->uniform());
    jglobal = static_cast<int>(njswap * random_equal->uniform());
  }
  //MPI_Barrier(world);
  MPI_Bcast(&iglobal,1,MPI_INT,0,world);
  MPI_Bcast(&jglobal,1,MPI_INT,0,world);
  
  
  //if (partner==-1)return 0;
  //if ((world2root[world]<partner && niswap == 0) || (world2root[world]>partner && njswap == 0)) return 0;

  // pre-swap energy

  double energy_before = energy_stored;
  // pick a random pair of atoms
  // swap their properties
  int i=-1;
  int j=-1;
  int itype = type_list[0];
  int jtype = type_list[1];
  //printf("a1 %d pr:%d\n",me_universe,partner_root);
  if (universe->iworld < partner_root){
    i = pick_i_swap_atom(iglobal);
   // printf("a2 %d itype: %d\n",me_universe,itype);
  }
  else {
    j = pick_j_swap_atom(jglobal);
  //  printf("a3 %d jtype: %d\n",me_universe,jtype);
  }
  int nproc = universe->procs_per_world[universe->iworld];
  int *ichecks = new int [nproc];
  int *jchecks = new int [nproc];
  MPI_Allgather(&i, 1, MPI_INT, ichecks,1, MPI_INT, world);
  MPI_Allgather(&j, 1, MPI_INT, jchecks,1, MPI_INT, world);
  int icheck=-1;
  int jcheck=-1;
  for (int l=0;l<nproc;l++){
    if (ichecks[l]>-1)icheck=1;
    if (jchecks[l]>-1)jcheck=1;
  }
  int check=0;
  if (universe->iworld < partner_root && icheck == 1){
    check = 1;
  }
  else if (universe->iworld > partner_root && jcheck == 1){
    check = 1;
  }
  int *checks = new int[nworlds];
  for (int l=0;l<nworlds;l++)checks[l]=0;
  if (me==0)MPI_Allgather(&check,1,MPI_INT,checks,1,MPI_INT,roots);
  MPI_Bcast(checks,nworlds,MPI_INT,0,world);
  for (int l=0;l<nworlds;l++){
    if (checks[l]==0)return 0;
  }

  

  MPI_Barrier(universe->uworld);
  if (i >= 0 && universe->iworld < partner_root) {
    atom->type[i] = jtype;
    if (atom->q_flag) atom->q[i] = qtype[1];
  }
  else if (j >= 0 && universe->iworld > partner_root) {
    atom->type[j] = itype;
    if (atom->q_flag) atom->q[j] = qtype[0];
  }
  MPI_Barrier(universe->uworld);
  // if unequal_cutoffs, call comm->borders() and rebuild neighbor list
  // else communicate ghost atoms
  // call to comm->exchange() is a no-op but clears ghost atoms

//  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);


  // post-swap energy
  //printf("a4 %d\n",me_universe);
  double energy_after = energy_full();
  //printf("a5 %d\n",me_universe);
  double energy_after_partner=energy_stored_partner;
  //MPI_Barrier(universe->uworld);

  if (me==0) {
    if (me_universe > partner)
      MPI_Send(&energy_after,1,MPI_DOUBLE,partner,12,universe->uworld);
    else
      MPI_Recv(&energy_after_partner,1,MPI_DOUBLE,partner,12,universe->uworld,MPI_STATUS_IGNORE);

    //printf("a6 %d\n",me_universe);
    // swap accepted, return 1
    // if ke_flag, rescale atom velocities

    if (me_universe < partner){
      double press_units = press_set/nktv2p;
      double delr = (energy_after_partner+energy_after - energy_stored - energy_stored_partner)
                    *(1.0/boltz/temp) + press_units*(1.0/boltz/temp)*(vol_stored_partner - vol_stored);
      double boltz_factor = -delr;
      if (boltz_factor >= 0.0) swap = 1;
      else if (ranboltz->uniform() < exp(boltz_factor)) swap = 1;
    }
    //printf("a7 %d\n",me_universe);
    //MPI_Barrier(universe->uworld);

    if (me_universe < partner)
      MPI_Send(&swap,1,MPI_INT,partner,13,universe->uworld);
    else
      MPI_Recv(&swap,1,MPI_INT,partner,13,universe->uworld,MPI_STATUS_IGNORE);
  }
  MPI_Barrier(world);
  //printf("a8 %d\n",me_universe);
  MPI_Bcast(&swap,1,MPI_INT,0,world);
  //printf("me_universe: %d,i: %d, j: %d, partner_root: %d, swap: %d\n",me_universe,i,j,partner_root,swap);
  MPI_Barrier(world);
  //printf("a9 %d %d %d\n",me_universe,swap,partner);
  if (swap==1){
    //printf("a10 %d %d %d %d %d\n",me_universe,itype,jtype,i,j);
    update_swap_atoms_list();
    //printf("a11 %d\n",me_universe);
    if (ke_flag) {
      if (i >= 0) {
        atom->v[i][0] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][1] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][2] *= sqrt_mass_ratio[itype][jtype];
      }
      if (j >= 0) {
        atom->v[j][0] *= sqrt_mass_ratio[jtype][itype];
        atom->v[j][1] *= sqrt_mass_ratio[jtype][itype];
        atom->v[j][2] *= sqrt_mass_ratio[jtype][itype];
      }
    }
    energy_stored = energy_after;
    energy_stored_partner = energy_after_partner;
    
    //if (swap) my_set_index = partner_set_index;
    //if (me == 0) {
    //  MPI_Allgather(&my_set_index,1,MPI_INT,world2index,1,MPI_INT,roots);
    //  for (i = 0; i < nworlds; i++) index2world[world2index[i]] = i;
    //}
    //MPI_Bcast(index2world,nworlds,MPI_INT,0,world);

    return 1;
  }
//printf("a12 %d\n",me_universe);
  // swap not accepted, return 0
  // restore the swapped itype & jtype atoms
  // do not need to re-call comm->borders() and rebuild neighbor list
  //   since will be done on next cycle or in Verlet when this fix finishes
  //MPI_Barrier(universe->uworld);
  if (i >= 0 && universe->iworld < partner_root) {
    atom->type[i] = itype;
    if (atom->q_flag) atom->q[i] = qtype[0];
  }
  else if (j >= 0 && universe->iworld > partner_root) {
    atom->type[j] = jtype;
    if (atom->q_flag) atom->q[j] = qtype[1];
  }
  //MPI_Barrier(universe->uworld);
  //MPI_Barrier(universe->uworld);
      if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);


  // post-swap energy
  //printf("a4 %d\n",me_universe);
  energy_after = energy_full();
  return 0;
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */

void ReplicaAtomSwapNPT::update_swap_atoms_list()
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double **x = atom->x;
  //printf("s0 %d\n",me_universe);
  if (atom->nmax > atom_swap_nmax) {
    memory->sfree(local_swap_iatom_list);
    memory->sfree(local_swap_jatom_list);
    atom_swap_nmax = atom->nmax;
    local_swap_iatom_list =
        (int *) memory->smalloc(atom_swap_nmax * sizeof(int), "MCSWAP:local_swap_iatom_list");
    local_swap_jatom_list =
        (int *) memory->smalloc(atom_swap_nmax * sizeof(int), "MCSWAP:local_swap_jatom_list");
  }
  
  niswap_local = 0;
  njswap_local = 0;
  MPI_Barrier(universe->uworld);
  
  //MPI_Bcast(&type_list[1],1,MPI_INT,0,world);
  MPI_Barrier(universe->uworld);
  for (int i = 0; i < nlocal; i++) {
    if (atom->mask[i] & groupbit) {
      if (type[i] == type_list[0]) {
        local_swap_iatom_list[niswap_local] = i;
        niswap_local++;
      } else if (type[i] == type_list[1]) {
        local_swap_jatom_list[njswap_local] = i;
        njswap_local++;
      }
    }
  }
  //printf("s1 %d\n",me_universe);
  //fprintf(universe->uscreen,"me: %d locali: %d i: %d j: %d jlocal: %d ibefore: %d jbefore: %d\n",me_universe,niswap_local,niswap,njswap,njswap_local,niswap_before,njswap_before);
  MPI_Allreduce(&niswap_local, &niswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&niswap_local, &niswap_before, 1, MPI_INT, MPI_SUM, world);
  niswap_before -= niswap_local;
  //printf("s2 %d\n",me_universe);
  MPI_Allreduce(&njswap_local, &njswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&njswap_local, &njswap_before, 1, MPI_INT, MPI_SUM, world);
  njswap_before -= njswap_local;


  //printf("s3 %d\n",me_universe);
  //determine whether next swap will move i to 2nd partition and j to 1st partition or vice versa

  //printf("s4: me_un: %d,me: %d locali: %d i: %d j: %d jlocal: %d ibefore: %d jbefore: %d,typei: %d, typej: %d\n",me_universe,me,niswap_local,niswap,njswap,njswap_local,niswap_before,njswap_before,type_list[0],type_list[1]);
}
