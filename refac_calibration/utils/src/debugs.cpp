#include "pair_spin_rann.h"
#include "rann_activation.h"
#include "rann_fingerprint.h"
#include "rann_stateequation.h"

using namespace LAMMPS_NS;

// energy and error per simulation
void PairRANN::write_debug_level1(double* fit_err, double* val_err)
{
    int  check = mkdir("DEBUG", 0777);
    char debug_summary[strlen(potential_output_file) + 14];
    sprintf(debug_summary, "DEBUG/%s.debug1", potential_output_file);
    FILE* fid_summary = fopen(debug_summary, "w");
    fprintf(fid_summary, "#Error from simulations included in training:\n");
    fprintf(fid_summary, "#training_size, validation_size:\n");
    fprintf(fid_summary, "%d %d\n", nsimr, nsimv);
    fprintf(fid_summary, "#fit_index, filename, timestep, natoms, target/atom, state/atom, (t-s)/a, error/atom\n");
    double targetmin = 10 ^ 300;
    double errmax    = 0;
    double err    = 0;
    double target = 0;
    
    for (int i = 0; i < nsimr; i++) {
        err    = fit_err[i] / sims[r(i)].energy_weight / sims[r(i)].inum;
        target = sims[r(i)].energy / sims[r(i)].inum;
        double state = sims[r(i)].state_e / sims[r(i)].inum;
        if (target < targetmin) {
            targetmin = target;
        }
        if (fabs(err) > errmax) {
            errmax = fabs(err);
        }
        fprintf(fid_summary, "%d %s %d %d %f %f %f %f\n", r(i), sims[r(i)].filename, sims[r(i)].timestep, sims[r(i)].inum, target + state, state, target, err);
    }
    fprintf(fid_summary, "#Error from simulations included in validation:\n");
    fprintf(fid_summary, "#val_index, filename, timestep, natoms, target/atom, state/atom, (t-s)/a, error/atom\n");
    for (int i = 0; i < nsimv; i++) {
        err    = val_err[i] / sims[v(i)].energy_weight / sims[v(i)].inum;
        target = sims[v(i)].energy / sims[v(i)].inum;
        double state = sims[v(i)].state_e / sims[v(i)].inum;
        if (target < targetmin) {
            targetmin = target;
        }
        if (fabs(err) > errmax) {
            errmax = fabs(err);
        }
        fprintf(fid_summary, "%d %s %d %d %f %f %f %f\n", v(i), sims[v(i)].filename, sims[v(i)].timestep, sims[v(i)].inum, target + state, state, target, err);
    }
    fprintf(fid_summary, "#mininum target/atom: %f, max error/atom: %f\n", targetmin, errmax);
    fclose(fid_summary);
}

// dump files with energy of each atom
void PairRANN::write_debug_level2(double* fit_err, double* val_err)
{
    FILE*    dumps[nsets];
    FILE*    current;
    double** energies;
    int index[nsims];
    for (int i = 0; i < nsims; i++) {
        index[i] = i;
    }
    energies = new double*[nsims];
    get_per_atom_energy(energies, index, nsims, net);
    char* debugnames[nsets];
    int   check = mkdir("DEBUG", 0777);
    for (int i = 0; i < nsets; i++) {
        if (Xset[i] == 0) {
            continue;
        }
        debugnames[i] = new char [strlen(dumpfilenames[i]) + 20];
        sprintf(debugnames[i], "DEBUG/%s.debug2", dumpfilenames[i]);
        dumps[i] = fopen(debugnames[i], "w");
    }
    for (int i = 0; i < nsims; i++) {
        bool foundcurrent = false;
        int  nsims, j;
        for (j = 0; j < nsets; j++) {
            if (Xset[j] == 0) {
                continue;
            }
            if (strcmp(dumpfilenames[j], sims[i].filename) == 0) {
                current = dumps[j];
                foundcurrent = true;
                nsims = Xset[j];
                break;
            }
        }
        double energy_weight = sims[i].energy_weight;
        if (targettype == 1) {
            energy_weight *= sqrt(sims[i].inum);
        }
        if (!foundcurrent) {
            errorf("something happened!\n");
        }
        if (sims[i].spinspirals) {
            fprintf(current, "ITEM: TIMESTEP energy, energy_weight, force_weight, nsims, qx, qy, qz, ax, ay, az\n");
            fprintf(current, "%d %f %f %f %d %f %f %f %f %f %f\n", sims[i].timestep, sims[i].energy + sims[i].state_e, energy_weight, sims[i].force_weight, nsims, sims[i].spinvec[0],
                sims[i].spinvec[1], sims[i].spinvec[2], sims[i].spinaxis[0], sims[i].spinaxis[1], sims[i].spinaxis[2]);
        }
        else {
            fprintf(current, "ITEM: TIMESTEP energy, energy_weight, force_weight, nsims\n");
            fprintf(current, "%d %f %f %f %d\n", sims[i].timestep, sims[i].energy + sims[i].state_e, sims[i].energy_weight, sims[i].force_weight, nsims);
        }
        fprintf(current, "ITEM: NUMBER OF ATOMS\n");
        fprintf(current, "%d\n", sims[i].inum);
        fprintf(current, "ITEM: BOX BOUNDS xy xz yz pp pp pp\n");
        double xmin = sims[i].origin[0];
        double xmax = sims[i].box[0][0] + sims[i].origin[0];
        double ymin = sims[i].origin[1];
        double ymax = sims[i].box[1][1] + sims[i].origin[1];
        double zmin = sims[i].origin[2];
        double zmax = sims[i].box[2][2] + sims[i].origin[2];
        if (sims[i].box[0][1] > 0) {
            xmax += sims[i].box[0][1];
        }
        else {
            xmin += sims[i].box[0][1];
        }
        if (sims[i].box[0][2] > 0) {
            xmax += sims[i].box[0][2];
        }
        else {
            xmin += sims[i].box[0][2];
        }
        if (sims[i].box[1][2] > 0) {
            ymax += sims[i].box[1][2];
        }
        else {
            ymin += sims[i].box[1][2];
        }
        fprintf(current, "%f %f %f\n", xmin, xmax, sims[i].box[0][1]);
        fprintf(current, "%f %f %f\n", ymin, ymax, sims[i].box[0][2]);
        fprintf(current, "%f %f %f\n", zmin, zmax, sims[i].box[1][2]);
        fprintf(current, "ITEM: ATOMS id type x y z c_eng c_state c_nn\n");
        for (int j = 0; j < sims[i].inum; j++) {
            fprintf(current, "%d %d %f %f %f %f %f %f\n", sims[i].ilist[j] + 1, sims[i].type[j] + 1, sims[i].x[j][0], sims[i].x[j][1], sims[i].x[j][2], energies[i][j] + sims[i].state_ea[j],
                sims[i].state_ea[j], energies[i][j]);
        }
    }
    for (int i = 0; i < nsets; i++) {
        if (Xset[i] == 0) {
            continue;
        }
        fclose(dumps[i]);
    }
}

// calibration data: current Jacobian, target vector, solve vector, and step.
void PairRANN::write_debug_level3(double* jacob, double* target, double* beta, double* delta)
{
    int   check = mkdir("DEBUG", 0777);
    FILE* fid   = fopen("DEBUG/jacobian.debug3.csv", "w");
    for (int i = 0; i < jlen1; i++) {
        for (int j = 0; j < betalen; j++) {
            fprintf(fid, "%f, ", jacob[i * betalen + j]);
        }
        fprintf(fid, "\n");
    }
    fclose(fid);
    fid = fopen("DEBUG/target.debug3.csv", "w");
    for (int i = 0; i < jlen1; i++) {
        fprintf(fid, "%f\n", target[i]);
    }
    fclose(fid);
    fid = fopen("DEBUG/beta.debug3.csv", "w");
    for (int j = 0; j < betalen; j++) {
        fprintf(fid, "%f\n", beta[j]);
    }
    fclose(fid);
    fid = fopen("DEBUG/delta.debug3.csv", "w");
    for (int j = 0; j < betalen; j++) {
        fprintf(fid, "%f\n", delta[j]);
    }
    fclose(fid);
}

// dump files with energy and forces of each atom
// SLOW - recomputes fingerprints because memory storage is unreasonable to keep everything from first time.
void PairRANN::write_debug_level4(double* fit_err, double* val_err)
{
    printf("starting debug level 4\n");
    // double est=0;
    // for (int nn=0;nn<nsims;nn++){
    //	est+=sims[nn].time;
    // }
    // printf("estimated time: %f seconds\n",est/omp_get_num_threads());
    FILE* dumps[nsims];
    FILE* current;
    char* debugnames[nsims];
    int   check = mkdir("DEBUG", 0777);
    for (int i = 0; i < nsims; i++) {
        debugnames[i] = new char [strlen(sims[i].filename) + 20];
        sprintf(debugnames[i], "DEBUG/%s.debug4.%d", sims[i].filename, sims[i].timestep);
    }
// double **energies;
// int index[nsims];
// for (int i = 0;i<nsims;i++){
//	 index[i]=i;
// }
// energies = new double*[nsims];
// get_per_atom_energy(energies,index,nsims,net);
#pragma omp parallel
    {
        int i, ii, itype, f, jnum, len, j, nn;
#pragma omp for schedule(guided)
        for (nn = 0; nn < nsims; nn++) {
            dumps[nn] = fopen(debugnames[nn], "w");
            current   = dumps[nn];
            int     j;
            double* force[sims[nn].inum + sims[nn].gnum];
            double  force1[(sims[nn].inum + sims[nn].gnum) * 3];
            double* fm[sims[nn].inum + sims[nn].gnum];
            double  fm1[(sims[nn].inum + sims[nn].gnum) * 3];
            for (ii = 0; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                force[ii] = &force1[ii * 3];
                fm[ii]    = &fm1[ii * 3];
            }
            double energy_nn[sims[nn].inum];
            double energy_state[sims[nn].inum];
            for (ii = 0; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                for (j = 0; j < 3; j++) {
                    force[ii][j] = 0;
                    fm[ii][j]    = 0;
                }
            }
            for (ii = 0; ii < sims[nn].inum; ii++) {
                i     = sims[nn].ilist[ii];
                itype = map(sims[nn].type[i]);
                f     = net(itype).dimensions[0];
                jnum  = sims[nn].numneigh[i];
                double xn[jnum];
                double yn[jnum];
                double zn[jnum];
                int    tn[jnum];
                int    jl[jnum];
                cull_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, cutmax);
                double features[f];
                double dfeaturesx[f * jnum];
                double dfeaturesy[f * jnum];
                double dfeaturesz[f * jnum];
                for (j = 0; j < f; j++) {
                    features[j] = 0;
                }
                for (j = 0; j < f * jnum; j++) {
                    dfeaturesx[j] = dfeaturesy[j] = dfeaturesz[j] = 0;
                }
                // screening is calculated once for all atoms if any fingerprint uses it.
                double Sik[jnum];
                double dSikx[jnum];
                double dSiky[jnum];
                double dSikz[jnum];
                // TO D0: add check against stack size
                double dSijkx[jnum * jnum];
                double dSijky[jnum * jnum];
                double dSijkz[jnum * jnum];
                bool   Bij[jnum];
                double sx[jnum * f];
                double sy[jnum * f];
                double sz[jnum * f];
                double sxx[jnum * f];
                double sxy[jnum * f];
                double sxz[jnum * f];
                double syy[jnum * f];
                double syz[jnum * f];
                double szz[jnum * f];
                for (j = 0; j < f * jnum; j++) {
                    sx[j]  = sy[j] = sz[j] = 0;
                    sxx[j] = sxy[j] = sxz[j] = syy[j] = syz[j] = szz[j] = 0;
                }
                if (doscreen) {
                    screen(Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1);
                }
                if (allscreen) {
                    screen_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, Bij, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz);
                }
                // do fingerprints for atom type
                len = fingerprintperelement[itype];
                for (j = 0; j < len; j++) {
                    if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn,
                            jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky,
                            dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                }
                itype = nelements;
                // do fingerprints for type "all"
                len = fingerprintperelement[itype];
                for (j = 0; j < len; j++) {
                    if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn,
                            jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky,
                            dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                }
                double e  = 0.0;
                double e1 = 0.0;
                itype = map(sims[nn].type[i]);
                len   = stateequationperelement[itype];
                for (j = 0; j < len; j++) {
                    if (state[itype][j]->screen == false && state[itype][j]->spin == false) {
                        state[itype][j]->eos_function(&e, force, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == true && state[itype][j]->spin == false) {
                        state[itype][j]->eos_function(&e, force, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == false && state[itype][j]->spin == true) {
                        state[itype][j]->eos_function(&e, force, fm, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == true && state[itype][j]->spin == true) {
                        state[itype][j]->eos_function(&e, force, fm, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                }
                itype = nelements;
                len   = stateequationperelement[itype];
                for (j = 0; j < len; j++) {
                    if (state[itype][j]->screen == false && state[itype][j]->spin == false) {
                        state[itype][j]->eos_function(&e, force, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == true && state[itype][j]->spin == false) {
                        state[itype][j]->eos_function(&e, force, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == false && state[itype][j]->spin == true) {
                        state[itype][j]->eos_function(&e, force, fm, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == true && state[itype][j]->spin == true) {
                        state[itype][j]->eos_function(&e, force, fm, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                }
                itype = map(sims[nn].type[i]);
                // NNarchitecture* net_out = new NNarchitecture[nelementsp];
                auto net_out = CArray<NNarchitecture>(nelementsp);
                
                if (normalizeinput) {
                    unnormalize_net(net_out);
                }
                else {
                    copy_network(net, net_out);
                }
                // for (int k=0;k<f;k++){
                //	fprintf(current,"%.10e ",features[k]-sims[nn].features[ii][k]);
                // }
                // fprintf(current,"\n");
                propagateforward(&e1, force, ii, jnum, itype, features, dfeaturesx, dfeaturesy, dfeaturesz, jl, nn, net_out);
                energy_nn[ii]    = e1;
                energy_state[ii] = e;
            }
            // apply ghost neighbor forces back into box
            for (ii = sims[nn].inum; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                for (j = 0; j < 3; j++) {
                    force[sims[nn].id[ii]][j] += force[ii][j];
                }
            }
            i = nn;
            double energy_weight = sims[i].energy_weight;
            if (targettype == 1) {
                energy_weight *= sqrt(sims[i].inum);
            }
            fprintf(current, "ITEM: TIMESTEP energy, energy_weight, force_weight, nsims, qx, qy, qz, ax, ay, az\n");
            fprintf(current, "%d %f %f %f %d %f %f %f %f %f %f\n", sims[i].timestep, sims[i].energy + sims[i].state_e, energy_weight, sims[i].force_weight, nsims, sims[i].spinvec[0],
                sims[i].spinvec[1], sims[i].spinvec[2], sims[i].spinaxis[0], sims[i].spinaxis[1], sims[i].spinaxis[2]);
            fprintf(current, "ITEM: NUMBER OF ATOMS\n");
            fprintf(current, "%d\n", sims[i].inum);
            fprintf(current, "ITEM: BOX BOUNDS xy xz yz pp pp pp\n");
            double xmin = sims[i].origin[0];
            double xmax = sims[i].box[0][0] + sims[i].origin[0];
            double ymin = sims[i].origin[1];
            double ymax = sims[i].box[1][1] + sims[i].origin[1];
            double zmin = sims[i].origin[2];
            double zmax = sims[i].box[2][2] + sims[i].origin[2];
            if (sims[i].box[0][1] > 0) {
                xmax += sims[i].box[0][1];
            }
            else {
                xmin += sims[i].box[0][1];
            }
            if (sims[i].box[0][2] > 0) {
                xmax += sims[i].box[0][2];
            }
            else {
                xmin += sims[i].box[0][2];
            }
            if (sims[i].box[1][2] > 0) {
                ymax += sims[i].box[1][2];
            }
            else {
                ymin += sims[i].box[1][2];
            }
            fprintf(current, "%f %f %f\n", xmin, xmax, sims[i].box[0][1]);
            fprintf(current, "%f %f %f\n", ymin, ymax, sims[i].box[0][2]);
            fprintf(current, "%f %f %f\n", zmin, zmax, sims[i].box[1][2]);
            fprintf(current, "ITEM: ATOMS id type x y z c_eng c_state c_nn c_test fx fy fz\n");
            for (int j = 0; j < sims[i].inum; j++) {
                fprintf(current, "%d %d %f %f %f %f %f %f %f %f %f\n", sims[i].ilist[j] + 1, sims[i].type[j] + 1, sims[i].x[j][0], sims[i].x[j][1], sims[i].x[j][2], energy_nn[j] + energy_state[j],
                    energy_state[j], energy_nn[j], force[j][0], force[j][1], force[j][2]);
            }
            fclose(dumps[i]);
        }
    }
}

// dump files with numerical forces on each atom along with analytical forces
// VERY SLOW - computes fingerprints for each simulation 6*inum times.
// You can check only state equations by commenting all calls to propagateforward, likewise only network by commenting all state->eos_function calls
void PairRANN::write_debug_level5(double* fit_err, double* val_err)
{
    printf("starting debug level 5\n");
    printf("Very slow. Do NOT run this on more than a handful of atoms!\n");
    double diff = 1e-7; // numerical derivative resolution
    FILE*  dumps[nsims];
    FILE*  dumps1[nsims];
    FILE*  current;
    FILE*  current1;
    char*  debugnames[nsims];
    char*  debugfeaturenames[nsims];
    int    check = mkdir("DEBUG", 0777);
    for (int i = 0; i < nsims; i++) {
        debugnames[i] = new char [strlen(sims[i].filename) + 20];
        debugfeaturenames[i] = new char [strlen(sims[i].filename) + 20];
        sprintf(debugnames[i], "DEBUG/%s.debug5.%d", sims[i].filename, sims[i].timestep);
        sprintf(debugfeaturenames[i], "DEBUG/%s.debug.features.5.%d", sims[i].filename, sims[i].timestep);
    }
#pragma omp parallel
    {
        int i, ii, itype, f, jnum, len, j, nn, s, k, v;
#pragma omp for schedule(guided)
        for (nn = 0; nn < nsims; nn++) {
            dumps[nn]  = fopen(debugnames[nn], "w");
            dumps1[nn] = fopen(debugfeaturenames[nn], "w");
            int j;
            current  = dumps[nn];
            current1 = dumps1[nn];
            double  energy1[sims[nn].inum][3][2];
            double  energyeos1[sims[nn].inum][3][2];
            double  fn[sims[nn].inum][3];
            double  feos[sims[nn].inum][3];
            double* features1[sims[nn].inum][sims[nn].inum][3][2];
            double* dfeatures1[sims[nn].inum][sims[nn].inum][3];
            for (k = 0; k < sims[nn].inum; k++) {
                for (v = 0; v < 3; v++) {
                    for (s = -1; s < 2; s = s + 2) {
                        sims[nn].x[k][v] += s * diff;
                        for (ii = sims[nn].inum; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                            if (sims[nn].id[ii] == k) {
                                sims[nn].x[ii][v] += s * diff;
                            }
                        }
                        double* force[sims[nn].inum + sims[nn].gnum];
                        double  force1[(sims[nn].inum + sims[nn].gnum) * 3];
                        double* fm[sims[nn].inum + sims[nn].gnum];
                        double  fm1[(sims[nn].inum + sims[nn].gnum) * 3];
                        for (ii = 0; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                            force[ii] = &force1[ii * 3];
                            fm[ii]    = &fm1[ii * 3];
                        }
                        double energy[sims[nn].inum];
                        double energyeos[sims[nn].inum];
                        for (ii = 0; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                            for (j = 0; j < 3; j++) {
                                force[ii][j] = 0;
                                fm[ii][j]    = 0;
                            }
                        }
                        for (ii = 0; ii < sims[nn].inum; ii++) {
                            i     = sims[nn].ilist[ii];
                            itype = map(sims[nn].type[i]);
                            f     = net(itype).dimensions[0];
                            features1[k][ii][v][(s + 1) >> 1] = new double [f];
                            dfeatures1[k][ii][v] = new double [f];
                            jnum = sims[nn].numneigh[i];
                            double xn[jnum];
                            double yn[jnum];
                            double zn[jnum];
                            int    tn[jnum];
                            int    jl[jnum];
                            cull_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, cutmax);
                            double features[f];
                            double dfeaturesx[f * jnum];
                            double dfeaturesy[f * jnum];
                            double dfeaturesz[f * jnum];
                            for (j = 0; j < f; j++) {
                                features[j] = 0;
                            }
                            for (j = 0; j < f * jnum; j++) {
                                dfeaturesx[j] = dfeaturesy[j] = dfeaturesz[j] = 0;
                            }
                            // screening is calculated once for all atoms if any fingerprint uses it.
                            double Sik[jnum];
                            double dSikx[jnum];
                            double dSiky[jnum];
                            double dSikz[jnum];
                            // TO D0: add check against stack size
                            double dSijkx[jnum * jnum];
                            double dSijky[jnum * jnum];
                            double dSijkz[jnum * jnum];
                            bool   Bij[jnum];
                            double sx[jnum * f];
                            double sy[jnum * f];
                            double sz[jnum * f];
                            double sxx[jnum * f];
                            double sxy[jnum * f];
                            double sxz[jnum * f];
                            double syy[jnum * f];
                            double syz[jnum * f];
                            double szz[jnum * f];
                            for (j = 0; j < f * jnum; j++) {
                                sx[j]  = sy[j] = sz[j] = 0;
                                sxx[j] = sxy[j] = sxz[j] = syy[j] = syz[j] = szz[j] = 0;
                            }
                            if (doscreen) {
                                screen(Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1);
                            }
                            if (allscreen) {
                                screen_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, Bij, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz);
                            }
                            // do fingerprints for atom type
                            len = fingerprintperelement[itype];
                            for (j = 0; j < len; j++) {
                                if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn,
                                        tn, jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn,
                                        jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz,
                                        dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                            }
                            itype = nelements;
                            // do fingerprints for type "all"
                            len = fingerprintperelement[itype];
                            for (j = 0; j < len; j++) {
                                if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn,
                                        tn, jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn,
                                        jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz,
                                        dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                            }
                            double e  = 0.0;
                            double e1 = 0.0;
                            itype = map(sims[nn].type[i]);
                            len   = stateequationperelement[itype];
                            for (j = 0; j < len; j++) {
                                if (state[itype][j]->screen == false && state[itype][j]->spin == false) {
                                    state[itype][j]->eos_function(&e, force, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (state[itype][j]->screen == true && state[itype][j]->spin == false) {
                                    state[itype][j]->eos_function(&e, force, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (state[itype][j]->screen == false && state[itype][j]->spin == true) {
                                    state[itype][j]->eos_function(&e, force, fm, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (state[itype][j]->screen == true && state[itype][j]->spin == true) {
                                    state[itype][j]->eos_function(&e, force, fm, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                            }
                            itype = nelements;
                            len   = stateequationperelement[itype];
                            for (j = 0; j < len; j++) {
                                if (state[itype][j]->screen == false && state[itype][j]->spin == false) {
                                    state[itype][j]->eos_function(&e, force, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (state[itype][j]->screen == true && state[itype][j]->spin == false) {
                                    state[itype][j]->eos_function(&e, force, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (state[itype][j]->screen == false && state[itype][j]->spin == true) {
                                    state[itype][j]->eos_function(&e, force, fm, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (state[itype][j]->screen == true && state[itype][j]->spin == true) {
                                    state[itype][j]->eos_function(&e, force, fm, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                            }
                            itype = map(sims[nn].type[i]);
                            // NNarchitecture* net_out = new NNarchitecture[nelementsp];
                            auto net_out = CArray<NNarchitecture>(nelementsp);

                            if (normalizeinput) {
                                unnormalize_net(net_out);
                            }
                            else {
                                copy_network(net, net_out);
                            }
                            propagateforward(&e1, force, ii, jnum, itype, features, dfeaturesx, dfeaturesy, dfeaturesz, jl, nn, net_out);
                            energy[ii]    = e + e1;
                            energyeos[ii] = e;
                            for (j = 0; j < f; j++) {
                                features1[k][ii][v][(s + 1) >> 1][j] = features[j];
                            }
                        }
                        sims[nn].x[k][v] -= s * diff;
                        for (ii = sims[nn].inum; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                            if (sims[nn].id[ii] == k) {
                                sims[nn].x[ii][v] -= s * diff;                     // update all ghosts of atom k too
                            }
                        }
                        energy1[k][v][(s + 1) >> 1]    = 0;
                        energyeos1[k][v][(s + 1) >> 1] = 0;
                        for (ii = 0; ii < sims[nn].inum; ii++) {
                            energy1[k][v][(s + 1) >> 1]    += energy[ii];
                            energyeos1[k][v][(s + 1) >> 1] += energyeos[ii];
                            i     = sims[nn].ilist[ii];
                            itype = map(sims[nn].type[i]);
                            f     = net(itype).dimensions[0];
                        }
                    }
                    fn[k][v]   = (energy1[k][v][0] - energy1[k][v][1]) / diff / 2;
                    feos[k][v] = (energyeos1[k][v][0] - energyeos1[k][v][1]) / diff / 2;
                    for (ii = 0; ii < sims[nn].inum; ii++) {
                        i     = sims[nn].ilist[ii];
                        itype = map(sims[nn].type[i]);
                        f     = net(itype).dimensions[0];
                        for (j = 0; j < f; j++) {
                            dfeatures1[k][ii][v][j] = (features1[k][ii][v][1][j] - features1[k][ii][v][0][j]) / diff / 2;
                        }
                    }
                }
            }
            double* force[sims[nn].inum + sims[nn].gnum];
            double  force1[(sims[nn].inum + sims[nn].gnum) * 3];
            double* forceeos[sims[nn].inum + sims[nn].gnum];
            double  forceeos1[(sims[nn].inum + sims[nn].gnum) * 3];
            double* fm[sims[nn].inum + sims[nn].gnum];
            double  fm1[(sims[nn].inum + sims[nn].gnum) * 3];
            for (ii = 0; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                force[ii]    = &force1[ii * 3];
                fm[ii]       = &fm1[ii * 3];
                forceeos[ii] = &forceeos1[ii * 3];
            }
            double energy_nn[sims[nn].inum];
            double energy_state[sims[nn].inum];
            for (ii = 0; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                for (j = 0; j < 3; j++) {
                    force[ii][j]    = 0;
                    forceeos[ii][j] = 0;
                    fm[ii][j] = 0;
                }
            }
            for (ii = 0; ii < sims[nn].inum; ii++) {
                i     = sims[nn].ilist[ii];
                itype = map(sims[nn].type[i]);
                f     = net(itype).dimensions[0];
                jnum  = sims[nn].numneigh[i];
                double xn[jnum];
                double yn[jnum];
                double zn[jnum];
                int    tn[jnum];
                int    jl[jnum];
                cull_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, cutmax);
                double features[f];
                double dfeaturesx[f * jnum];
                double dfeaturesy[f * jnum];
                double dfeaturesz[f * jnum];
                for (j = 0; j < f; j++) {
                    features[j] = 0;
                }
                for (j = 0; j < f * jnum; j++) {
                    dfeaturesx[j] = dfeaturesy[j] = dfeaturesz[j] = 0;
                }
                // screening is calculated once for all atoms if any fingerprint uses it.
                double Sik[jnum];
                double dSikx[jnum];
                double dSiky[jnum];
                double dSikz[jnum];
                // TO D0: add check against stack size
                double dSijkx[jnum * jnum];
                double dSijky[jnum * jnum];
                double dSijkz[jnum * jnum];
                bool   Bij[jnum];
                double sx[jnum * f];
                double sy[jnum * f];
                double sz[jnum * f];
                double sxx[jnum * f];
                double sxy[jnum * f];
                double sxz[jnum * f];
                double syy[jnum * f];
                double syz[jnum * f];
                double szz[jnum * f];
                for (j = 0; j < f * jnum; j++) {
                    sx[j]  = sy[j] = sz[j] = 0;
                    sxx[j] = sxy[j] = sxz[j] = syy[j] = syz[j] = szz[j] = 0;
                }
                if (doscreen) {
                    screen(Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1);
                }
                if (allscreen) {
                    screen_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, Bij, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz);
                }
                // do fingerprints for atom type
                len = fingerprintperelement[itype];
                for (j = 0; j < len; j++) {
                    if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn,
                            jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky,
                            dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                }
                itype = nelements;
                // do fingerprints for type "all"
                len = fingerprintperelement[itype];
                for (j = 0; j < len; j++) {
                    if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn,
                            jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky,
                            dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                }
                double e  = 0.0;
                double e1 = 0.0;
                itype = map(sims[nn].type[i]);
                len   = stateequationperelement[itype];
                for (j = 0; j < len; j++) {
                    if (state[itype][j]->screen == false && state[itype][j]->spin == false) {
                        state[itype][j]->eos_function(&e, forceeos, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == true && state[itype][j]->spin == false) {
                        state[itype][j]->eos_function(&e, forceeos, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == false && state[itype][j]->spin == true) {
                        state[itype][j]->eos_function(&e, forceeos, fm, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == true && state[itype][j]->spin == true) {
                        state[itype][j]->eos_function(&e, forceeos, fm, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                }
                itype = nelements;
                len   = stateequationperelement[itype];
                for (j = 0; j < len; j++) {
                    if (state[itype][j]->screen == false && state[itype][j]->spin == false) {
                        state[itype][j]->eos_function(&e, forceeos, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == true && state[itype][j]->spin == false) {
                        state[itype][j]->eos_function(&e, forceeos, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == false && state[itype][j]->spin == true) {
                        state[itype][j]->eos_function(&e, forceeos, fm, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == true && state[itype][j]->spin == true) {
                        state[itype][j]->eos_function(&e, forceeos, fm, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                }
                itype = map(sims[nn].type[i]);
                // NNarchitecture* net_out = new NNarchitecture[nelementsp];
                auto net_out = CArray<NNarchitecture>(nelementsp);
                if (normalizeinput) {
                    unnormalize_net(net_out);
                }
                else {
                    copy_network(net, net_out);
                }
                propagateforward(&e1, force, ii, jnum, itype, features, dfeaturesx, dfeaturesy, dfeaturesz, jl, nn, net_out);
                energy_nn[ii]    = e1;
                energy_state[ii] = e;

                // fprintf("%d ");
            }
            // apply ghost neighbor forces back into box
            for (ii = sims[nn].inum; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                for (j = 0; j < 3; j++) {
                    force[sims[nn].id[ii]][j]    += force[ii][j];
                    forceeos[sims[nn].id[ii]][j] += forceeos[ii][j];
                }
            }
            i = nn;
            double energy_weight = sims[i].energy_weight;
            if (targettype == 1) {
                energy_weight *= sqrt(sims[i].inum);
            }
            fprintf(current, "ITEM: TIMESTEP energy, energy_weight, force_weight, nsims, qx, qy, qz, ax, ay, az\n");
            fprintf(current, "%d %f %f %f %d %f %f %f %f %f %f\n", sims[i].timestep, sims[i].energy + sims[i].state_e, energy_weight, sims[i].force_weight, nsims, sims[i].spinvec[0],
                sims[i].spinvec[1], sims[i].spinvec[2], sims[i].spinaxis[0], sims[i].spinaxis[1], sims[i].spinaxis[2]);
            fprintf(current, "ITEM: NUMBER OF ATOMS\n");
            fprintf(current, "%d\n", sims[i].inum);
            fprintf(current, "ITEM: BOX BOUNDS xy xz yz pp pp pp\n");
            double xmin = sims[i].origin[0];
            double xmax = sims[i].box[0][0] + sims[i].origin[0];
            double ymin = sims[i].origin[1];
            double ymax = sims[i].box[1][1] + sims[i].origin[1];
            double zmin = sims[i].origin[2];
            double zmax = sims[i].box[2][2] + sims[i].origin[2];
            if (sims[i].box[0][1] > 0) {
                xmax += sims[i].box[0][1];
            }
            else {
                xmin += sims[i].box[0][1];
            }
            if (sims[i].box[0][2] > 0) {
                xmax += sims[i].box[0][2];
            }
            else {
                xmin += sims[i].box[0][2];
            }
            if (sims[i].box[1][2] > 0) {
                ymax += sims[i].box[1][2];
            }
            else {
                ymin += sims[i].box[1][2];
            }
            fprintf(current, "%f %f %f\n", xmin, xmax, sims[i].box[0][1]);
            fprintf(current, "%f %f %f\n", ymin, ymax, sims[i].box[0][2]);
            fprintf(current, "%f %f %f\n", zmin, zmax, sims[i].box[1][2]);
            fprintf(current, "ITEM: ATOMS id type x y z c_eng c_state c_nn fax fnx fay fny faz fnz faeosx fneosx faeosy fneosy faeosz fneosz fannx fnnnx fanny fnnny fannz fnnnz\n");
            for (int j = 0; j < sims[i].inum; j++) {
                fprintf(current, "%d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", sims[i].ilist[j] + 1, sims[i].type[j] + 1, sims[i].x[j][0], sims[i].x[j][1],
                    sims[i].x[j][2], energy_nn[j] + energy_state[j], energy_state[j], energy_nn[j], force[j][0] + forceeos[j][0], fn[j][0], force[j][1] + forceeos[j][1], fn[j][1],
                    force[j][2] + forceeos[j][2], fn[j][2], forceeos[j][0], feos[j][0], forceeos[j][1], feos[j][1], forceeos[j][2], feos[j][2], force[j][0], fn[j][0] - feos[j][0], force[j][1],
                    fn[j][1] - feos[j][1], force[j][2], fn[j][2] - feos[j][2]);
            }
            fclose(dumps[i]);
            errorf(FLERR, "stop");
        }
    }
}

// dump files with numerical spin derivatives on each atom along with analytical forces
// VERY SLOW - computes fingerprints for each simulation 6*inum times.
// You can check only state equations by commenting all calls to propagateforward, likewise only network by commenting all state->eos_function calls
void PairRANN::write_debug_level5_spin(double* fit_err, double* val_err)
{
    printf("starting debug level 5\n");
    printf("Very slow. Do NOT run this on more than a handful of atoms!\n");
    double diff = 1e-6; // numerical derivative resolution
    FILE*  dumps[nsims];
    FILE*  dumps1[nsims];
    FILE*  current;
    FILE*  current1;
    char*  debugnames[nsims];
// char *debugfeaturenames[nsims];
    int check = mkdir("DEBUG", 0777);
    for (int i = 0; i < nsims; i++) {
        debugnames[i] = new char [strlen(sims[i].filename) + 25];
        // debugfeaturenames[i] = new char [strlen(sims[i].filename)+25];
        sprintf(debugnames[i], "DEBUG/%s.debug5.%d", sims[i].filename, sims[i].timestep);
        // sprintf(debugfeaturenames[i],"DEBUG/%s.debug.features.5.%d",sims[i].filename,sims[i].timestep);
    }
#pragma omp parallel
    {
        int i, ii, itype, f, jnum, len, j, nn, s, k, v;
#pragma omp for schedule(guided)
        for (nn = 0; nn < nsims; nn++) {
            dumps[nn] = fopen(debugnames[nn], "w");
            // dumps1[nn]=fopen(debugfeaturenames[nn],"w");
            int j;
            current = dumps[nn];

            double* force[sims[nn].inum + sims[nn].gnum];
            double  force1[(sims[nn].inum + sims[nn].gnum) * 3];
            double* forceeos[sims[nn].inum + sims[nn].gnum];
            double  forceeos1[(sims[nn].inum + sims[nn].gnum) * 3];
            double* fm[sims[nn].inum + sims[nn].gnum];
            double* hm[sims[nn].inum + sims[nn].gnum];
            double  fm1[(sims[nn].inum + sims[nn].gnum) * 3];
            double  hm1[(sims[nn].inum + sims[nn].gnum) * 6];
            for (ii = 0; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                force[ii]    = &force1[ii * 3];
                fm[ii]       = &fm1[ii * 3];
                hm[ii]       = &hm1[ii * 6];
                forceeos[ii] = &forceeos1[ii * 3];
            }
            double energy_nn[sims[nn].inum];
            double energy_state[sims[nn].inum];
            for (ii = 0; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                for (j = 0; j < 3; j++) {
                    force[ii][j]    = 0;
                    forceeos[ii][j] = 0;
                    fm[ii][j] = 0;
                }
                for (j = 0; j < 6; j++) {
                    hm[ii][j] = 0;
                }
            }
            for (ii = 0; ii < sims[nn].inum; ii++) {
                i     = sims[nn].ilist[ii];
                itype = map(sims[nn].type[i]);
                f     = net(itype).dimensions[0];
                jnum  = sims[nn].numneigh[i];
                double xn[jnum];
                double yn[jnum];
                double zn[jnum];
                int    tn[jnum];
                int    jl[jnum];
                cull_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, cutmax);
                double features[f];
                double dfeaturesx[f * jnum];
                double dfeaturesy[f * jnum];
                double dfeaturesz[f * jnum];
                for (j = 0; j < f; j++) {
                    features[j] = 0;
                }
                for (j = 0; j < f * jnum; j++) {
                    dfeaturesx[j] = dfeaturesy[j] = dfeaturesz[j] = 0;
                }
                // screening is calculated once for all atoms if any fingerprint uses it.
                double Sik[jnum];
                double dSikx[jnum];
                double dSiky[jnum];
                double dSikz[jnum];
                // TO D0: add check against stack size
                double dSijkx[jnum * jnum];
                double dSijky[jnum * jnum];
                double dSijkz[jnum * jnum];
                bool   Bij[jnum];
                double sx[jnum * f];
                double sy[jnum * f];
                double sz[jnum * f];
                double sxx[jnum * f];
                double sxy[jnum * f];
                double sxz[jnum * f];
                double syy[jnum * f];
                double syz[jnum * f];
                double szz[jnum * f];
                for (j = 0; j < f * jnum; j++) {
                    sx[j]  = sy[j] = sz[j] = 0;
                    sxx[j] = sxy[j] = sxz[j] = syy[j] = syz[j] = szz[j] = 0;
                }
                if (doscreen) {
                    screen(Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1);
                }
                if (allscreen) {
                    screen_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, Bij, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz);
                }
                // do fingerprints for atom type
                len = fingerprintperelement[itype];
                for (j = 0; j < len; j++) {
                    if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn,
                            jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky,
                            dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                }
                itype = nelements;
                // do fingerprints for type "all"
                len = fingerprintperelement[itype];
                for (j = 0; j < len; j++) {
                    if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn,
                            jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                        fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky,
                            dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                }
                double e  = 0.0;
                double e1 = 0.0;
                itype = map(sims[nn].type[i]);
                len   = stateequationperelement[itype];
                for (j = 0; j < len; j++) {
                    if (state[itype][j]->screen == false && state[itype][j]->spin == false) {
                        state[itype][j]->eos_function(&e, forceeos, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == true && state[itype][j]->spin == false) {
                        state[itype][j]->eos_function(&e, forceeos, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == false && state[itype][j]->spin == true) {
                        state[itype][j]->eos_function(&e, forceeos, fm, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == true && state[itype][j]->spin == true) {
                        state[itype][j]->eos_function(&e, forceeos, fm, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                }
                itype = nelements;
                len   = stateequationperelement[itype];
                for (j = 0; j < len; j++) {
                    if (state[itype][j]->screen == false && state[itype][j]->spin == false) {
                        state[itype][j]->eos_function(&e, forceeos, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == true && state[itype][j]->spin == false) {
                        state[itype][j]->eos_function(&e, forceeos, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == false && state[itype][j]->spin == true) {
                        state[itype][j]->eos_function(&e, forceeos, fm, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                    else if (state[itype][j]->screen == true && state[itype][j]->spin == true) {
                        state[itype][j]->eos_function(&e, forceeos, fm, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                    }
                }
                itype = map(sims[nn].type[i]);
                // NNarchitecture* net_out = new NNarchitecture[nelementsp];
                auto net_out = CArray<NNarchitecture>(nelementsp);
                if (normalizeinput) {
                    unnormalize_net(net_out);
                }
                else {
                    copy_network(net, net_out);
                }
                propagateforwardspin(&e1, force, fm, hm, ii, jnum, itype, features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, jl, nn, net_out);
                energy_nn[ii]    = e1;
                energy_state[ii] = e;

                // fprintf("%d ");
            }
            // apply ghost neighbor forces back into box
            for (ii = sims[nn].inum; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                for (j = 0; j < 3; j++) {
                    force[sims[nn].id[ii]][j]    += force[ii][j];
                    forceeos[sims[nn].id[ii]][j] += forceeos[ii][j];
                    fm[sims[nn].id[ii]][j] += fm[ii][j];
                }
                for (j = 0; j < 6; j++) {
                    hm[sims[nn].id[ii]][j] += hm[ii][j];
                }
            }

            double energy0 = 0;
            for (ii = 0; ii < sims[nn].inum; ii++) {
                energy0 += energy_nn[ii] + energy_state[ii];
            }
            // current1 = dumps1[nn];
            double  energy1[sims[nn].inum][9][2];
            double  energyeos1[sims[nn].inum][9][2];
            double  fn[sims[nn].inum][3];
            double  feos[sims[nn].inum][3];
            double  hn[sims[nn].inum][6];
            double* features1[sims[nn].inum][sims[nn].inum][3][2];
            double* dfeatures1[sims[nn].inum][sims[nn].inum][3];
            for (k = 0; k < sims[nn].inum; k++) {
                for (v = 0; v < 9; v++) {
                    for (s = -1; s < 2; s = s + 2) {
                        for (ii = 0; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                            if (v < 3) {
                                if (sims[nn].id[ii] == k) {
                                    sims[nn].s[ii][v] += s * diff;
                                }
                            }
                            else {
                                if (v == 3) {
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][0] += s * diff;
                                    }
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][0] += s * diff;
                                    }
                                }
                                if (v == 4) {
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][0] += s * diff;
                                    }
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][1] += s * diff;
                                    }
                                }
                                if (v == 5) {
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][0] += s * diff;
                                    }
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][2] += s * diff;
                                    }
                                }
                                if (v == 6) {
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][1] += s * diff;
                                    }
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][1] += s * diff;
                                    }
                                }
                                if (v == 7) {
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][1] += s * diff;
                                    }
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][2] += s * diff;
                                    }
                                }
                                if (v == 8) {
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][2] += s * diff;
                                    }
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][2] += s * diff;
                                    }
                                }
                            }
                        }
                        printf("%f %d %d %d 0\n", hn[0][5], k, s, v);
                        double* force[sims[nn].inum + sims[nn].gnum];
                        double  force1[(sims[nn].inum + sims[nn].gnum) * 3];
                        double* fm2[sims[nn].inum + sims[nn].gnum];
                        double  fm1[(sims[nn].inum + sims[nn].gnum) * 3];
                        double* hm2[sims[nn].inum + sims[nn].gnum];
                        double  hm1[(sims[nn].inum + sims[nn].gnum) * 6];
                        for (ii = 0; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                            force[ii] = &force1[ii * 3];
                            fm2[ii]   = &fm1[ii * 3];
                            hm2[ii]   = &hm1[ii * 6];
                        }
                        double energy[sims[nn].inum];
                        double energyeos[sims[nn].inum];
                        for (ii = 0; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                            for (j = 0; j < 3; j++) {
                                force[ii][j] = 0;
                                fm2[ii][j]   = 0;
                            }
                            for (j = 0; j < 6; j++) {
                                hm2[ii][j] = 0;
                            }
                        }
                        printf("%f %d %d %d 2\n", hn[0][5], k, s, v);
                        for (ii = 0; ii < sims[nn].inum; ii++) {
                            i     = sims[nn].ilist[ii];
                            itype = map(sims[nn].type[i]);
                            f     = net(itype).dimensions[0];
                            features1[k][ii][v][(s + 1) >> 1] = new double [f];
                            dfeatures1[k][ii][v] = new double [f];
                            jnum = sims[nn].numneigh[i];
                            double xn[jnum];
                            double yn[jnum];
                            double zn[jnum];
                            int    tn[jnum];
                            int    jl[jnum];
                            printf("%f %d %d %d %d 1\n", hn[0][5], k, s, v, ii);
                            cull_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, cutmax);
                            double features[f];
                            double dfeaturesx[f * jnum];
                            double dfeaturesy[f * jnum];
                            double dfeaturesz[f * jnum];
                            for (j = 0; j < f; j++) {
                                features[j] = 0;
                            }
                            for (j = 0; j < f * jnum; j++) {
                                dfeaturesx[j] = dfeaturesy[j] = dfeaturesz[j] = 0;
                            }
                            printf("%f %d %d %d %d 2\n", hn[0][5], k, s, v, ii);
                            // screening is calculated once for all atoms if any fingerprint uses it.
                            double Sik[jnum];
                            double dSikx[jnum];
                            double dSiky[jnum];
                            double dSikz[jnum];
                            // TO D0: add check against stack size
                            double dSijkx[jnum * jnum];
                            double dSijky[jnum * jnum];
                            double dSijkz[jnum * jnum];
                            bool   Bij[jnum];
                            double sx[jnum * f];
                            double sy[jnum * f];
                            double sz[jnum * f];
                            double sxx[jnum * f];
                            double sxy[jnum * f];
                            double sxz[jnum * f];
                            double syy[jnum * f];
                            double syz[jnum * f];
                            double szz[jnum * f];
                            for (j = 0; j < f * jnum; j++) {
                                sx[j]  = sy[j] = sz[j] = 0;
                                sxx[j] = sxy[j] = sxz[j] = syy[j] = syz[j] = szz[j] = 0;
                            }
                            if (doscreen) {
                                screen(Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1);
                            }
                            if (allscreen) {
                                screen_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, Bij, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz);
                            }
                            printf("%f %d %d %d %d 3\n", hn[0][5], k, s, v, ii);
                            // do fingerprints for atom type
                            len = fingerprintperelement[itype];
                            for (j = 0; j < len; j++) {
                                if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn,
                                        tn, jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn,
                                        jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz,
                                        dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                            }
                            itype = nelements;
                            // do fingerprints for type "all"
                            len = fingerprintperelement[itype];
                            for (j = 0; j < len; j++) {
                                if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn,
                                        tn, jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn,
                                        jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz,
                                        dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                            }
                            double e  = 0.0;
                            double e1 = 0.0;
                            itype = map(sims[nn].type[i]);
                            len   = stateequationperelement[itype];
                            printf("%f %d %d %d %d 4\n", hn[0][5], k, s, v, ii);
                            for (j = 0; j < len; j++) {
                                if (state[itype][j]->screen == false && state[itype][j]->spin == false) {
                                    state[itype][j]->eos_function(&e, force, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (state[itype][j]->screen == true && state[itype][j]->spin == false) {
                                    state[itype][j]->eos_function(&e, force, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (state[itype][j]->screen == false && state[itype][j]->spin == true) {
                                    state[itype][j]->eos_function(&e, force, fm2, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (state[itype][j]->screen == true && state[itype][j]->spin == true) {
                                    state[itype][j]->eos_function(&e, force, fm2, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                            }
                            itype = nelements;
                            len   = stateequationperelement[itype];
                            for (j = 0; j < len; j++) {
                                if (state[itype][j]->screen == false && state[itype][j]->spin == false) {
                                    state[itype][j]->eos_function(&e, force, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (state[itype][j]->screen == true && state[itype][j]->spin == false) {
                                    state[itype][j]->eos_function(&e, force, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (state[itype][j]->screen == false && state[itype][j]->spin == true) {
                                    state[itype][j]->eos_function(&e, force, fm2, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (state[itype][j]->screen == true && state[itype][j]->spin == true) {
                                    state[itype][j]->eos_function(&e, force, fm2, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, i, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                            }
                            itype = map(sims[nn].type[i]);
                            // NNarchitecture* net_out = new NNarchitecture[nelementsp];
                            auto net_out = CArray<NNarchitecture>(nelementsp);
                            if (normalizeinput) {
                                unnormalize_net(net_out);
                            }
                            else {
                                copy_network(net, net_out);
                            }
                            printf("%f %d %d %d %d 5\n", hn[0][5], k, s, v, ii);
                            propagateforwardspin(&e1, force, fm2, hm2, ii, jnum, itype, features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, jl, nn, net_out);
                            energy[ii]    = e + e1;
                            energyeos[ii] = e;
                            for (j = 0; j < f; j++) {
                                features1[k][ii][v][(s + 1) >> 1][j] = features[j];
                            }
                            printf("%f %d %d %d %d 6\n", hn[0][5], k, s, v, ii);
                        }
                        printf("%f %d %d %d 1\n", hn[0][5], k, s, v);
                        for (ii = 0; ii < sims[nn].inum + sims[nn].gnum; ii++) {
                            if (v < 3) {
                                if (sims[nn].id[ii] == k) {
                                    sims[nn].s[ii][v] -= s * diff;
                                }
                            }
                            else {
                                if (v == 3) {
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][0] -= s * diff;
                                    }
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][0] -= s * diff;
                                    }
                                }
                                if (v == 4) {
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][0] -= s * diff;
                                    }
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][1] -= s * diff;
                                    }
                                }
                                if (v == 5) {
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][0] -= s * diff;
                                    }
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][2] -= s * diff;
                                    }
                                }
                                if (v == 6) {
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][1] -= s * diff;
                                    }
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][1] -= s * diff;
                                    }
                                }
                                if (v == 7) {
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][1] -= s * diff;
                                    }
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][2] -= s * diff;
                                    }
                                }
                                if (v == 8) {
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][2] -= s * diff;
                                    }
                                    if (sims[nn].id[ii] == k) {
                                        sims[nn].s[ii][2] -= s * diff;
                                    }
                                }
                            }
                        }
                        energy1[k][v][(s + 1) >> 1]    = 0;
                        energyeos1[k][v][(s + 1) >> 1] = 0;
                        for (ii = 0; ii < sims[nn].inum; ii++) {
                            energy1[k][v][(s + 1) >> 1]    += energy[ii];
                            energyeos1[k][v][(s + 1) >> 1] += energyeos[ii];
                            i     = sims[nn].ilist[ii];
                            itype = map(sims[nn].type[i]);
                            f     = net(itype).dimensions[0];
                        }
                    }
                    if (v < 3) {
                        fn[k][v]   = (energy1[k][v][0] - energy1[k][v][1]) / diff / 2;
                        fn[k][v]  /= hbar;
                        feos[k][v] = (energyeos1[k][v][0] - energyeos1[k][v][1]) / diff / 2;
                        for (ii = 0; ii < sims[nn].inum; ii++) {
                            i     = sims[nn].ilist[ii];
                            itype = map(sims[nn].type[i]);
                            f     = net(itype).dimensions[0];
                            for (j = 0; j < f; j++) {
                                dfeatures1[k][ii][v][j] = (features1[k][ii][v][1][j] - features1[k][ii][v][0][j]) / diff / 2;
                            }
                        }
                    }
                    else if (v == 3) {
                        hn[k][0] = ((energy1[k][0][0] - energy0) / diff - (energy0 - energy1[k][0][1]) / diff) / diff / hbar;
                    }
                    else if (v == 4) {
                        hn[k][1] = ((energy1[k][4][0] - energy1[k][1][0]) / diff - (energy1[k][0][0] - energy0) / diff) / diff / 2 +
                            ((energy0 - energy1[k][0][1]) / diff - (energy1[k][1][1] - energy1[k][4][1]) / diff) / diff / 2 / hbar;
                    }
                    else if (v == 5) {
                        hn[k][2] = ((energy1[k][5][0] - energy1[k][2][0]) / diff - (energy1[k][0][0] - energy0) / diff) / diff / 2 +
                            ((energy0 - energy1[k][0][1]) / diff - (energy1[k][2][1] - energy1[k][5][1]) / diff) / diff / 2 / hbar;
                    }
                    else if (v == 6) {
                        hn[k][3] = ((energy1[k][1][0] - energy0) / diff - (energy0 - energy1[k][1][1]) / diff) / diff / hbar;
                    }
                    else if (v == 7) {
                        hn[k][4] = ((energy1[k][7][0] - energy1[k][2][0]) / diff - (energy1[k][1][0] - energy0) / diff) / diff / 2 +
                            ((energy0 - energy1[k][1][1]) / diff - (energy1[k][2][1] - energy1[k][7][1]) / diff) / diff / 2 / hbar;
                    }
                    else if (v == 8) {
                        hn[k][5] = ((energy1[k][2][0] - energy0) / diff - (energy0 - energy1[k][2][1]) / diff) / diff / hbar;
                        printf("%f\n", hn[0][5]);
                    }
                }
            }
            i = nn;
            double energy_weight = sims[i].energy_weight;
            if (targettype == 1) {
                energy_weight *= sqrt(sims[i].inum);
            }
            fprintf(current, "ITEM: TIMESTEP energy, energy_weight, force_weight, nsims, qx, qy, qz, ax, ay, az\n");
            fprintf(current, "%d %f %f %f %d %f %f %f %f %f %f\n", sims[i].timestep, sims[i].energy + sims[i].state_e, energy_weight, sims[i].force_weight, nsims, sims[i].spinvec[0],
                sims[i].spinvec[1], sims[i].spinvec[2], sims[i].spinaxis[0], sims[i].spinaxis[1], sims[i].spinaxis[2]);
            fprintf(current, "ITEM: NUMBER OF ATOMS\n");
            fprintf(current, "%d\n", sims[i].inum);
            fprintf(current, "ITEM: BOX BOUNDS xy xz yz pp pp pp\n");
            double xmin = sims[i].origin[0];
            double xmax = sims[i].box[0][0] + sims[i].origin[0];
            double ymin = sims[i].origin[1];
            double ymax = sims[i].box[1][1] + sims[i].origin[1];
            double zmin = sims[i].origin[2];
            double zmax = sims[i].box[2][2] + sims[i].origin[2];
            if (sims[i].box[0][1] > 0) {
                xmax += sims[i].box[0][1];
            }
            else {
                xmin += sims[i].box[0][1];
            }
            if (sims[i].box[0][2] > 0) {
                xmax += sims[i].box[0][2];
            }
            else {
                xmin += sims[i].box[0][2];
            }
            if (sims[i].box[1][2] > 0) {
                ymax += sims[i].box[1][2];
            }
            else {
                ymin += sims[i].box[1][2];
            }
            fprintf(current, "%f %f %f\n", xmin, xmax, sims[i].box[0][1]);
            fprintf(current, "%f %f %f\n", ymin, ymax, sims[i].box[0][2]);
            fprintf(current, "%f %f %f\n", zmin, zmax, sims[i].box[1][2]);
            fprintf(current, "ITEM: ATOMS id type x y z sx sy sz c_eng c_state c_nn fmnx fmax fmny fmay fmnz fmaz hnxx haxx hnxy haxy hnxz haxz hnyy hayy hnyz hayz hnzz hazz\n");
            for (int j = 0; j < sims[i].inum; j++) {
                fprintf(current, "%d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", sims[i].ilist[j] + 1, sims[i].type[j] + 1, sims[i].x[j][0],
                    sims[i].x[j][1], sims[i].x[j][2], sims[i].s[j][0], sims[i].s[j][1], sims[i].s[j][2], energy_nn[j] + energy_state[j], energy_state[j], energy_nn[j], fn[j][0], fm[j][0], fn[j][1],
                    fm[j][1], fn[j][2], fm[j][2], hn[j][0], hm[j][0], hn[j][1], hm[j][1], hn[j][2], hm[j][2], hn[j][3], hm[j][3], hn[j][4], hm[j][4], hn[j][5], hm[j][5]);
            }
            fclose(dumps[i]);
            // errorf(FLERR,"stop");
        }
    }
    errorf(FLERR, "stop");
}

// Calculate numerical derivatives of features wrt to neighbor list and compare with analytical derivatives
void PairRANN::write_debug_level6(double* fit_err, double* val_err)
{
    printf("starting debug level 6\n");
    printf("Very slow. Do NOT run this on more than a handful of atoms!\n");
    double diff = 1e-5;    // numerical derivative resolution
    FILE*  dumps1[nsims];
    FILE*  dumps2[nsims];
    FILE*  current1;
    FILE*  current2;
    char*  debugnames1[nsims];
    char*  debugnames2[nsims];
    int    check = mkdir("DEBUG", 0777);
    for (int i = 0; i < nsims; i++) {
        debugnames1[i] = new char [strlen(sims[i].filename) + 30];
        sprintf(debugnames1[i], "DEBUG/%s.features.debug6.%d.csv", sims[i].filename, sims[i].timestep);
        debugnames2[i] = new char [strlen(sims[i].filename) + 30];
        sprintf(debugnames2[i], "DEBUG/%s.screening.debug6.%d.csv", sims[i].filename, sims[i].timestep);
    }
    int fmax = 0;
    for (int itype = 0; itype < nelementsp; itype++) {
        if (net(itype).layers == 0) {
            continue;
        }
        if (net(itype).dimensions[0] > fmax) {
            fmax = net(itype).dimensions[0];
        }
    }
        #pragma omp parallel
    {
        int i, ii, itype, f, jnum, len, j, nn, s, k, v;
        #pragma omp for schedule(guided)
        for (nn = 0; nn < nsims; nn++) {
            int nmax = 0;
            for (j = 0; j < sims[nn].inum; j++) {
                if (nmax < sims[nn].numneigh[j]) {
                    nmax = sims[nn].numneigh[j];
                }
            }
            dumps1[nn] = fopen(debugnames1[nn], "w");
            dumps2[nn] = fopen(debugnames2[nn], "w");
            int j;
            current1 = dumps1[nn];
            current2 = dumps2[nn];
            fprintf(current1, "sim,atom,neighbor,neighbor_type,type,neighbor_id");
            fprintf(current2, "sim,atom,neighbor,neighbor_type,type,neighbor_id,Sik");
            for (f = 0; f < fmax; f++) {
                fprintf(current1, ",features%d,dfxn%d,dfxa%d,dfyn%d,dfya%d,dfzn%d,dfza%d", f, f, f, f, f, f, f);
            }
            fprintf(current1, "\n");
            for (f = 0; f < nmax; f++) {
                fprintf(current2, ",dSikxn%d,dSikxa%d,dSikyn%d,dSikya%d,dSikzn%d,dSikza%d", f, f, f, f, f, f);
            }
            fprintf(current2, "\n");
            for (ii = 0; ii < sims[nn].inum; ii++) {
                i     = sims[nn].ilist[ii];
                itype = map(sims[nn].type[i]);
                f     = net(itype).dimensions[0];
                jnum  = sims[nn].numneigh[i];
                double xn[jnum];
                double yn[jnum];
                double zn[jnum];
                int    tn[jnum];
                int    jl[jnum];
                cull_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, cutmax);
                double features1[jnum][f][3][2];
                double dfeatures1[jnum][f][3];
                double screenik[jnum][3][2];
                double dscreenik[jnum][3];
                double screenijk[jnum][jnum][3][2];
                double dscreenijk[jnum][jnum][3];
                double features[f];
                double dfeaturesx[f * jnum];
                double dfeaturesy[f * jnum];
                double dfeaturesz[f * jnum];
                int    jnum1 = jnum;
                for (int jj = 0; jj < jnum1; jj++) {
                    int j1 = sims[nn].firstneigh[i][jj];
                    for (v = 0; v < 3; v++) {
                        for (s = -1; s < 2; s = s + 2) {
                            // printf("%d %d %d %d %d\n",nn,ii,jj,v,s);
                            if (jj < jnum - 1) {
                                if (v == 0) {
                                    sims[nn].x[j1][0] -= s * diff;
                                }
                                else if (v == 1) {
                                    sims[nn].x[j1][1] -= s * diff;
                                }
                                else if (v == 2) {
                                    sims[nn].x[j1][2] -= s * diff;
                                }
                            }
                            else {
                                if (v == 0) {
                                    sims[nn].x[i][0] -= s * diff;
                                }
                                else if (v == 1) {
                                    sims[nn].x[i][1] -= s * diff;
                                }
                                else if (v == 2) {
                                    sims[nn].x[i][2] -= s * diff;
                                }
                            }
                            jnum = sims[nn].numneigh[i];
                            cull_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, cutmax);
                            for (j = 0; j < f; j++) {
                                features[j] = 0;
                            }
                            for (j = 0; j < f * jnum; j++) {
                                dfeaturesx[j] = dfeaturesy[j] = dfeaturesz[j] = 0.0;
                            }
                            // screening is calculated once for all atoms if any fingerprint uses it.
                            double Sik[jnum];
                            double dSikx[jnum];
                            double dSiky[jnum];
                            double dSikz[jnum];
                            // TO D0: add check against stack size
                            double dSijkx[jnum * jnum];
                            double dSijky[jnum * jnum];
                            double dSijkz[jnum * jnum];
                            bool   Bij[jnum];
                            double sx[jnum * f];
                            double sy[jnum * f];
                            double sz[jnum * f];
                            double sxx[jnum * f];
                            double sxy[jnum * f];
                            double sxz[jnum * f];
                            double syy[jnum * f];
                            double syz[jnum * f];
                            double szz[jnum * f];
                            for (j = 0; j < f * jnum; j++) {
                                sx[j]  = sy[j] = sz[j] = 0;
                                sxx[j] = sxy[j] = sxz[j] = syy[j] = syz[j] = szz[j] = 0;
                            }
                            if (doscreen) {
                                screen(Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1);
                            }
                            // do fingerprints for atom type
                            len = fingerprintperelement[itype];
                            for (j = 0; j < len; j++) {
                                if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn,
                                        tn, jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn,
                                        jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz,
                                        dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                            }
                            itype = nelements;
                            // do fingerprints for type "all"
                            len = fingerprintperelement[itype];
                            for (j = 0; j < len; j++) {
                                if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn,
                                        tn, jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn,
                                        jnum - 1, jl);
                                }
                                else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                                    fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz,
                                        dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                                }
                            }
                            for (j = 0; j < f; j++) {
                                features1[jj][j][v][(s + 1) >> 1] = features[j];
                            }
                            screenik[jj][v][(s + 1) >> 1] = Sik[jj];
                            for (j = 0; j < jnum - 1; j++) {
                                screenijk[jj][j][v][(s + 1) >> 1] = Sik[j];
                            }
                            itype = map(sims[nn].type[i]);
                            if (jj < jnum - 1) {
                                if (v == 0) {
                                    sims[nn].x[j1][0] += s * diff;
                                }
                                else if (v == 1) {
                                    sims[nn].x[j1][1] += s * diff;
                                }
                                else if (v == 2) {
                                    sims[nn].x[j1][2] += s * diff;
                                }
                            }
                            else {
                                if (v == 0) {
                                    sims[nn].x[i][0] += s * diff;
                                }
                                else if (v == 1) {
                                    sims[nn].x[i][1] += s * diff;
                                }
                                else if (v == 2) {
                                    sims[nn].x[i][2] += s * diff;
                                }
                            }
                        }
                        for (j = 0; j < f; j++) {
                            dfeatures1[jj][j][v] = (features1[jj][j][v][1] - features1[jj][j][v][0]) / diff / 2;
                        }
                        dscreenik[jj][v] = (screenik[jj][v][1] - screenik[jj][v][0]) / diff / 2;
                        for (j = 0; j < jnum - 1; j++) {
                            dscreenijk[jj][j][v] = (screenijk[jj][j][v][1] - screenijk[jj][j][v][0]) / diff / 2;
                        }
                    }
                    jnum = sims[nn].numneigh[i];
                    cull_neighbor_list(xn, yn, zn, tn, &jnum, jl, i, nn, cutmax);
                    double features[f];
                    double dfeaturesx[f * jnum];
                    double dfeaturesy[f * jnum];
                    double dfeaturesz[f * jnum];
                    for (j = 0; j < f; j++) {
                        features[j] = 0;
                    }
                    for (j = 0; j < f * jnum; j++) {
                        dfeaturesx[j] = dfeaturesy[j] = dfeaturesz[j] = 0;
                    }
                    // screening is calculated once for all atoms if any fingerprint uses it.
                    double Sik[jnum];
                    double dSikx[jnum];
                    double dSiky[jnum];
                    double dSikz[jnum];
                    // TO D0: add check against stack size
                    double dSijkx[jnum * jnum];
                    double dSijky[jnum * jnum];
                    double dSijkz[jnum * jnum];
                    bool   Bij[jnum];
                    double sx[jnum * f];
                    double sy[jnum * f];
                    double sz[jnum * f];
                    double sxx[jnum * f];
                    double sxy[jnum * f];
                    double sxz[jnum * f];
                    double syy[jnum * f];
                    double syz[jnum * f];
                    double szz[jnum * f];
                    for (j = 0; j < f * jnum; j++) {
                        sx[j]  = sy[j] = sz[j] = 0;
                        sxx[j] = sxy[j] = sxz[j] = syy[j] = syz[j] = szz[j] = 0;
                    }
                    if (doscreen) {
                        screen(Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1);
                    }
                    if (jj < jnum - 1) {
                        fprintf(current2, "%d, %d, %d, %d, %d, %d", nn, ii, jj, tn[jj], itype, jl[jj]);
                        fprintf(current2, ", %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e", Sik[jj], dscreenik[jj][0], dSikx[jj] * Sik[jj], dscreenik[jj][1], dSiky[jj] * Sik[jj], dscreenik[jj][2],
                            dSikz[jj] * Sik[jj]);
                        for (j = 0; j < jnum - 1; j++) {
                            if (j == jj) {
                                fprintf(current2, ", 0, 0, 0, 0, 0, 0");
                            }
                            else{
                                fprintf(current2, ", %.10e, %.10e, %.10e, %.10e, %.10e, %.10e", dscreenijk[jj][j][0], dSijkx[j * (jnum - 1) + jj] * Sik[j], dscreenijk[jj][j][1],
                                    dSijky[j * (jnum - 1) + jj] * Sik[j], dscreenijk[jj][j][2], dSijkz[j * (jnum - 1) + jj] * Sik[j]);
                            }
                        }
                        fprintf(current2, "\n");
                    }
                    // do fingerprints for atom type
                    len = fingerprintperelement[itype];
                    for (j = 0; j < len; j++) {
                        if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                            fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                        }
                        else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                            fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn,
                                jnum - 1, jl);
                        }
                        else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                            fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                        }
                        else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                            fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz, dSijkx,
                                dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                        }
                    }
                    itype = nelements;
                    // do fingerprints for type "all"
                    len = fingerprintperelement[itype];
                    for (j = 0; j < len; j++) {
                        if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == false) {
                            fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                        }
                        else if (fingerprints[itype][j]->spin == false && fingerprints[itype][j]->screen == true) {
                            fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, Sik, dSikx, dSiky, dSikz, dSijkx, dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn,
                                jnum - 1, jl);
                        }
                        else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == false) {
                            fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                        }
                        else if (fingerprints[itype][j]->spin == true && fingerprints[itype][j]->screen == true) {
                            fingerprints[itype][j]->compute_fingerprint(features, dfeaturesx, dfeaturesy, dfeaturesz, sx, sy, sz, sxx, sxy, sxz, syy, syz, szz, Sik, dSikx, dSiky, dSikz, dSijkx,
                                dSijky, dSijkz, Bij, ii, nn, xn, yn, zn, tn, jnum - 1, jl);
                        }
                    }
                    itype = map(sims[nn].type[i]);
                    fprintf(current1, "%d, %d, %d, %d, %d, %d", nn, ii, jj, tn[jj], itype, jl[jj]);
                    for (j = 0; j < f; j++) {
                        fprintf(current1, ", %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e", features[j], dfeatures1[jj][j][0], dfeaturesx[jj * f + j], dfeatures1[jj][j][1], dfeaturesy[jj * f + j],
                            dfeatures1[jj][j][2], dfeaturesz[jj * f + j]);
                    }
                    fprintf(current1, "\n");
                }
            }
            fclose(dumps1[i]);
            fclose(dumps2[i]);
        }
    }
}