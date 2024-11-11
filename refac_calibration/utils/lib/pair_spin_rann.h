
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <map>
#include <dirent.h>
#include <math.h>
#include <cmath>
#include <time.h>
#include <sys/resource.h>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "utils.h"
#define MAXLINE 4096
#define SHORTLINE 128
#define NEIGHMASK 0x3FFFFFFF
#define FLERR __FILE__, __LINE__


#include "matar.h"

using namespace mtr; // matar namespace

// Typedef for defining floating point precision 
typedef double real_t;


// Template name for dual type for 
template <typename T>
using DualCArray = DCArrayKokkos <T>;

template <typename T>
using ViewCArray = ViewCArrayKokkos <T>;



template <typename T>
using DualFArray = DFArrayKokkos <T>;


template <typename T>
using ViewFArray = ViewCArrayKokkos <T>;



#ifndef CALIBRATION_H_
#define CALIBRATION_H_

namespace LAMMPS_NS
{
namespace RANN
{
class Activation;
class Fingerprint;
class State;
} // namespace RANN

class PairRANN
{
public:
    PairRANN(char*);
    ~PairRANN();
    void setup();
    void run();
    void finish();

    // global parameters read from file, generally for parsing
    char*  algorithm;               // 1D DEFINE: 
    char*  potential_input_file;    // 1D DEFINE:
    char*  dump_directory;          // 1D DEFINE:
    bool   doforces;
    double tolerance;   // DEFINE: convergence tolerance for calibration
    double regularizer; // DEFINE:
    bool   doregularizer;
    char*  log_file;        // 1D
    char*  potential_output_file;   // 1D
    int    potential_output_freq;   
    int    max_epochs;
    bool   overwritepotentials;
    int    debug_level1_freq;
    int    debug_level2_freq;
    int    debug_level3_freq;
    int    debug_level4_freq;
    int    debug_level5_freq;
    int    debug_level5_spin_freq;
    int    debug_level6_freq;
    bool   adaptive_regularizer; // DEFINE:
    double lambda_initial; // DEFINE:
    double lambda_increase;// DEFINE:
    double lambda_reduce;// DEFINE:
    int    seed;
    double validation; // DEFINE:
    bool   normalizeinput; // DEFINE:
    int    targettype; // DEFINE:
    double inum_weight; // DEFINE:

    // global variables calculated internally
    bool     is_lammps = false;
    char*    lmp = nullptr; // 1D DEFINE:
    int      nsims;// DEFINE:
    int      nsets;// DEFINE:
    int      betalen;// DEFINE:
    int      jlen1; // DEFINE:

    DualCArray<int> betalen_v; // 1D DEFINE: NOTE: Possibly not needed on device
    DualCArray<int> betalen_f; // 1D DEFINE: NOTE: Possibly not needed on device
 
    int      natoms; // DEFINE:
    int      natomsr; // DEFINE:
    int      natomsv; // DEFINE:
    int      fmax;  // DEFINE:
    int      fnmax; // DEFINE:
    


    DualCArray<int> r; // simulations included in training // 1D // DEFINE:
    DualCArray<int> v; // simulations held back for validation // 1D // DEFINE:

    int      nsimr, nsimv;  // DEFINE:
    
    DualCArray<int> Xset;  // 1D 

    char**   dumpfilenames; // 2D
    
    RaggedRightArrayKokkos<real_t> normalshift; // 2D   // DEFINE:
    RaggedRightArrayKokkos<real_t> normalgain; // 2D   // DEFINE:
    
    

    bool***  weightdefined; // 3D   // DEFINE:
    bool***  biasdefined;   // 3D   // DEFINE:
    
    


    bool**   dimensiondefined;  // 2D   // DEFINE:
    
    bool***  bundle_inputdefined;   // 3D   // DEFINE:
    bool***  bundle_outputdefined;  // 3D   // DEFINE:
    
    double   energy_fitv_best;  // DEFINE:
    
    int      nelements;               // # of elements (distinct from LAMMPS atom types since multiple atom types can be mapped to one element)
    int      nelementsp;              // nelements+1
    
    char**   elements;                // names of elements // 2D
    char**   elementsp;               // names of elements with "all" appended as the last "element" // 2D
    
    double   cutmax;                  // max radial distance for neighbor lists    

    DualCArray<real_t> mass; // mass of each element 1D      
    DualCArray<int> map;     // mapping from atom types to elements // 1D


    int*     fingerprintcount;        // static variable used in initialization // 1D
    int*     fingerprintlength;       // # of input neurons defined by fingerprints of each element. // 1D
    int*     fingerprintperelement;   // # of fingerprints for each element // 1D
    int*     stateequationperelement; // 1D // DEFINE:
    int*     stateequationcount;      // 1D // DEFINE:
    
    bool     doscreen; // screening is calculated if any defined fingerprint uses it
    bool     allscreen;
    bool     dospin;
    
    int      res; // Resolution of function tables for cubic interpolation.
    
    double*  screening_min; // 1D   // DEFINE:
    double*  screening_max; // 1D   // DEFINE:
    
    int      memguess;  // DEFINE:
    
    bool*    freezebeta; // 1D
    
    int      speciesnumberr;    // DEFINE:
    int      speciesnumberv;    // DEFINE:
    
    bool     freeenergy;    // DEFINE:
    
    double   hbar;  // DEFINE:

    struct NNarchitecture
    {
        
        int layers;
        
        // Dimension size (net.layers)
        int* dimensions;        // vector of length layers with entries for neurons per layer // 1D
        

        int* activations;        // unused // 1D
        
        int maxlayer;        // longest layer (for memory allocation)
        
        int sumlayers;
        
        int* startI;    // 1D
        
        bool bundle;
        
        // Sized by num_layers
        int* bundles;   // 1D
        

        int** bundleinputsize;  // 2D // DEFINE: 
        
        int** bundleoutputsize; // 2D // DEFINE:
        
        bool** identitybundle;  // 2D // DEFINE:
        

        int*** bundleinput;     // 3D // DEFINE:
        int*** bundleoutput;    // 3D // DEFINE:

        // BundleW/B sizes ( , , net(i).bundleinputsize[j][k] *  net(i).bundleoutputsize[j][k])
        double*** bundleW;      // 3D // DEFINE: 600 - 800 doubles (product of each adjoining layers, (num_elements,num_layers,num_bundles)
        double*** bundleB;      // 3D // DEFINE:
        


        bool*** freezeW;        // 3D // DEFINE:
        bool*** freezeB;        // 3D // DEFINE:

        // int layers;
        // int maxlayer;
        // int sumlayers;
        
        // DualCArray<int> dimensions;  // 1D (num_layers)
        // DualCArray<int> activations; // 1D

        // DualCArray<int> startI;     // 1D

        // bool bundle;
        // DualCArray<int> bundles;   // 1D (num_layers)
        // DualCArray<int> bundleinputsize;    // 2D
        // DualCArray<int> bundleoutputsize;   // 2D
        // DualCArray<bool> identitybundle;    // 2D
        
        // DualCArray<int> bundleinput;    // 3D
        // DualCArray<int> bundleoutput;   // 3D
        // DualCArray<real_t> bundleW; // 3D
        // DualCArray<real_t> bundleB; // 3D
        
        // DualCArray<bool> freezeW;   // 3D
        // DualCArray<bool> freezeB;   // 3D

    };
    
    // CArray<NNarchitecture> net;    // array of networks, 1 for each element.


    CArray<NNarchitecture> net; // array of networks, 1 for each element.

    // DEFINE:   Note: mostly ragged
    struct Simulation
    {
        bool forces; // DEFINE:
        bool spins; // DEFINE:
        int* id;    // 1D // DEFINE:
        double** x; // 2D // DEFINE: position (num_atoms, x, y, z)
        double** f; // 2D // DEFINE: forces  (num_atoms, x, y, z)
        double** s; // 2D // DEFINE: spins   (num_atoms, x, y, z)
        double box[3][3]; // 2D // DEFINE:
        double origin[3]; // 1D // DEFINE:
        double** features;  // 2D // DEFINE: (num_atoms, length_per_elem) (Ragged) lenght per elem is the feature length, ragged
        double** dfx;   // 2D,3D // DEFINE: Derivative of features in X direction wrt x coordinates of all neighbors (num_atoms, (num_neighbors+1)*feature_length) (note, make 3D) +1 is self term
        double** dfy;   // 2D // DEFINE:
        double** dfz;   // 2D // DEFINE:
        double** dsx;   // 2D // DEFINE:
        double** dsy;   // 2D // DEFINE:
        double** dsz;   // 2D // DEFINE:
        int* ilist;     // 1D // DEFINE:
        int* numneigh;  // 1D // DEFINE: Neighbor list size
        int** firstneigh; // 2D // DEFINE: Neighbor ID

        int* type;  // 1D // DEFINE:
        int inum;   // DEFINE:
        int gnum;   // DEFINE:
        double energy;  // DEFINE:
        double energy_weight;   // scalar? DEFINE:
        double force_weight;    // DEFINE:
        int startI;
        char* filename;
        int timestep;
        bool spinspirals;
        double spinvec[3]; // 1D // DEFINE:
        double spinaxis[3]; // 1D // DEFINE:
        double** force; // 2D // DEFINE:
        double** fm;    // 2D// DEFINE:
        double state_e;
        double* state_ea;   // 1D // DEFINE:
        double* total_ea;   // 1D // DEFINE:
        double time;
        int uniquespecies;
        int* speciesmap;    // 1D // DEFINE:
        int speciesoffset; // DEFINE:
        int atomoffset; // DEFINE:
        int* speciescount; // 1D // DEFINE:
        double temp; // DEFINE:
    };
    Simulation* sims; // Array of simulation structs, 

    // read potential file:
    void read_file(char* filename);
    
    void read_atom_types(std::vector<std::string> line, char* filename, int linenum);
    
    void read_fpe(std::vector<std::string> line, std::vector<std::string> line1, char* filename, int linenum);          // fingerprints per element. Count total fingerprints defined for each 1st element in element combinations
    
    void read_fingerprints(std::vector<std::string> line, std::vector<std::string> line1, char*, int linenum);
    
    void read_fingerprint_constants(std::vector<std::string>, std::vector<std::string>, char* filename, int);
    
    void read_network_layers(std::vector<std::string>, std::vector<std::string>, char*, int);         // include input and output layer (hidden layers + 2)
    
    void read_layer_size(std::vector<std::string>, std::vector<std::string>, char*, int);
    
    void read_weight(std::vector<std::string>, std::vector<std::string>, FILE*, char*, int*);           // weights should be formatted as properly shaped matrices
    
    void read_bias(std::vector<std::string>, std::vector<std::string>, FILE*, char*, int*);           // biases should be formatted as properly shaped vectors
    
    void read_activation_functions(std::vector<std::string>, std::vector<std::string>, char*, int);
    
    void read_screening(std::vector<std::string>, std::vector<std::string>, char*, int);

    void read_mass(const std::vector<std::string>&, const std::vector<std::string>&, const char*, int);
    
    void read_eospe(std::vector<std::string>, std::vector<std::string>, char*, int);
    
    void read_eos(std::vector<std::string>, std::vector<std::string>, char*, int);
    
    void read_eos_constants(std::vector<std::string>, std::vector<std::string>, char*, int);
    
    void read_bundles(std::vector<std::string>, std::vector<std::string>, char*, int);
    
    void read_bundle_input(std::vector<std::string>, std::vector<std::string>, char*, int);
    
    void read_bundle_output(std::vector<std::string>, std::vector<std::string>, char*, int);
    
    void read_bundle_id(std::vector<std::string>, std::vector<std::string>, char*, int);
    
    void read_parameters(std::vector<std::string>, std::vector<std::string>, FILE*, char*, int*, char*);
    
    bool check_potential();

    // process_data
    void read_dump_files();
    void create_neighbor_lists();
    void screen(double*, double*, double*, double*, double*, double*, double*, bool*, int, int, double*, double*, double*, int*, int);
    void cull_neighbor_list(double*, double*, double*, int*, int*, int*, int, int, double);
    void screen_neighbor_list(double*, double*, double*, int*, int*, int*, int, int, bool*, double*, double*, double*, double*, double*, double*, double*);
    void compute_fingerprints();
    void separate_validation();
    void normalize_data();
    int count_unique_species(DualCArray<int>, int);

    // handle network
    void create_random_weights(int, int, int, int, int);
    void create_random_biases(int, int, int, int);
    void create_identity_wb(int, int, int, int, int);
    void jacobian_convolution(double*, double*, DualCArray<int>, int, int, CArray<NNarchitecture>);
    void forward_pass(double*, DualCArray<int>, int, CArray<NNarchitecture>);
    void get_per_atom_energy(double**, int*, int, CArray<NNarchitecture>);
    void propagateforward(double*, double**, int, int, int, double*, double*, double*, double*, int*, int, CArray<NNarchitecture>);
    void propagateforwardspin(double*, double**, double**, double**, int, int, int, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*,
        int*, int, CArray<NNarchitecture>);
    void flatten_beta(CArray<NNarchitecture>, double*);   // fill beta vector from net structure
    void unflatten_beta(CArray<NNarchitecture>, double*);   // fill net structure from beta vector
    void copy_network(CArray<NNarchitecture>, CArray<NNarchitecture>);
    void normalize_net(CArray<NNarchitecture>);
    void unnormalize_net(CArray<NNarchitecture>);

    // run fitting
    void levenburg_marquardt_ch();
    void conjugate_gradient();
    void levenburg_marquardt_linesearch();
    void bfgs();

    // utility and misc
    void allocate(const std::vector<std::string>&);     // called after reading element list, but before reading the rest of the potential
    bool check_parameters();
    void update_stack_size();
    int factorial(int);
    void write_potential_file(bool, char*, int, double);
    void errorf(const std::string&, int, const char*);
    void errorf(char*, int, const char*);
    void errorf(const char*);
    std::vector<std::string> tokenmaker(std::string, std::string);
    int count_words(char*);
    int count_words(char*, char*);
    void qrsolve(double*, int, int, double*, double*);
    void chsolve(double*, int, double*, double*);

    // debugs:
    void write_debug_level1(double*, double*);
    void write_debug_level2(double*, double*);
    void write_debug_level3(double*, double*, double*, double*);
    void write_debug_level4(double*, double*);
    void write_debug_level5(double*, double*);
    void write_debug_level5_spin(double*, double*);
    void write_debug_level6(double*, double*);

    // create styles
    RANN::Fingerprint* create_fingerprint(const char*); // DEFINE: 
    RANN::Activation* create_activation(const char*);   // DEFINE:
    RANN::State* create_state(const char*); // DEFINE: 

protected:

    RANN::Activation**** activation;    // DEFINE: 
    RANN::Fingerprint*** fingerprints;  // DEFINE: 
    RANN::State***       state;         // DEFINE: 

};
} // namespace LAMMPS_NS
#endif /* CALIBRATION_H_ */
