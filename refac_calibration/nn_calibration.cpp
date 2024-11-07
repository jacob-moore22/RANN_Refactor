/*
 * main.cpp
 *
 *  Top-level file for calibration code. Reads command line input, and creates new instance of Calibration class.
 *  Calibration class handles everything else.
 */

#include "pair_spin_rann.h"


#include <chrono>  // For std::chrono

using namespace LAMMPS_NS;

// read command line input.
int main(int argc, char** argv)
{


    Kokkos::initialize();
    {

        DualCArray<int> arr(10);
        FOR_ALL(i, 0, 10, {
            arr(i) = i;
        });

        // Get the starting timepoint
        auto start = std::chrono::steady_clock::now();

        char str[MAXLINE];
        if (argc != 3 || strcmp(argv[1], "-in") != 0) {
            sprintf(str, "syntax: nn_calibration -in \"input_file.rann\"\n");
            std::cout << str;
        }
        else{
            PairRANN* cal = new PairRANN(argv[2]);

            auto start_setup = std::chrono::steady_clock::now();
            cal->setup();
            
            auto now = std::chrono::steady_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::seconds>(now - start_setup);
            
            std::cout << "******************************************" << std::endl;
            std::cout << "*****  Time taken for setup: " << duration_ms.count() << " s" << std::endl;
            std::cout << "******************************************" << std::endl;

            
            // auto start_run = std::chrono::steady_clock::now();
            // cal->run();

            // now = std::chrono::steady_clock::now();
            // duration_ms = std::chrono::duration_cast<std::chrono::seconds>(now - start_run);
            
            // std::cout << "******************************************" << std::endl;
            // std::cout << "*****  Time taken for run: " << duration_ms.count() << " s" << std::endl;
            // std::cout << "******************************************" << std::endl;


            // auto start_finish = std::chrono::steady_clock::now();

            // cal->finish();
            
            // now = std::chrono::steady_clock::now();
            // duration_ms = std::chrono::duration_cast<std::chrono::seconds>(now - start_finish);
            
            // std::cout << "******************************************" << std::endl;
            // std::cout << "*****  Time taken for finish: " << duration_ms.count() << " s" << std::endl;
            // std::cout << "******************************************" << std::endl;


            now = std::chrono::steady_clock::now();
            duration_ms = std::chrono::duration_cast<std::chrono::seconds>(now - start);
            std::cout << "******************************************" << std::endl;
            std::cout << "*****  Total time taken: " << duration_ms.count() << " s" << std::endl;
            std::cout << "******************************************" << std::endl;


            delete cal;
        }

    }

    Kokkos::finalize();

    std::cout << "**** End of main **** " << std::endl;

}
