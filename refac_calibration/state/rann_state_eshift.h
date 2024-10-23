
#ifndef LMP_RANN_STATE_ESHIFT_H
#define LMP_RANN_STATE_ESHIFT_H

#include "rann_stateequation.h"
#include "pair_spin_rann.h"

namespace LAMMPS_NS
{
namespace RANN
{
class State_eshift : public State
{
public:
    State_eshift(PairRANN* _pair) : State(_pair)
    {
        eshift = 0;
        n_body_type = 1;
        style            = "eshift";
        atomtypes        = new int[n_body_type];
        empty            = true;
        fullydefined     = false;
        _pair->allscreen = false;
    };

    ~State_eshift()
    {
        delete[] atomtypes;
    };

    /////////////////////////////////////////////////////////////////////////////
    ///
    /// \fn eos_function
    ///
    /// \brief <insert brief description>
    ///
    /// <Insert longer more detailed description which
    /// can span multiple lines if needed>
    ///
    /// \param <function parameter description>
    ///
    /// \return <return type and definition description if not void>
    ///
    /////////////////////////////////////////////////////////////////////////////
    void eos_function(double* ep, double** force, int ii, int nn,
        double* xn, double* yn, double* zn, int* tn, int jnum, int* jl)
    {
        ep[0] += eshift;
        return;
    };

    /////////////////////////////////////////////////////////////////////////////
    ///
    /// \fn parse_values
    ///
    /// \brief <insert brief description>
    ///
    /// <Insert longer more detailed description which
    /// can span multiple lines if needed>
    ///
    /// \param <function parameter description>
    ///
    /// \return <return type and definition description if not void>
    ///
    /////////////////////////////////////////////////////////////////////////////
    bool parse_values(std::string constant, std::vector<std::string> line1)
    {
        int nwords = line1.size();
        if (constant.compare("eshift") == 0) {
            eshift = strtod(line1[0].c_str(), nullptr);
        }
        else{ pair->errorf(FLERR, "Undefined value for eshift equation of state"); }
        // allow undefined delta, (default = 0)
        return true;
    };

    /////////////////////////////////////////////////////////////////////////////
    ///
    /// \fn write_values
    ///
    /// \brief <insert brief description>
    ///
    /// <Insert longer more detailed description which
    /// can span multiple lines if needed>
    ///
    /// \param <function parameter description>
    ///
    /// \return <return type and definition description if not void>
    ///
    /////////////////////////////////////////////////////////////////////////////
    void write_values(FILE* fid)
    {
        int i;
        fprintf(fid, "stateequationconstants:");
        fprintf(fid, "%s", pair->elementsp[atomtypes[0]]);
        for (i = 1; i < n_body_type; i++) {
            fprintf(fid, "_%s", pair->elementsp[atomtypes[i]]);
        }
        fprintf(fid, ":%s_%d:eshift:\n", style, id);
        fprintf(fid, "%f\n", eshift);
    }

    /////////////////////////////////////////////////////////////////////////////
    ///
    /// \fn init
    ///
    /// \brief <insert brief description>
    ///
    /// <Insert longer more detailed description which
    /// can span multiple lines if needed>
    ///
    /// \param <function parameter description>
    ///
    /// \return <return type and definition description if not void>
    ///
    /////////////////////////////////////////////////////////////////////////////
    void init(int* i, int _id)
    {
        empty = false;
        for (int j = 0; j < n_body_type; j++) { atomtypes[j] = i[j]; }
        id = _id;
    };
    double eshift;
};
}    // namespace RANN
}    // namespace LAMMPS_NS

#endif /* LMP_RANN_STATE_ROSE_H_ */
