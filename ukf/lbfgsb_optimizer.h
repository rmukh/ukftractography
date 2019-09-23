/**
 * \file lbfgsb_optimizer.h
 * \brief The implementation of the L-BFGS-B ITK-based optimization algotithm
*/

#ifndef LBFGSB_OPTIMIZER_H_
#define LBFGSB_OPTIMIZER_H_

#include <iostream>

#include "ukf_types.h"
#include "filter_model.h"
// ITK includes
#include "itkSingleValuedCostFunction.h"
#include "itkLBFGSBOptimizer.h"

class FilterModel;

namespace itk
{
class DiffusionPropagatorCostFunction : public SingleValuedCostFunction
{
public:
    /** Standard class typedefs. */
    typedef DiffusionPropagatorCostFunction Self;
    typedef SingleValuedCostFunction Superclass;
    typedef SmartPointer<Self> Pointer;
    typedef SmartPointer<const Self> ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(DiffusionPropagatorCostFunction, SingleValuedCostFunction);

    // We are estimating the parameters of the state for the
    // diffusion propagator model. Number of rows in the state
    // vector
    unsigned int GetNumberOfParameters() const { return _NumberOfParameters; }
    void SetNumberOfParameters(unsigned int NumberOfParameters) { _NumberOfParameters = NumberOfParameters; }

    // Number of fixed (not for optimization) parameters
    unsigned int GetNumberOfFixed() const { return _NumberOfFixed; }
    void SetNumberOfFixed(unsigned int NumberOfFixed) { _NumberOfFixed = NumberOfFixed; }

    // The number of gradient directions in which the signal is estimated
    unsigned int GetNumberOfValues() const { return _NumberOfValues; }
    void SetNumberOfValues(unsigned int NumberOfValues) { _NumberOfValues = NumberOfValues; }

    // Set the signal values = reference signal we are trying to fit
    void SetSignalValues(const ukfVectorType &signal)
    {
        _signal.resize(signal.size());
        for (unsigned int it = 0; it < signal.size(); ++it)
        {
            _signal(it) = signal[it];
        }
    }

    // Set the pointer to the model
    void SetModel(const FilterModel *model)
    {
        _model = model;
    }

    // Set fixed parameters
    void SetFixed(ukfVectorType &fixed)
    {
        _fixed_params.resize(fixed.size());
        for (unsigned int it = 0; it < fixed.size(); ++it)
        {
            _fixed_params(it) = fixed[it];
        }
    }

    void SetPhase(unsigned int phase)
    {
        _phase = phase;
    }

    /** Compute the relative error between the signal estimate and the signal data */
    void computeError(ukfMatrixType &signal_estimate, const ukfVectorType &signal, ukfPrecisionType &err) const
    {
        assert(signal_estimate.rows() == signal.size());

        ukfPrecisionType sum = 0.0;
        ukfPrecisionType norm_sq_signal = 0.0;
        unsigned int N = signal.size() / 2;

        for (unsigned int i = 0; i < N; ++i)
        {
            ukfPrecisionType diff = signal[i] - signal_estimate(i, 0);
            sum += diff * diff;
            norm_sq_signal += signal[i] * signal[i];
        }

        err = sum / (norm_sq_signal);
    }

    MeasureType GetValue(const ParametersType &parameters) const;
    void GetDerivative(const ParametersType &parameters, DerivativeType &derivative) const;

protected:
    DiffusionPropagatorCostFunction() {}
    ~DiffusionPropagatorCostFunction() {}

private:
    DiffusionPropagatorCostFunction(const Self &); //purposely not implemented
    void operator=(const Self &);                  //purposely not implemented

    unsigned int _NumberOfParameters;
    unsigned int _NumberOfFixed;
    unsigned int _NumberOfValues;
    unsigned int _phase;
    ukfVectorType _signal;
    const FilterModel *_model;
    ukfVectorType _fixed_params;
};
} // end namespace itk

/**
 * \class UnscentedKalmanFilter
 * \brief The C++ implementation of the unscented Kalman Filter
*/
class LBFGSBSolver
{
    typedef itk::LBFGSBOptimizer OptimizerType;
    typedef itk::DiffusionPropagatorCostFunction CostType;

public:
    /**
   * \brief Constructor
    */
    LBFGSBSolver(FilterModel *filter_model);

    void Optimize(State &state_inp, const ukfVectorType &signal_inp);

private:
    /** Pointer to the filter model */
    const FilterModel *const m_FilterModel;
};

#endif // LBFGSB_OPTIMIZER_H_
