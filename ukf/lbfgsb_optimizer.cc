/**
 * \file lbfgsb_optimizer.cc
 * \brief implementation of lbfgsb_optimizer.h
*/

#include "lbfgsb_optimizer.h"
#include "filter_model.h"

itk::SingleValuedCostFunction::MeasureType itk::DiffusionPropagatorCostFunction::GetValue(const ParametersType &parameters) const
{
    MeasureType residual = 0.0;

    //assert(this->GetNumberOfParameters() == 16);

    // Convert the parameter to the ukfMtarixType
    ukfMatrixType localState(this->GetNumberOfParameters() + this->GetNumberOfFixed(), 1);
    if (this->_phase == 1)
    {
        localState(0, 0) = _fixed_params(0);
        localState(1, 0) = _fixed_params(1);
        localState(2, 0) = _fixed_params(2);
        localState(7, 0) = _fixed_params(3);
        localState(8, 0) = _fixed_params(4);
        localState(9, 0) = _fixed_params(5);
        localState(14, 0) = _fixed_params(6);
        localState(15, 0) = _fixed_params(7);
        localState(16, 0) = _fixed_params(8);
        localState(21, 0) = _fixed_params(9);
        localState(22, 0) = _fixed_params(10);
        localState(23, 0) = _fixed_params(11);

        localState(3, 0) = parameters[0];
        localState(4, 0) = parameters[1];
        localState(5, 0) = parameters[2];
        localState(6, 0) = parameters[3];
        localState(10, 0) = parameters[4];
        localState(11, 0) = parameters[5];
        localState(12, 0) = parameters[6];
        localState(13, 0) = parameters[7];
        localState(17, 0) = parameters[8];
        localState(18, 0) = parameters[9];
        localState(19, 0) = parameters[10];
        localState(20, 0) = parameters[11];
        localState(24, 0) = parameters[12];
    }
    else if (this->_phase == 2)
    {
        localState(0, 0) = _fixed_params(0);
        localState(1, 0) = _fixed_params(1);
        localState(2, 0) = _fixed_params(2);
        localState(3, 0) = _fixed_params(3);
        localState(4, 0) = _fixed_params(4);
        localState(5, 0) = _fixed_params(5);
        localState(6, 0) = _fixed_params(6);
        localState(7, 0) = _fixed_params(7);
        localState(8, 0) = _fixed_params(8);
        localState(9, 0) = _fixed_params(9);
        localState(10, 0) = _fixed_params(10);
        localState(11, 0) = _fixed_params(11);
        localState(12, 0) = _fixed_params(12);
        localState(13, 0) = _fixed_params(13);
        localState(14, 0) = _fixed_params(14);
        localState(15, 0) = _fixed_params(15);
        localState(16, 0) = _fixed_params(16);
        localState(17, 0) = _fixed_params(17);
        localState(18, 0) = _fixed_params(18);
        localState(19, 0) = _fixed_params(19);
        localState(20, 0) = _fixed_params(20);
        localState(24, 0) = _fixed_params(21);

        localState(21, 0) = parameters[0];
        localState(22, 0) = parameters[1];
        localState(23, 0) = parameters[2];
    }
    else
    {
        std::cout << "You have not specified the phase!";
        throw;
    }

    // Estimate the signal
    ukfMatrixType estimatedSignal(this->GetNumberOfValues(), 1);

    //_model->F(localState);
    _model->H(localState, estimatedSignal);

    // Compute the error between the estimated signal and the acquired one
    ukfPrecisionType err = 0.0;
    this->computeError(estimatedSignal, _signal, err);

    // Return the result
    residual = err;
    return residual;
}

void itk::DiffusionPropagatorCostFunction::GetDerivative(const ParametersType &parameters, DerivativeType &derivative) const
{
    // We use numerical derivative
    // slope = [f(x+h) - f(x-h)] / (2h)

    ParametersType p_h(this->GetNumberOfParameters());  // for f(x+h)
    ParametersType p_hh(this->GetNumberOfParameters()); // for f(x-h)

    // The size of the derivative is not set by default,
    // so we have to do it manually
    derivative.SetSize(this->GetNumberOfParameters());

    // Set parameters
    for (unsigned int it = 0; it < this->GetNumberOfParameters(); ++it)
    {
        p_h[it] = parameters[it];
        p_hh[it] = parameters[it];
    }

    // Calculate derivative for each parameter (reference to the wikipedia page: Numerical Differentiation)
    for (unsigned int it = 0; it < this->GetNumberOfParameters(); ++it)
    {
        // Optimal h is sqrt(epsilon machine) * x
        double h = std::sqrt(2.22e-16) * std::abs(parameters[it]);
        // Volatile, otherwise compiler will optimize the value for dx
        volatile double xph = parameters[it] + h;

        // For taking into account the rounding error
        double dx = xph - parameters[it];

        // Compute the slope
        p_h[it] = xph;

        //p_hh[it] = parameters[it] - h;
        derivative[it] = (this->GetValue(p_h) - this->GetValue(p_hh)) / dx;

        // Set parameters back for next iteration
        p_h[it] = parameters[it];
        p_hh[it] = parameters[it];
    }
}

LBFGSBSolver::LBFGSBSolver(FilterModel *filter_model)
    : m_FilterModel(filter_model), state_temp{}, fixed{}
{
    state_temp.resize(13);
    fixed.resize(12);
}

void LBFGSBSolver::Optimize(State &state_inp, const ukfVectorType &signal_inp)
{
    ukfVectorType signal(signal_inp.size());
    for (unsigned int i = 0; i < signal_inp.size(); ++i)
    {
        signal[i] = signal_inp[i];
    }

    State state(state_inp.size());
    for (unsigned int i = 0; i < state_inp.size(); ++i)
    {
        state[i] = state_inp[i];
    }

    // Force a const version of the m_FilterModel to be used to ensure that it is not modified.
    FilterModel const *const localConstFilterModel = m_FilterModel;

    /* Pointer to cost function */
    CostType::Pointer cost = CostType::New();

    /* Pointer to optimizer */
    OptimizerType::Pointer optimizer = OptimizerType::New();

    /* ITK parameters holder */
    CostType::ParametersType p;

    /* Bounds */
    OptimizerType::BoundValueType lowerBound(13);
    OptimizerType::BoundValueType upperBound(13);
    OptimizerType::BoundSelectionType boundSelect(13);

    lowerBound.Fill(0.0);

    // Lower bound
    // First bi-exponential parameters
    lowerBound[0] = lowerBound[1] = 1.0;
    lowerBound[2] = lowerBound[3] = 0.1;

    // Second bi-exponential
    lowerBound[4] = lowerBound[5] = 1.0;
    lowerBound[6] = lowerBound[7] = 0.1;

    // Third bi-exponential
    lowerBound[8] = lowerBound[9] = 1.0;
    lowerBound[10] = lowerBound[11] = 0.1;

    // Upper bound
    upperBound.Fill(3000.0);
    upperBound[12] = 1.0;

    boundSelect.Fill(2); // BOTHBOUNDED = 2

    // Fill in array of parameters we are not intented to optimized
    // We still need to pass this parameters to optimizer because we need to compute
    // estimated signal during optimization and it requireds full state
    fixed(0) = state(0);
    fixed(1) = state(1);
    fixed(2) = state(2);
    fixed(3) = state(7);
    fixed(4) = state(8);
    fixed(5) = state(9);
    fixed(6) = state(14);
    fixed(7) = state(15);
    fixed(8) = state(16);

    fixed(9) = state(21);
    fixed(10) = state(22);
    fixed(11) = state(23);

    // std::cout << "state before\n " << state << std::endl;

    state_temp(0) = state(3);
    state_temp(1) = state(4);
    state_temp(2) = state(5);
    state_temp(3) = state(6);
    state_temp(4) = state(10);
    state_temp(5) = state(11);
    state_temp(6) = state(12);
    state_temp(7) = state(13);
    state_temp(8) = state(17);
    state_temp(9) = state(18);
    state_temp(10) = state(19);
    state_temp(11) = state(20);

    state_temp(12) = state(24);

    cost->SetNumberOfParameters(state_temp.size());
    cost->SetNumberOfFixed(fixed.size());
    cost->SetNumberOfValues(signal.size());
    cost->SetSignalValues(signal);
    cost->SetModel(localConstFilterModel);
    cost->SetFixed(fixed);
    cost->SetPhase(1);

    optimizer->SetCostFunction(cost);

    p.SetSize(13);

    // Fill p
    for (int it = 0; it < state_temp.size(); ++it)
        p[it] = state_temp[it];
    std::cout << "before " << p << std::endl;
    optimizer->SetInitialPosition(p);
    optimizer->SetProjectedGradientTolerance(1e-12);
    optimizer->SetMaximumNumberOfIterations(500);
    optimizer->SetMaximumNumberOfEvaluations(500);
    optimizer->SetMaximumNumberOfCorrections(10);     // The number of corrections to approximate the inverse hessian matrix
    optimizer->SetCostFunctionConvergenceFactor(1e1); // Precision of the solution: 1e+12 for low accuracy; 1e+7 for moderate accuracy and 1e+1 for extremely high accuracy.
    optimizer->SetTrace(false);                        // Print debug info

    optimizer->SetBoundSelection(boundSelect);
    optimizer->SetUpperBound(upperBound);
    optimizer->SetLowerBound(lowerBound);

    // std::cout << "init position " << optimizer->GetInitialPosition() << std::endl;
    //     std::cout << "state " << state.transpose() << std::endl;
    // std::cout << "signal " << signal.transpose() << std::endl;
    // std::cout << "lowerBound " << lowerBound << std::endl;
    // std::cout << "upperBound " << upperBound << std::endl;
    // std::cout << "boundSelect " << boundSelect << std::endl;
    // std::cout << "state before " << p << std::endl;

    optimizer->StartOptimization();

    p = optimizer->GetCurrentPosition();
    std::cout << "after " << p << std::endl;
    /*
    // Fill back the state tensor to return it the callee
    state(0) = fixed(0);
    state(1) = fixed(1);
    state(2) = fixed(2);
    state(7) = fixed(3);
    state(8) = fixed(4);
    state(9) = fixed(5);
    state(14) = fixed(6);
    state(15) = fixed(7);
    state(16) = fixed(8);

    state(21) = fixed(9);
    state(22) = fixed(10);
    state(23) = fixed(11);

    state(3) = p[0];
    state(4) = p[1];
    state(5) = p[2];
    state(6) = p[3];
    state(10) = p[4];
    state(11) = p[5];
    state(12) = p[6];
    state(13) = p[7];
    state(17) = p[8];
    state(18) = p[9];
    state(19) = p[10];
    state(20) = p[11];
    state(24) = p[12];

    // Second phase of optimization (optional)
    // In this phase only w1, w2, w3 are optimizing

    // Set bounds
    OptimizerType::BoundSelectionType boundSelect2(3);
    OptimizerType::BoundValueType lowerBound2(3);
    OptimizerType::BoundValueType upperBound2(3);

    boundSelect2.Fill(2); // BOTHBOUNDED = 2
    lowerBound2.Fill(0.0);
    upperBound2.Fill(1.0);

    // Fill in array of parameters we are not intented to optimized
    // We still need to pass this parameters to optimizer because we need to compute
    // estimated signal during optimization and it requireds full state
    fixed.resize(22);
    fixed(0) = state(0);
    fixed(1) = state(1);
    fixed(2) = state(2);
    fixed(3) = state(3);
    fixed(4) = state(4);
    fixed(5) = state(5);
    fixed(6) = state(6);
    fixed(7) = state(7);
    fixed(8) = state(8);
    fixed(9) = state(9);
    fixed(10) = state(10);
    fixed(11) = state(11);
    fixed(12) = state(12);
    fixed(13) = state(13);
    fixed(14) = state(14);
    fixed(15) = state(15);
    fixed(16) = state(16);
    fixed(17) = state(17);
    fixed(18) = state(18);
    fixed(19) = state(19);
    fixed(20) = state(20);
    fixed(21) = state(24);

    // std::cout << "state before\n " << state << std::endl;

    state_temp.resize(3);
    state_temp(0) = state(21);
    state_temp(1) = state(22);
    state_temp(2) = state(23);

    cost->SetNumberOfParameters(state_temp.size());
    cost->SetNumberOfFixed(fixed.size());
    cost->SetNumberOfValues(signal.size());
    cost->SetSignalValues(signal);
    cost->SetModel(localConstFilterModel);
    cost->SetFixed(fixed);
    cost->SetPhase(2);

    optimizer->SetCostFunction(cost);
    p.SetSize(3);
    // Fill p
    for (int it = 0; it < state_temp.size(); ++it)
        p[it] = state_temp[it];

    //std::cout << "state 2" << state.transpose() << std::endl;
    //std::cout << "signal 2" << signal.transpose() << std::endl;
    //std::cout << "lowerBound 2" << lowerBound2 << std::endl;
    //std::cout << "upperBound 2" << upperBound2 << std::endl;
    //std::cout << "boundSelect 2" << boundSelect2 << std::endl;
    //std::cout << "state before 2" << p2 << std::endl;

    optimizer->SetInitialPosition(p);
    optimizer->SetProjectedGradientTolerance(1e-12);
    optimizer->SetMaximumNumberOfIterations(500);
    optimizer->SetMaximumNumberOfEvaluations(500);
    optimizer->SetMaximumNumberOfCorrections(10);     // The number of corrections to approximate the inverse hessian matrix
    optimizer->SetCostFunctionConvergenceFactor(1e1); // Precision of the solution: 1e+12 for low accuracy; 1e+7 for moderate accuracy and 1e+1 for extremely high accuracy.
    optimizer->SetTrace(false);                       // Print debug info

    optimizer->SetBoundSelection(boundSelect2);
    optimizer->SetUpperBound(upperBound2);
    optimizer->SetLowerBound(lowerBound2);
    optimizer->StartOptimization();

    p = optimizer->GetCurrentPosition();
    //std::cout << "state after 2 " << p2 << std::endl;

    // Fill back the state tensor to return it the callee
    state(0) = fixed(0);
    state(1) = fixed(1);
    state(2) = fixed(2);
    state(3) = fixed(3);
    state(4) = fixed(4);
    state(5) = fixed(5);
    state(6) = fixed(6);
    state(7) = fixed(7);
    state(8) = fixed(8);
    state(9) = fixed(9);
    state(10) = fixed(10);
    state(11) = fixed(11);
    state(12) = fixed(12);
    state(13) = fixed(13);
    state(14) = fixed(14);
    state(15) = fixed(15);
    state(16) = fixed(16);
    state(17) = fixed(17);
    state(18) = fixed(18);
    state(19) = fixed(19);
    state(20) = fixed(20);
    state(24) = fixed(21);

    state(21) = p[0];
    state(22) = p[1];
    state(23) = p[2];
*/
}
