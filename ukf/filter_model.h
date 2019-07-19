/**
 * \file filter_model.h
 * \brief Contains all of the model reconstructions
 * \todo It would be a minor change to constrain the non free water cases as well. There would be a small additional time cost. Needs to be decided.
*/

#ifndef FILTER_MODEL_H_
#define FILTER_MODEL_H_

#include <cassert>
#include <vector>
#include <iostream>

#include "linalg.h"
#include "ISignalData.h"
#include "unscented_kalman_filter.h"
#include "lbfgsb_optimizer.h"

/**
 * \struct FilterModel
 * \brief Interface for implementation of a signal model
 * \todo IMO rename to SignalModel, FilterModel is confusing.
 * A generic class that defines the transition function and the observation
 * model to be used in the Kalman filter.
*/
class FilterModel
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** Constructor */
  FilterModel(const int local_state_dim, const ukfPrecisionType rs, const ukfVectorType &weights_on_tensors, bool constrained)
      : _state_dim(local_state_dim),
        _rs(rs), _signal_dim(0), _signal_data(NULL), weights_on_tensors_(weights_on_tensors),
        _constrained(constrained), _ridgelets_used(false)
  {
    _Q.resize(_state_dim, _state_dim);
    _Q.setConstant(ukfZero); // necessary because otherwise there is memory left overs in the matrix

    if (constrained)
    {
      std::cout << "Using constrained filter\n";
    }
    else
    {
      std::cout << "Using unconstrained filter\n";
    }
  }

  FilterModel(const int local_state_dim, const ukfPrecisionType rs, const ukfVectorType &weights_on_tensors, bool constrained, bool ridgelets_used)
      : _state_dim(local_state_dim),
        _rs(rs), _signal_dim(0), _signal_data(NULL), weights_on_tensors_(weights_on_tensors),
        _constrained(constrained), _ridgelets_used(ridgelets_used)
  {
    _Q.resize(_state_dim, _state_dim);
    _Q.setConstant(ukfZero); // necessary because otherwise there is memory left overs in the matrix

    if (constrained)
    {
      std::cout << "Using constrained filter\n";
    }
    else
    {
      std::cout << "Using unconstrained filter\n";
    }
    if (ridgelets_used)
    {
      std::cout << "Using Spherical Ridgelets\n";
    }
  }

  /** Destructor */
  virtual ~FilterModel()
  {
  }

  /** state transition function */
  virtual void F(ukfMatrixType & /* X */) const = 0;
  virtual void F(ukfMatrixType & /* X */, ukfVectorType /* s */, const ukfMatrixType & /* covMatrix */) const = 0;

  /** observation, i.e. signal reconstruction */
  virtual void H(const ukfMatrixType &X, ukfMatrixType &Y) const = 0;

  // Functions that convert the state into a tensor representation that can be
  // used for tractography (meaning the main direction and all the eigenvalues/
  // lambdas).

  /** Extracts principal diffusion direction and eigen values from the state for the 1T cases */
  virtual void State2Tensor1T(const State &, vec3_t &, vec3_t &)
  {
    assert(!"Not implemented");
  }

  /** Extracts principal diffusion direction and eigen values from the state for the 2T cases */
  virtual void State2Tensor2T(const State &, const vec3_t &, vec3_t &,
                              vec3_t &, vec3_t &, vec3_t &)
  {
    assert(!"Not implemented");
  }

  /** Extracts principal diffusion direction and eigen values from the state for the 3T cases */
  virtual void State2Tensor3T(const State &, const vec3_t &, vec3_t &,
                              vec3_t &, vec3_t &, vec3_t &, vec3_t &,
                              vec3_t &)
  {
    assert(!"Not implemented");
  }

  virtual void State2Tensor3T(const State &, const vec3_t &, vec3_t &, vec3_t &, vec3_t &)
  {
    assert(!"Not implemented");
  }

  /** Returns the dimension of the state */
  int state_dim() const
  {
    return _state_dim;
  }

  /** Set the dimension of the signal, and resize the corresponding matrices correspondigly */
  void set_signal_dim(const int dim)
  {
    _signal_dim = dim;

    _R.resize(_signal_dim, _signal_dim);
    _R.setConstant(ukfZero); // necessary because otherwise there is memory leftovers in the matrix
    for (int i = 0; i < _signal_dim; ++i)
    {
      _R(i, i) = _rs;
    }
  }

  /** Returns the dimension of the signal */
  int signal_dim() const
  {
    return _signal_dim;
  }

  /** The noise in the state transfer function used by the UKF. */
  const ukfMatrixType &Q() const
  {
    return _Q;
  }

  /** The noise in the signal reconstruction used by the UKF. */
  const ukfMatrixType &R() const
  {
    return _R;
  }

  // The next two functions are only used for the constrained case

  /** The inequality constraint matrix for the constrained UKF */
  const ukfMatrixType &D() const
  {
    return _D;
  }

  /** The inequality constraint right hand side for the constrained UKF */
  const ukfVectorType &d() const
  {
    return _d;
  }

  /** The equality constraint matrix for the constrained UKF */
  const ukfMatrixType &E() const
  {
    return _E;
  }

  /** The equality constraint right hand side for the constrained UKF */
  const ukfVectorType &e() const
  {
    return _e;
  }

  /** Are we running the contrained version of the filter? */
  bool isConstrained() const
  {
    return _constrained;
  }

  /** Set the pointer to the diffusion signal data */
  void set_signal_data(ISignalData *signal_data)
  {
    _signal_data = signal_data;
  }

  /** Are we using Spherical Ridgelets **/
  bool isRidgelets() const
  {
    return _ridgelets_used;
  }

protected:
  /** Checks if d is smaller than a small negative threshold. If yes an error is returned. Otherwise d is rounded to ukfZero
    */
  ukfPrecisionType CheckZero(const ukfPrecisionType &d) const;

  /** The dimension of the state */
  const int _state_dim;

  /** The constant signal noise for each signal component */
  ukfPrecisionType _rs;

  /** The dimension of the signal */
  int _signal_dim;

  /** Pointer to the diffusion signal data */
  ISignalData *_signal_data;

  /** Process noise */
  ukfMatrixType _Q;

  /** Signal reconstruction noise */
  ukfMatrixType _R;

  /** Inequality constraint matrix, only used for constrained UKF */
  ukfMatrixType _D;

  /** Inequality right hand side, only used for constrained UKF */
  ukfVectorType _d;

  /** Equality constraint matrix, only used for constrained UKF */
  ukfMatrixType _E;

  /** Equality right hand side, only used for constrained UKF */
  ukfVectorType _e;

  /** The weights of each tensor */
  const ukfVectorType weights_on_tensors_;

  /** Are we using the constrained filter */
  bool _constrained;

  /** Are we using spherical ridgelets**/
  bool _ridgelets_used;
};

inline mat33_t SetIdentityScaled(double diff_fw)
{
  mat33_t tmp;
  tmp.setIdentity();
  tmp *= diff_fw;
  return tmp;
}

const double Pi(std::atan(static_cast<double>(1.0)) * 4);

inline ukfPrecisionType BhattacharyyaCoeff(ukfPrecisionType x_sr, ukfPrecisionType x_pred, ukfPrecisionType cov)
{
  return std::exp(-0.25 * (std::pow((x_sr - x_pred), 2) / cov));
}

inline ukfPrecisionType BhattacharyyaCoeff(vec3_t &x_sr, vec3_t &x_pred, const ukfMatrixType &cov)
{
  vec3_t diff = x_sr - x_pred;
  return std::exp(-0.125 * diff.transpose() * cov.inverse() * diff);
}

inline ukfPrecisionType BhattacharyyaCoeff(vec3_t &x_sr, vec3_t &x_pred, const ukfMatrixType &cov, const ukfMatrixType &cov2)
{
  vec3_t diff = x_sr - x_pred;
  ukfMatrixType cov_total = (cov + cov2) / 2.0;
  return std::exp(-( (0.125 * diff.transpose() * cov_total.inverse() * diff) + 0.5 * std::log(cov_total.determinant() / std::sqrt(cov.determinant() * cov2.determinant()))));
}

inline ukfPrecisionType AngularSimilarity(vec3_t &x_sr, vec3_t &x_pred)
{
  ukfPrecisionType dot = x_sr.dot(x_pred);
  ukfPrecisionType den_a = x_sr.norm();
  ukfPrecisionType den_b = x_pred.norm();

  if (den_a == 0.0 || den_b == 0.0)
  {
    std::cout << "cosine similarity is not defined whenever one or both input vectors are zero-vectors." << std::endl;
    throw;
  }

  return 1.0 - (std::acos(std::min(std::max(dot / (den_a * den_b), -1.0), 1.0)) / Pi);
}

/* Common Function used for NODDI */
extern void createProtocol(const ukfVectorType &_b_values,
                           ukfVectorType &_gradientStrength, ukfVectorType &_pulseSeparation);

#endif // FILTER_MODEL_H_
