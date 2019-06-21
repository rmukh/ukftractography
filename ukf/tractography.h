/**
 * \file tractography.h
 * \brief Contains the Class Tractography, which contains the functions that deal with the
 * actual tracing of the fibers for each model
*/
#ifndef TRACTOGRAPHY_H_
#define TRACTOGRAPHY_H_

#include <string>
#include <vector>
#include "ukffiber.h"
#include "seed.h"
#include "ukf_types.h"
#include "ukf_exports.h"

// ITK includes
#include "itkSingleValuedCostFunction.h"
#include "itkLBFGSBOptimizer.h"

// Spherical ridgelets
#include "SOLVERS.h"
#include "SPH_RIDG.h"
#include "UtilMath.h"

class NrrdData;
class vtkPolyData;
class Tractography;

// Internal constants
const ukfPrecisionType SIGMA_MASK = 0.5;
const ukfPrecisionType P0 = 0.01;
const ukfPrecisionType MIN_RADIUS = 0.87;
const ukfPrecisionType FULL_BRAIN_MEAN_SIGNAL_MIN = 0.18;
const ukfPrecisionType D_ISO = 0.003; // Diffusion coefficient of free water

struct UKFSettings
{
  bool record_fa;
  bool record_nmse;
  bool record_trace;
  bool record_state;
  bool record_cov;
  bool record_free_water;
  bool record_tensors;
  bool record_Vic;
  bool record_kappa;
  bool record_Viso;
  bool record_weights;
  bool transform_position;
  bool store_glyphs;
  bool branches_only;
  ukfPrecisionType fa_min;
  ukfPrecisionType mean_signal_min;
  ukfPrecisionType seeding_threshold;
  int num_tensors;
  int seeds_per_voxel;
  ukfPrecisionType min_branching_angle;
  ukfPrecisionType max_branching_angle;
  bool is_full_model;
  bool free_water;
  bool noddi;
  bool diffusion_propagator;
  ukfPrecisionType rtop_min;
  bool record_rtop;
  ukfPrecisionType max_nmse;
  int maxUKFIterations;
  ukfPrecisionType stepLength;
  ukfPrecisionType recordLength;
  ukfPrecisionType maxHalfFiberLength;
  std::vector<int> labels;

  ukfPrecisionType Qm;
  ukfPrecisionType Ql;
  ukfPrecisionType Qw;
  ukfPrecisionType Qt;
  ukfPrecisionType Qwiso;
  ukfPrecisionType Qkappa;
  ukfPrecisionType Qvic;
  ukfPrecisionType Rs;

  ukfPrecisionType p0;
  ukfPrecisionType sigma_signal;
  ukfPrecisionType sigma_mask;
  ukfPrecisionType min_radius;
  ukfPrecisionType full_brain_mean_signal_min;
  size_t num_threads;

  /*
  *  TODO refactor
  */
  bool writeAsciiTracts;
  bool writeUncompressedTracts;

  // TODO MRMLID support?
  std::string output_file;
  std::string output_file_with_second_tensor;
  std::string dwiFile;
  std::string seedsFile;
  std::string maskFile;
};

/**
 * \class Tractography
 * \brief This class performs the tractography and saves each step.
*/
class UKFBASELIB_EXPORTS Tractography
{

  friend class vtkSlicerInteractiveUKFLogic;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** Defines a type to specify the model */
  enum model_type
  {
    _1T,
    _1T_FW,
    _1T_FULL,
    _1T_FW_FULL,
    _2T,
    _2T_FW,
    _2T_FULL,
    _2T_FW_FULL,
    _3T,
    _3T_FULL,
    _3T_BIEXP_RIDG
  };

  /** Constructor, is called from main.cc where all parameters are defined. */
  Tractography(UKFSettings settings);

  /** Destructor */
  ~Tractography();

  /**
   * Load the files that contain the DWI signal, the seeds and a mask
   * defining the volume of the brain.
  */
  bool LoadFiles(const std::string &data_file, const std::string &seed_file, const std::string &mask_file,
                 const bool normalized_DWI_data, const bool output_normalized_DWI_data);

  /**
   * Directly set the data volume pointers
  */

  bool SetData(void *data, void *mask, void *seed, bool normalizedDWIData);

  /**
   * Directly set the seed locations
  */

  void SetSeeds(stdVec_t seeds)
  {
    _ext_seeds = seeds;
  }

  /**
   * Creates the seeds and initilizes them by finding the tensor directions,
   * eigenvalues and Euler Angles. This also sets the initial state and
   * covariance.
  */
  void Init(std::vector<SeedPointInfo> &seed_infos);

  /** \breif Performs the tractography
      \return true if files written successfully, else false
  */
  bool Run();

  /**
   * Follows one seed point for the 3 Tensor case
  */
  void Follow3T(const int thread_id, const SeedPointInfo &seed, UKFFiber &fiber);

  void Follow3T_Other(const int thread_id, const size_t seed_index, const SeedPointInfo &seed, UKFFiber &fiber,
                      bool is_branching, std::vector<SeedPointInfo> &branching_seeds,
                      std::vector<BranchingSeedAffiliation> &branching_seed_affiliation);

  /**
   * Follows one seed point for the 2 Tensor case
  */
  void Follow2T(const int thread_id, const size_t seed_index, const SeedPointInfo &seed, UKFFiber &fiber,
                bool is_branching, std::vector<SeedPointInfo> &branching_seeds,
                std::vector<BranchingSeedAffiliation> &branching_seed_affiliation);

  /**
   * Follows one seed point for the 1 Tensor case
  */
  void Follow1T(const int thread_id, const SeedPointInfo &seed, UKFFiber &fiber);

  /*
  * Update filter model type
  */
  void UpdateFilterModelType();

  /**
  * Helper functions for library use to set internal data
  */
  void SetWriteBinary(bool wb) { this->_writeBinary = wb; }
  void SetWriteCompressed(bool wb) { this->_writeCompressed = wb; }
  void SetOutputPolyData(vtkPolyData *pd) { this->_outputPolyData = pd; }

  void SetDebug(bool v) { this->debug = v; }

private:
  /**
   * Calculate six tensor coefficients by solving B * d = log(s), where d are
   * tensor coefficients, B is gradient weighting, s is signal.
  */
  void UnpackTensor(const ukfVectorType &b, const stdVec_t &u, stdEigVec_t &s,
                    stdEigVec_t &ret);

  /**
  * Creates necessary variable for noddi
  */
  void createProtocol(const ukfVectorType &b, ukfVectorType &gradientStrength,
                      ukfVectorType &pulseSeparation);

  /** One step along the fiber for the 3-tensor case. */
  void Step3T(const int thread_id, vec3_t &x, vec3_t &m1, vec3_t &l1, vec3_t &m2, vec3_t &l2, vec3_t &m3, vec3_t &l3,
              ukfPrecisionType &fa, ukfPrecisionType &fa2, ukfPrecisionType &fa3, State &state, ukfMatrixType &covariance, ukfPrecisionType &dNormMSE,
              ukfPrecisionType &trace, ukfPrecisionType &trace2);

  /** One step for ridgelets bi-exp case */
  void Step3T(const int thread_id, vec3_t &x, vec3_t &m1, vec3_t &m2, vec3_t &m3, State &state, ukfMatrixType &covariance,
              ukfPrecisionType &dNormMSE, ukfPrecisionType &fa, ukfPrecisionType &fa2, ukfPrecisionType &fa3, ukfPrecisionType &trace, ukfPrecisionType &trace2);

  /** One step along the fiber for the 2-tensor case. */
  void Step2T(const int thread_id, vec3_t &x, vec3_t &m1, vec3_t &l1, vec3_t &m2, vec3_t &l2, ukfPrecisionType &fa, ukfPrecisionType &fa2,
              State &state, ukfMatrixType &covariance, ukfPrecisionType &dNormMSE, ukfPrecisionType &trace, ukfPrecisionType &trace2);

  /** One step along the fiber for the 1-tensor case. */
  void Step1T(const int thread_id, vec3_t &x, ukfPrecisionType &fa, State &state, ukfMatrixType &covariance, ukfPrecisionType &dNormMSE,
              ukfPrecisionType &trace);

  /**
   * Swaps the first tensor with the i-th tensor in state and covariance matrix for the 3 Tensor case.
   * This is used when the main direction of the tractography 'switches' tensor.
  */
  void SwapState3T(State &state, ukfMatrixType &covariance, int i);
  void SwapState3T(stdVecState &state, ukfMatrixType &covariance, int i);

  void SwapState3T_BiExp(State &state, ukfMatrixType &covariance, int i);
  void SwapState3T_BiExp(stdVecState &state, ukfMatrixType &covariance, int i);

  /**
   * Swap the tensors in the state and covariance matrix for the 2-tensor case. This is used when the
   * principal direction of the minor tensor has more weight than the one of the current tensor.
  */
  void SwapState2T(State &state, ukfMatrixType &covariance);

  /**
   * Saves one point along the fiber so that everything can be written to a
   * file at the end.
  */
  void Record(const vec3_t &x, const ukfPrecisionType fa, const ukfPrecisionType fa2, const ukfPrecisionType fa3,
              const State &state, const ukfMatrixType p, UKFFiber &fiber,
              const ukfPrecisionType dNormMSE, const ukfPrecisionType trace, const ukfPrecisionType trace2);

  /**  Reserving fiber array memory so as to avoid resizing at every step*/
  void FiberReserve(UKFFiber &fiber, int fiber_size);

  /** Compute the Return to Origin probability in the case of the diffusionPropagator model, using the state parameters */
  void computeRTOPfromState(stdVecState &state, ukfPrecisionType &rtop, ukfPrecisionType &rtop1, ukfPrecisionType &rtop2, ukfPrecisionType &rtop3);

  /** Compute the Return to Origin probability in the case of the diffusionPropagator model, using the interpolated signal */
  void computeRTOPfromSignal(ukfPrecisionType &rtopSignal, ukfVectorType &signal);

  /** Print the State on the standard output in the case of the diffusion propagator model */
  void PrintState(State &state);

  /** Non Linear Least Square Optimization of input parameters */
  void NonLinearLeastSquareOptimization(State &state, ukfVectorType &signal, FilterModel *model);

  /** Make the seed point in the other direction */
  void InverseStateDiffusionPropagator(stdVecState &reference, stdVecState &inverted);

  /** Loop the UKF with 5 iterations, used by step 2T */
  void LoopUKF(const int thread_id, State &state, ukfMatrixType &covariance, ukfVectorType &signal, State &state_new, ukfMatrixType &covariance_new, ukfPrecisionType &dNormMSE);

  /** Convert State to Matrix */
  void StateToMatrix(State &state, ukfMatrixType &matrix);

  /** Convert Matrix to State */
  void MatrixToState(ukfMatrixType &matrix, State &state);

  /** Vector of Pointers to Unscented Kalaman Filters. One for each thread. */
  std::vector<UnscentedKalmanFilter *> _ukf;

  /** Output file for tracts generated with first tensor */
  const std::string _output_file;
  /** Output file for tracts generated with second tensor */
  const std::string _output_file_with_second_tensor;

  /** Pointer to generic diffusion data */
  NrrdData *_signal_data;

  /** Switch for attaching the FA value to the fiber at each point of the tractography */
  const bool _record_fa;
  /**
   * Switch for attaching the normalized mean squared error of the reconstructed signal to the real signal
   * to the fiber at each point of the tractography
  */
  const bool _record_nmse;
  /** Switch for attaching the trace to the fiber at each point of the tractography */
  const bool _record_trace;
  /** Switch for attaching the state to the fiber at each point of the tractography */
  const bool _record_state;
  /** Switch for attaching the covariance to the fiber at each point of the tractography */
  const bool _record_cov;
  /** Switch for attaching the free water percentage to the fiber at each point of the tractography */
  const bool _record_free_water;
  // Noddi Model parameters
  /** Switch for attaching the Vic to the fiber at each point of the tractography */
  const bool _record_Vic;
  /** Switch for attaching the kappa to the fiber at each point of the tractography */
  const bool _record_kappa;
  /** Switch for attaching the Viso to the fiber at each point of the tractography */
  const bool _record_Viso;
  /**
   * Switch for attaching the diffusion tensors to the fiber at each point of the tractography.
   * This is important for visualizing the fiber properties in Slicer.
  */
  const bool _record_tensors;
  /**
   * Wheather to transform the points back to RAS-space before writing the VTK or not.
  */
  const bool _record_weights;
  const bool _transform_position;
  /** Attach the glyphs to the VTK file */
  const bool _store_glyphs;
  /** To output branches only */
  bool _branches_only;

  // Internal parameters
  bool _is_branching;
  const ukfPrecisionType _p0;
  const ukfPrecisionType _sigma_signal;
  const ukfPrecisionType _sigma_mask;
  const ukfPrecisionType _min_radius;

  ukfVectorType weights_on_tensors;

  /** Maximal number of points in the tract */
  const int _max_length;
  bool _full_brain;
  bool _noddi;
  ukfVectorType _gradientStrength, _pulseSeparation;
  /** Diffustion propagator parameters **/
  bool _diffusion_propagator;
  ukfPrecisionType _rtop_min;
  bool _record_rtop;
  const ukfPrecisionType _max_nmse;
  int _maxUKFIterations;
  /** Index of the weight in the state for the free water cases */
  int _nPosFreeWater;

  // Parameters for the tractography
  ukfPrecisionType _fa_min;
  ukfPrecisionType _mean_signal_min;
  ukfPrecisionType _seeding_threshold;
  int _num_tensors;
  int _seeds_per_voxel;
  ukfPrecisionType _cos_theta_min;
  ukfPrecisionType _cos_theta_max;
  bool _is_full_model;
  bool _free_water;
  ukfPrecisionType _stepLength;
  int _steps_per_record;
  std::vector<int> _labels;
  stdVec_t _ext_seeds;

  ukfPrecisionType Qm;
  ukfPrecisionType Ql;
  ukfPrecisionType Qw;
  ukfPrecisionType Qt;
  ukfPrecisionType Qwiso;
  ukfPrecisionType Qkappa;
  ukfPrecisionType Qvic;
  ukfPrecisionType Rs;

  bool _writeBinary;
  bool _writeCompressed;
  // Threading control
  const int _num_threads;

  vtkPolyData *_outputPolyData;

  // TODO smartpointer
  model_type _filter_model_type;
  FilterModel *_model;
  signalMaskType signal_mask;

  bool debug;

  // Spherical Ridgelets bases
  ukfMatrixType ARidg;
  ukfMatrixType QRidg;

  // Sphereical Ridgelets helper matricies/vectors
  ukfMatrixType fcs;
  ukfMatrixType nu;
  vector<vector<unsigned int>> conn;

  // Spherical Ridgelets helper variables
  ukfPrecisionType sph_rho;
  unsigned int sph_J;
  ukfPrecisionType fista_lambda;
  unsigned int lvl;
  ukfPrecisionType max_odf_thresh;
};

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
  void SetSignalValues(ukfVectorType &signal)
  {
    _signal.resize(signal.size());
    for (unsigned int it = 0; it < signal.size(); ++it)
    {
      _signal(it) = signal[it];
    }
  }

  // Set the pointer to the model
  void SetModel(FilterModel *model)
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
  FilterModel *_model;
  ukfVectorType _fixed_params;
};
} // end namespace itk

#endif // TRACTOGRAPHY_H_
