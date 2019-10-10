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
#ifdef _OPENMP
#include <omp.h>
#endif

// Spherical ridgelets
#include "SOLVERS.h"
#include "SPH_RIDG.h"
#include "UtilMath.h"

//L-BFGS-B solver
#include "lfbgsb.hpp"

#include "itkSimpleFastMutexLock.h"

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
  bool record_uncertainties;
  bool transform_position;
  bool store_glyphs;
  bool branches_only;
  ukfPrecisionType fa_min;
  ukfPrecisionType mean_signal_min;
  ukfPrecisionType seeding_threshold;
  int num_tensors;
  ukfPrecisionType seeds_per_voxel;
  ukfPrecisionType min_branching_angle;
  ukfPrecisionType max_branching_angle;
  bool is_full_model;
  bool free_water;
  bool noddi;
  bool diffusion_propagator;
  ukfPrecisionType rtop1_min_stop;
  bool record_rtop;
  ukfPrecisionType max_nmse;
  int maxUKFIterations;
  ukfPrecisionType max_odf_threshold;
  ukfPrecisionType fw_thresh;
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
  int num_threads;

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
  std::string csfFile;
  std::string wmFile;
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
  bool LoadFiles(const std::string &data_file, const std::string &seed_file, const std::string &mask_file, const std::string &csf_file,
                 const std::string &wm_file, const bool normalized_DWI_data, const bool output_normalized_DWI_data);

  /**
   * Directly set the data volume pointers
  */

  bool SetData(void *data, void *mask, void *csf, void *wm, void *seed, bool normalizedDWIData);

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
  void Follow3T(const int thread_id, const SeedPointInfo &seed, UKFFiber &fiber, unsigned char &is_discarded);

  void Follow3T(const int thread_id, const SeedPointInfo &fiberStartSeed,
                UKFFiber &fiber, UKFFiber &fiber1, UKFFiber &fiber2, UKFFiber &fiber3);

  void Follow3T(const int thread_id, const size_t seed_index, const SeedPointInfo &seed, UKFFiber &fiber,
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
  void UnpackTensor(const ukfVectorType &b, const stdVec_t &u, stdEigVec_t &s, stdEigVec_t &ret);

  /**
  * Creates necessary variable for noddi
  */
  void createProtocol(const ukfVectorType &b, ukfVectorType &gradientStrength, ukfVectorType &pulseSeparation);

  /** One step along the fiber for the 3-tensor case. */
  void Step3T(const int thread_id, vec3_t &x, vec3_t &m1, vec3_t &l1, vec3_t &m2, vec3_t &l2, vec3_t &m3, vec3_t &l3, ukfPrecisionType &fa,
              ukfPrecisionType &fa2, ukfPrecisionType &fa3, State &state, ukfMatrixType &covariance, ukfPrecisionType &dNormMSE,
              ukfPrecisionType &trace, ukfPrecisionType &trace2);

  /** One step for ridgelets bi-exp case */
  void Step3T(const int thread_id, vec3_t &x, vec3_t &m1, vec3_t &m2, vec3_t &m3, State &state, ukfMatrixType &covariance, ukfPrecisionType &dNormMSE,
              ukfPrecisionType &rtop1, ukfPrecisionType &rtop2, ukfPrecisionType &rtop3, ukfPrecisionType &Fm1, ukfPrecisionType &lmd1,
              ukfPrecisionType &Fm2, ukfPrecisionType &lmd2, ukfPrecisionType &Fm3, ukfPrecisionType &lmd3, ukfPrecisionType &varW1,
              ukfPrecisionType &varW2, ukfPrecisionType &varW3, ukfPrecisionType &varWiso, ukfPrecisionType &rtopModel,
              ukfPrecisionType &rtopSignal);

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

  // BiExp version only
  void Record(const vec3_t &x, const ukfPrecisionType fa, const ukfPrecisionType fa2, const ukfPrecisionType fa3, const ukfPrecisionType Fm1,
              const ukfPrecisionType lmd1, const ukfPrecisionType Fm2, const ukfPrecisionType lmd2, const ukfPrecisionType Fm3,
              const ukfPrecisionType lmd3, const ukfPrecisionType varW1, const ukfPrecisionType varW2, const ukfPrecisionType varW3,
              const ukfPrecisionType varWiso, const State &state, const ukfMatrixType p, UKFFiber &fiber, const ukfPrecisionType dNormMSE,
              const ukfPrecisionType trace, const ukfPrecisionType trace2);

  // Other models version
  void Record(const vec3_t &x, const ukfPrecisionType fa, const ukfPrecisionType fa2, const ukfPrecisionType fa3,
              const State &state, const ukfMatrixType p, UKFFiber &fiber, const ukfPrecisionType dNormMSE,
              const ukfPrecisionType trace, const ukfPrecisionType trace2);

  void RecordWeightTrack(const vec3_t &x, UKFFiber &fiber, ukfPrecisionType d1, ukfPrecisionType d2, ukfPrecisionType d3);

  /**  Reserving fiber array memory so as to avoid resizing at every step*/
  void FiberReserve(UKFFiber &fiber, int fiber_size);

  void FiberReserveWeightTrack(UKFFiber &fiber, int fiber_size);

  /** Compute the Return to Origin probability in the case of the diffusionPropagator model, using the state parameters */
  void computeRTOPfromState(State &state, ukfPrecisionType &rtop, ukfPrecisionType &rtop1, ukfPrecisionType &rtop2, ukfPrecisionType &rtop3);

  /** Compute the Return to Origin probability in the case of the diffusionPropagator model, using the interpolated signal */
  void computeRTOPfromSignal(ukfPrecisionType &rtopSignal, const ukfVectorType &signal);

  /** Compute uncertanties characteristics */
  void computeUncertaintiesCharacteristics(ukfMatrixType &cov, ukfPrecisionType &Fm1, ukfPrecisionType &lmd1, ukfPrecisionType &Fm2, ukfPrecisionType &lmd2,
                                           ukfPrecisionType &Fm3, ukfPrecisionType &lmd3, ukfPrecisionType &varW1, ukfPrecisionType &varW2,
                                           ukfPrecisionType &varW3, ukfPrecisionType &varWiso);

  /** Print the State on the standard output in the case of the diffusion propagator model */
  void PrintState(State &state);

  /** Non Linear Least Square Optimization of input parameters */
  void NonLinearLeastSquareOptimization(State &state, const ukfVectorType &signal);

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
  const bool _record_uncertainties;
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
  bool _full_brain;   // is full brain?
  bool _is_seeds;     // check if seeds file provided?
  bool _csf_provided; // check if CSF file provided?
  bool _wm_provided;  // check if WM file provided?
  bool _noddi;
  ukfVectorType _gradientStrength, _pulseSeparation;
  /** Diffustion propagator parameters **/
  bool _diffusion_propagator;
  ukfPrecisionType _rtop1_min_stop;
  bool _record_rtop;
  const ukfPrecisionType _max_nmse;
  int _maxUKFIterations;
  ukfPrecisionType _fw_thresh;
  /** Index of the weight in the state for the free water cases */
  int _nPosFreeWater;

  // Parameters for the tractography
  ukfPrecisionType _fa_min;
  ukfPrecisionType _mean_signal_min;
  ukfPrecisionType _seeding_threshold;
  int _num_tensors;
  ukfPrecisionType _seeds_per_voxel;
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
  SignalModel *_model;
  signalMaskType signal_mask;

  bool debug;

  // Spherical Ridgelets bases
  ukfMatrixType ARidg;
  ukfMatrixType QRidg;
  ukfMatrixType QRidgSignal;

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

#endif // TRACTOGRAPHY_H_
