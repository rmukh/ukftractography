/**
 * \file tractography.cc
 * \brief implementation of tractography.h
*/

#include <itkMacro.h> // needed for ITK_VERSION_MAJOR
#if ITK_VERSION_MAJOR >= 5
#include "itkMultiThreaderBase.h"
#include "itkPlatformMultiThreader.h"
#else

#include "itkMultiThreader.h"
#endif

// System includes
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

// VTK includes
#include "vtkNew.h"
#include "vtkPolyData.h"

// UKF includes
#include "tractography.h"
#include "filter_model.h"
#include "ISignalData.h"
#include "NrrdData.h"
#include "utilities.h"
#include "vtk_writer.h"
#include "thread.h"
#include "math_utilities.h"

// filters
#include "filter_Full1T.h"
#include "filter_Full1T_FW.h"
#include "filter_Full2T.h"
#include "filter_Full2T_FW.h"
#include "filter_Full3T.h"
#include "filter_NODDI1F.h"
#include "filter_NODDI2F.h"
#include "filter_Simple1T.h"
#include "filter_Simple1T_FW.h"
#include "filter_Simple2T.h"
#include "filter_Simple2T_FW.h"
#include "filter_Simple3T.h"
#include "filter_ridg.h"

// TODO implement this switch
//#include "config.h"

Tractography::Tractography(UKFSettings s) :

                                            // begin initializer list

                                            _ukf(0, NULL),
                                            _output_file(s.output_file),
                                            _output_file_with_second_tensor(s.output_file_with_second_tensor),

                                            _record_fa(s.record_fa),
                                            _record_nmse(s.record_nmse),
                                            _record_trace(s.record_trace),
                                            _record_state(s.record_state),
                                            _record_cov(s.record_cov),
                                            _record_free_water(s.record_free_water),
                                            _record_Vic(s.record_Vic),
                                            _record_kappa(s.record_kappa),
                                            _record_Viso(s.record_Viso),
                                            _record_tensors(s.record_tensors),
                                            _record_weights(s.record_weights),
                                            _transform_position(s.transform_position),
                                            _store_glyphs(s.store_glyphs),
                                            _branches_only(s.branches_only),

                                            _p0(s.p0),
                                            _sigma_signal(s.sigma_signal),
                                            _sigma_mask(s.sigma_mask),
                                            _min_radius(s.min_radius),
                                            _max_length(static_cast<int>(std::ceil(s.maxHalfFiberLength / s.stepLength))),
                                            _full_brain(false),
                                            _noddi(s.noddi),
                                            _diffusion_propagator(s.diffusion_propagator),
                                            _rtop_min(s.rtop_min),
                                            _record_rtop(s.record_rtop),
                                            _max_nmse(s.max_nmse),
                                            _maxUKFIterations(s.maxUKFIterations),
                                            _fa_min(s.fa_min),
                                            _mean_signal_min(s.mean_signal_min),
                                            _seeding_threshold(s.seeding_threshold),
                                            _num_tensors(s.num_tensors),
                                            _seeds_per_voxel(s.seeds_per_voxel),
                                            _cos_theta_min(std::cos(DegToRad(s.min_branching_angle))),
                                            _cos_theta_max(std::cos(DegToRad(s.max_branching_angle))),
                                            _is_full_model(s.is_full_model),
                                            _free_water(s.free_water),
                                            _stepLength(s.stepLength),
                                            _steps_per_record(s.recordLength / s.stepLength),
                                            _labels(s.labels),

                                            Qm(s.Qm),
                                            Ql(s.Ql),
                                            Qw(s.Qw),
                                            Qt(s.Qt),
                                            Qwiso(s.Qwiso),
                                            Qkappa(s.Qkappa),
                                            Qvic(s.Qvic),
                                            Rs(s.Rs),

                                            _writeBinary(true),
                                            _writeCompressed(true),

                                            _num_threads(s.num_threads),
                                            _outputPolyData(NULL),

                                            _filter_model_type(Tractography::_1T),
                                            _model(NULL),
                                            debug(false),
                                            sph_rho(3.125),
                                            sph_J(2),
                                            fista_lambda(0.01),
                                            lvl(4),
                                            max_odf_thresh(0.7),
                                            _lbfgsb(0, NULL)
// end initializer list
{
}

Tractography::~Tractography()
{
  if (this->_signal_data)
  {
    delete this->_signal_data;
  }
  if (this->_model)
  {
    delete this->_model; // TODO smartpointer
  }
}

void Tractography::UpdateFilterModelType()
{
  if (!this->_signal_data)
  {
    return;
  }

  if (this->_model)
  {
    delete this->_model; // TODO smartpointer
  }

  this->_filter_model_type = Tractography::_1T;
  bool simpleTensorModel = !(this->_is_full_model);
  if (_diffusion_propagator)
    simpleTensorModel = false;

  if (_noddi)
  {
    if (_num_tensors == 1)
    {
      std::cout << "Using NODDI 1-Fiber model." << std::endl;
      this->_filter_model_type = Tractography::_1T_FW; // same vtk writer can be used
    }
    else if (_num_tensors == 2)
    {
      std::cout << "Using NODDI 2-Fiber model." << std::endl;
      this->_filter_model_type = Tractography::_2T_FW; // same vtk writer can be used
    }
  }
  else if (_num_tensors == 1)
  {
    if (simpleTensorModel && !_free_water)
    {
      std::cout << "Using 1-tensor simple model." << std::endl;
      this->_filter_model_type = Tractography::_1T;
    }
    else if (simpleTensorModel && _free_water)
    {
      std::cout << "Using 1-tensor simple model with free water estimation." << std::endl;
      this->_filter_model_type = Tractography::_1T_FW;
    }
    else if (!simpleTensorModel && !_free_water)
    {
      std::cout << "Using 1-tensor full model." << std::endl;
      this->_filter_model_type = Tractography::_1T_FULL;
    }
    else if (!simpleTensorModel && _free_water)
    {
      std::cout << "Using 1-tensor full model with free water estimation." << std::endl;
      this->_filter_model_type = Tractography::_1T_FW_FULL;
    }
  }
  else if (_num_tensors == 2)
  {
    if (simpleTensorModel && !_free_water)
    {
      std::cout << "Using 2-tensor simple model." << std::endl;
      this->_filter_model_type = Tractography::_2T;
    }
    else if (simpleTensorModel && _free_water)
    {
      std::cout << "Using 2-tensor simple model with free water estimation." << std::endl;
      this->_filter_model_type = Tractography::_2T_FW;
    }
    else if (!simpleTensorModel && !_free_water)
    {
      std::cout << "Using 2-tensor full model." << std::endl;
      this->_filter_model_type = Tractography::_2T_FULL;
    }
    else if (!simpleTensorModel && _free_water)
    {
      std::cout << "Using 2-tensor full model with free water estimation." << std::endl;
      this->_filter_model_type = Tractography::_2T_FW_FULL;
    }
  }
  else if (_num_tensors == 3)
  {
    if (simpleTensorModel)
    {
      std::cout << "Using 3-tensor simple model." << std::endl;
      this->_filter_model_type = Tractography::_3T;
    }
    else if (_diffusion_propagator)
    {
      std::cout << "Using BiExponential model (3 tensors, free water, spherical ridgelets)" << std::endl;
      this->_filter_model_type = Tractography::_3T_BIEXP_RIDG;
    }
    else
    {
      std::cout << "Using 3-tensor full model." << std::endl;
      this->_filter_model_type = Tractography::_3T_FULL;
    }
  }

  if ((_cos_theta_max != ukfOne) && (_cos_theta_max <= _cos_theta_min))
  {
    std::cout << "Maximum branching angle must be greater than " << RadToDeg(_cos_theta_min) << " degrees." << std::endl;
    throw;
  }

  if (_num_tensors < 1 || _num_tensors > 3)
  {
    std::cout << "Only one, two or three tensors are supported." << std::endl;
    throw;
  }

  if (_noddi)
  {
    if (_record_fa || _record_trace || _record_free_water || _record_tensors)
    {
      std::cout << "recordFA, recordTrace, record_free_water, recordTensors parameters can only be used with tensor models\n";
      throw;
    }
  }

  // Diffusion propagator model
  if (_diffusion_propagator)
  {
    if (_noddi)
    {
      std::cout << "Noddi and Diffusion Propagator parameters are mutually exclusive. Use either one of the two" << std::endl;
      throw;
    }

    if (_record_fa || _record_trace)
    {
      std::cout << "recordFA and recordTrace cannot be used with the diffusion propagator model\n";
      throw;
    }

    if (!_free_water)
    {
      std::cout << "Since the Diffusion Propagator model is used, the free water parameter will be estimated" << std::endl;
      _free_water = true;
    }

    if (_num_tensors != 3)
    {
      std::cout << "Since the Diffusion Propagator model is used, the number of tensors is set to three (3)" << std::endl;
      _num_tensors = 3;
    }
  }

  if (_record_rtop && !_diffusion_propagator)
  {
    std::cout << "recordRTOP cannot be used with any other models than the diffusionPropagator" << std::endl;
    throw;
  }

  if (_record_weights && !_diffusion_propagator)
  {
    std::cout << "record_weights parameterscan only be used with Diffusion Propagator Biexponential model\n";
    throw;
  }

  if (Qt != 0.0 && !_diffusion_propagator)
  {
    std::cout << "Qt parameter cannot be set with any other models than the diffusionPropagator model" << std::endl;
    throw;
  }

  if (_rtop_min != 0.0 && !_diffusion_propagator)
  {
    std::cout << "minRTOP parameter cannot be set with any other models than the diffusionPropagator model" << std::endl;
    throw;
  }

  if (_max_nmse != 0.0 && !_diffusion_propagator)
  {
    std::cout << "maxNMSE parameter cannot be set with any other models than the diffusionPropagator model" << std::endl;
    throw;
  }

  if (_maxUKFIterations > 0 && !_diffusion_propagator)
  {
    std::cout << "maxUKFIterations parameter cannot be set with any other models than the diffusionPropagator model" << std::endl;
    throw;
  }

  // Double check branching.
  _is_branching = _num_tensors > 1 && _cos_theta_max < ukfOne; // The branching is enabled when the maximum branching
                                                               // angle is not 0
  std::cout << "Branching " << (_is_branching ? "enabled" : "disabled") << std::endl;
  if (!_is_branching)
  {
    _branches_only = false;
  }

  _nPosFreeWater = -1; // not used for normal case
  // for free water case used in the Record function to know where the fw is in the state
  if (_num_tensors == 1) // 1 TENSOR CASE /////////////////////////////////////
  {
    if (!_is_full_model && _free_water) // simple model with free water
    {
      _nPosFreeWater = 5;
    }
    else if (_is_full_model && _free_water) // full model with free water
    {
      _nPosFreeWater = 6;
    }
    else if (_noddi) // Viso is recorded
    {
      _nPosFreeWater = 5;
    }
  }
  else if (_num_tensors == 2) // 2 TENSOR CASE ////////////////////////////////
  {
    if (!_is_full_model && _free_water) // simple model with free water
    {
      _nPosFreeWater = 10;
    }
    else if (_is_full_model && _free_water) // full model with free water
    {
      _nPosFreeWater = 12;
    }
    else if (_noddi) // Viso is recorded
    {
      _nPosFreeWater = 10;
    }
  }
  else if (_num_tensors == 3)
  {
    if (_diffusion_propagator)
      _nPosFreeWater = 24;
  }

  // set up tensor weights
  this->weights_on_tensors.resize(_num_tensors);
  for (int i = 0; i < _num_tensors; i++)
  {
    this->weights_on_tensors[i] = (1.0 / _num_tensors);
  }

  ukfPrecisionType weight_accumu = 0;
  for (int i = 0; i < _num_tensors; i++)
  {
    weight_accumu += weights_on_tensors[i];
  }
  if (std::abs(weight_accumu - 1.0) > 0.000001)
  {
    std::cout << "The weights on different tensors must add up to 1!" << std::endl
              << std::endl;
    throw;
  }
  else
  {
    weights_on_tensors.norm(); // Normalize for all to add up to 1.
  }

  if (_diffusion_propagator)
  {
    // 0.1.1. Compute ridgelets basis...
    // but first convert gradients to ukfMatrixType
    const int s_dim = _signal_data->GetSignalDimension() * 2;
    ukfMatrixType GradientDirections(s_dim, 3);
    const stdVec_t &gradients = _signal_data->gradients();
    for (int j = 0; j < s_dim; ++j)
    {
      const vec3_t &u = gradients[j];
      GradientDirections.row(j) = u;
    }

    // Get indicies of voxels in a range
    const ukfVectorType b_vals = _signal_data->GetBValues();

    //std::cout << "Bvals \n" << b_vals << std::endl;
    int vx = 0;
    for (int i = 0; i < b_vals.size() / 2; ++i)
    {
      if (b_vals(i) > 2400)
      {
        signal_mask.conservativeResize(signal_mask.size() + 1);
        signal_mask(vx) = i;
        vx++;
      }
    }

    //Take only highest b-value gradient directions
    ukfMatrixType HighBGradDirss(signal_mask.size(), 3);
    for (int indx = 0; indx < signal_mask.size(); ++indx)
      HighBGradDirss.row(indx) = GradientDirections.row(signal_mask(indx));

    // Compute A basis
    // Spherical Ridgelets helper functions
    UtilMath<ukfPrecisionType, ukfMatrixType, ukfVectorType> m;
    SPH_RIDG<ukfPrecisionType, ukfMatrixType, ukfVectorType> ridg(sph_J, 1 / sph_rho);

    ridg.RBasis(ARidg, HighBGradDirss);
    ridg.normBasis(ARidg);

    // Compute Q basis
    m.icosahedron(nu, fcs, lvl);
    ridg.QBasis(QRidg, nu); //Build a Q basis

    // Compute connectivity
    m.FindConnectivity(conn, fcs, nu.rows());
  }

  // TODO refactor this NODDI switch
  if (this->_filter_model_type == _1T_FW && this->_noddi && this->_num_tensors == 1)
  {
    _model = new NODDI1F(Qm, Qkappa, Qvic, Rs, this->weights_on_tensors, this->_noddi);
  }
  else if (this->_filter_model_type == _2T_FW && this->_noddi && this->_num_tensors == 2)
  {
    _model = new NODDI2F(Qm, Qkappa, Qvic, Rs, this->weights_on_tensors, this->_noddi);
  }
  else if (this->_filter_model_type == _1T)
  {
    _model = new Simple1T(Qm, Ql, Rs, this->weights_on_tensors, this->_free_water);
  }
  else if (this->_filter_model_type == _1T_FW)
  {
    _model = new Simple1T_FW(Qm, Ql, Qw, Rs, this->weights_on_tensors, this->_free_water, D_ISO);
  }
  else if (this->_filter_model_type == _1T_FULL)
  {
    _model = new Full1T(Qm, Ql, Rs, this->weights_on_tensors, this->_free_water);
  }
  else if (this->_filter_model_type == _1T_FW_FULL)
  {
    _model = new Full1T(Qm, Ql, Rs, this->weights_on_tensors, this->_free_water);
  }
  else if (this->_filter_model_type == _2T)
  {
    _model = new Simple2T(Qm, Ql, Rs, this->weights_on_tensors, this->_free_water);
  }
  else if (this->_filter_model_type == _2T_FW)
  {
    _model = new Simple2T_FW(Qm, Ql, Qw, Rs, this->weights_on_tensors, this->_free_water, D_ISO);
  }
  else if (this->_filter_model_type == _2T_FULL)
  {
    _model = new Full2T(Qm, Ql, Rs, this->weights_on_tensors, this->_free_water);
  }
  else if (this->_filter_model_type == _2T_FW_FULL)
  {
    _model = new Full2T_FW(Qm, Ql, Qw, Rs, this->weights_on_tensors, this->_free_water, D_ISO);
  }
  else if (this->_filter_model_type == _3T)
  {
    _model = new Simple3T(Qm, Ql, Rs, this->weights_on_tensors, this->_free_water);
  }
  else if (this->_filter_model_type == _3T_FULL)
  {
    _model = new Full3T(Qm, Ql, Rs, this->weights_on_tensors, this->_free_water);
  }
  else if (this->_filter_model_type == _3T_BIEXP_RIDG)
  {
    _model = new Ridg_BiExp_FW(Qm, Ql, Qt, Qw, Qwiso, Rs, this->weights_on_tensors, this->_free_water,
                               D_ISO, ARidg, QRidg, fcs, nu, conn, signal_mask, fista_lambda,
                               max_odf_thresh);
  }
  else
  {
    std::cerr << "Unknown filter type!" << std::endl;
    assert(false);
  }

  _model->set_signal_data(_signal_data);
  _model->set_signal_dim(_signal_data->GetSignalDimension() * 2);
}

bool Tractography::SetData(void *data, void *mask, void *seed,
                           bool normalizedDWIData)
{
  if (!data || !mask)
  {
    std::cout << "Invalid input Nrrd pointers!" << std::endl;
    return true;
  }

  if (!seed)
  {
    _full_brain = true;
  }

  _signal_data = new NrrdData(_sigma_signal, _sigma_mask);
  _signal_data->SetData((Nrrd *)data, (Nrrd *)mask, (Nrrd *)seed, normalizedDWIData);

  return false;
}

bool Tractography::LoadFiles(const std::string &data_file,
                             const std::string &seed_file,
                             const std::string &mask_file,
                             const bool normalized_DWI_data,
                             const bool output_normalized_DWI_data)
{
  _signal_data = new NrrdData(_sigma_signal, _sigma_mask);

  if (seed_file.empty())
  {
    _full_brain = true;
  }

  if (_signal_data->LoadData(data_file, seed_file, mask_file, normalized_DWI_data, output_normalized_DWI_data))
  {
    std::cout << "ISignalData could not be loaded" << std::endl;
    delete _signal_data;
    _signal_data = NULL;
    return true;
  }
  return false;
}

void Tractography::Init(std::vector<SeedPointInfo> &seed_infos)
{
  if (!(_signal_data))
  {
    itkGenericExceptionMacro(<< "No signal data!");
  }

  int signal_dim = _signal_data->GetSignalDimension();

  stdVec_t seeds;
  if (!(_labels.size() > 0))
  {
    itkGenericExceptionMacro(<< "No label data!");
  }

  if (!_ext_seeds.empty())
  {
    seeds = _ext_seeds;
  }
  else if (!_full_brain)
  {
    _signal_data->GetSeeds(_labels, seeds);
  }
  else
  {
    // Iterate through all brain voxels and take those as seeds voxels.
    const vec3_t dim = _signal_data->dim();
    for (int x = 0; x < dim[0]; ++x)
    {
      for (int y = 0; y < dim[1]; ++y)
      {
        for (int z = 0; z < dim[2]; ++z)
        {
          vec3_t pos(x, y, z); //  = make_vec(x, y, z);
          if (_signal_data->ScalarMaskValue(pos) > 0)
          {
            seeds.push_back(pos);
          }
        }
      }
    }
  }

  if (!(seeds.size() > 0))
  {
    itkGenericExceptionMacro(<< "No matching label ROI seeds found! Please verify label selection.");
  }

  // Determinism.
  srand(0);

  // Create random offsets from the seed voxel.
  stdVec_t rand_dirs;

  if ((seeds.size() == 1 && _seeds_per_voxel == 1) || _seeds_per_voxel == 1) // if there is only one seed don't use offset so fibers can be
                                                                             // compared
  {
    rand_dirs.push_back(vec3_t(0, 0, 0) /* make_vec(0, 0, 0) */); // in the test cases.
  }
  else
  {
    for (int i = 0; i < _seeds_per_voxel; ++i)
    {
      vec3_t dir(static_cast<ukfPrecisionType>((rand() % 10001) - 5000),
                 static_cast<ukfPrecisionType>((rand() % 10001) - 5000),
                 static_cast<ukfPrecisionType>((rand() % 10001) - 5000));

      // CB: those directions are to compare against the matlab output
      // dir._[2] = 0.439598093988175;
      // dir._[1] = 0.236539281163321;
      // dir._[0] = 0.028331682419209;

      dir = dir.normalized();
      dir *= ukfHalf;

      rand_dirs.push_back(dir);
    }
  }

  // Calculate all starting points.
  stdVec_t starting_points;
  stdEigVec_t signal_values;
  ukfVectorType signal(signal_dim * 2);

  int num_less_than_zero = 0;
  int num_invalid = 0;
  int num_mean_signal_too_low = 0;

  int tmp_counter = 1;
  for (stdVec_t::const_iterator cit = seeds.begin(); cit != seeds.end(); ++cit)
  {
    for (stdVec_t::iterator jt = rand_dirs.begin(); jt != rand_dirs.end(); ++jt)
    {
      vec3_t point = *cit + *jt;

      _signal_data->Interp3Signal(point, signal); // here and in every step
      tmp_counter++;

      // DEBUG
      // std::cout << "point: " << point._[0] << " " << point._[1] << " " << point._[2] << std::endl;

      // Filter out all starting points that have negative signal values (due to
      // noise) or that otherwise have invalid signal values.
      bool keep = true;
      // We only scan half of the signal values since the second half is simply
      // a copy of the first half.
      for (int k = 0; k < signal_dim; ++k)
      {
        if (signal[k] < 0)
        {
          keep = false;
          ++num_less_than_zero;
          break;
        }

        if (std::isnan(signal[k]) || std::isinf(signal[k]))
        {
          keep = false;
          ++num_invalid;
          break;
        }

        // If we do full brain tractography we only want seed voxels where the
        // GA is bigger than 0.18.
        ukfMatrixType signal_tmp(signal_dim * 2, 1);
        signal_tmp.col(0) = signal;
        if (_full_brain && s2adc(signal_tmp) < _seeding_threshold)
        {
          keep = false;
          ++num_mean_signal_too_low;
          break;
        }
      }

      // If all the criteria is met we keep that point and the signal data.
      if (keep)
      {
        signal_values.push_back(signal);
        starting_points.push_back(point);
      }
    }
  }

  stdEigVec_t starting_params(starting_points.size());

  UnpackTensor(_signal_data->GetBValues(), _signal_data->gradients(),
               signal_values, starting_params);

  // If we work with the simple model we have to change the second and third
  // eigenvalues: l2 = l3 = (l2 + l3) / 2.
  if (!_is_full_model) // i.e. simple model
  {
    for (size_t i = 0; i < starting_params.size(); ++i)
    {
      starting_params[i][7] = starting_params[i][8] = (starting_params[i][7] + starting_params[i][8]) / 2.0;
      // two minor eigenvalues are treated equal in simplemodel
    }
  }

  std::vector<SeedPointInfo> seed_infos2;
  seed_infos2.reserve(6 * starting_points.size());
  const int num_of_threads = std::min(_num_threads, static_cast<int>(starting_points.size()));
  assert(num_of_threads > 0);

  // Create separate model for each thread
  _lbfgsb.reserve(num_of_threads); //Allocate, but do not assign
  for (int i = 0; i < num_of_threads; i++)
  {
    _lbfgsb.push_back(new LFBGSB(_model));
  }

  std::cout << "Processing " << starting_points.size() << " starting points with " << _lbfgsb.size() << " threads" << std::endl;
  {
    WorkDistribution work_distribution = GenerateWorkDistribution(num_of_threads, static_cast<int>(starting_points.size()));

#if ITK_VERSION_MAJOR >= 5
    itk::PlatformMultiThreader::Pointer threader = itk::PlatformMultiThreader::New();
    threader->SetNumberOfWorkUnits(num_of_threads);
    std::vector<std::thread> vectorOfThreads;
    vectorOfThreads.reserve(num_of_threads);
#else
    itk::MultiThreader::Pointer threader = itk::MultiThreader::New();
    threader->SetNumberOfThreads(num_of_threads);
#endif
    seed_init_thread_struct str;
    str.tractography_ = this;
    str.work_distribution = &work_distribution;

    str.seed_infos_ = &seed_infos2;
    str.starting_points_ = &starting_points;
    str.signal_values_ = &signal_values;
    str.starting_params_ = &starting_params;

    for (int i = 0; i < num_of_threads; i++)
    {
#if ITK_VERSION_MAJOR >= 5
      vectorOfThreads.push_back(std::thread(SeedInitThreadCallback, i, &str));
#else
      threader->SetMultipleMethod(i, SeedInitThreadCallback, &str);
#endif
    }
#if ITK_VERSION_MAJOR < 5
    threader->SetGlobalDefaultNumberOfThreads(num_of_threads);
#else
    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(num_of_threads);
#endif

#if ITK_VERSION_MAJOR >= 5
    for (auto &li : vectorOfThreads)
    {
      if (li.joinable())
      {
        li.join();
      }
    }
#else
    threader->MultipleMethodExecute();
#endif
  }
  seed_infos = seed_infos2;
  cout << "Final seeds vector size " << seed_infos.size() << std::endl;
}

void Tractography::ProcessStartingPointsBiExp(const int thread_id,
                                              std::vector<SeedPointInfo> &seed_infos,
                                              const vec3_t &starting_points,
                                              const ukfVectorType &signal_values,
                                              const ukfVectorType &starting_params)
{
  //const ukfVectorType &param = starting_params;

  // Filter out seeds whose FA is too low.
  ukfPrecisionType fa = l2fa(starting_params[6], starting_params[7], starting_params[8]);
  ukfPrecisionType trace = starting_params[6] + starting_params[7] + starting_params[8];
  ukfPrecisionType fa2 = -1;
  ukfPrecisionType fa3 = -1;
  ukfPrecisionType trace2 = -1;

  if (_num_tensors >= 2)
  {
    fa2 = fa;
    fa3 = fa;
    trace2 = trace;
  }

  if (_full_brain && fa <= _seeding_threshold)
  {
    return;
  }

  // Create seed info for both directions;
  SeedPointInfo info;
  stdVecState tmp_info_state;
  stdVecState tmp_info_inv_state;
  SeedPointInfo info_inv;

  info.point = starting_points;
  info.start_dir << starting_params[0], starting_params[1], starting_params[2];
  info.fa = fa;
  info.fa2 = fa2;
  info.fa3 = fa3;
  info.trace = trace;
  info.trace2 = trace2;
  info_inv.point = starting_points;
  info_inv.start_dir << -starting_params[0], -starting_params[1], -starting_params[2];
  info_inv.fa = fa;
  info_inv.fa2 = fa2;
  info_inv.fa3 = fa3;
  info_inv.trace = trace;
  info_inv.trace2 = trace2;

  if (_is_full_model)
  {
    tmp_info_state.resize(6);
    tmp_info_inv_state.resize(6);
    tmp_info_state[0] = starting_params[3];     // Theta
    tmp_info_state[1] = starting_params[4];     // Phi
    tmp_info_state[2] = starting_params[5];     // Psi
    tmp_info_state[5] = starting_params[8];     // l3
    tmp_info_inv_state[0] = starting_params[3]; // Theta
    tmp_info_inv_state[1] = starting_params[4]; // Phi
    // Switch psi angle.
    tmp_info_inv_state[2] = (starting_params[5] < ukfZero ? starting_params[5] + UKF_PI : starting_params[5] - UKF_PI);
    tmp_info_inv_state[5] = starting_params[8]; // l3
  }
  else if (_diffusion_propagator)
  {
    tmp_info_state.resize(25);
    tmp_info_inv_state.resize(25);
  }
  else // i.e. simple model
  {    // Starting direction.
    tmp_info_state.resize(5);
    tmp_info_inv_state.resize(5);
    tmp_info_state[0] = info.start_dir[0];
    tmp_info_state[1] = info.start_dir[1];
    tmp_info_state[2] = info.start_dir[2];
    tmp_info_inv_state[0] = info_inv.start_dir[0];
    tmp_info_inv_state[1] = info_inv.start_dir[1];
    tmp_info_inv_state[2] = info_inv.start_dir[2];
  }

  int signal_dim = _signal_data->GetSignalDimension();
  ukfVectorType signal(signal_dim * 2);

  if (_noddi)
  {
    ukfPrecisionType minnmse, nmse;
    State state(_model->state_dim());
    minnmse = 99999;
    _signal_data->Interp3Signal(info.point, signal); // here and in every step

    ukfPrecisionType dPar, dIso;
    int n = 5;
    dPar = 0.0000000017;
    dIso = 0.000000003;
    createProtocol(_signal_data->GetBValues(), _gradientStrength, _pulseSeparation);
    double kappas[5] = {0.5, 1, 2, 4, 8};
    double Vic[5] = {0, 0.25, 0.5, 0.75, 1};
    double Visoperm[5] = {0, 0.25, 0.5, 0.75, 1};
    for (int a = 0; a < n; ++a)
      for (int b = 0; b < n; ++b)
        for (int c = 0; c < n; ++c)
        {
          ukfVectorType Eec, Eic, Eiso, E;
          ExtraCelluarModel(dPar, Vic[b], kappas[a], _gradientStrength,
                            _pulseSeparation, _signal_data->gradients(), info.start_dir, Eec);
          IntraCelluarModel(dPar, kappas[a], _gradientStrength, _pulseSeparation,
                            _signal_data->gradients(), info.start_dir, Eic);
          IsoModel(dIso, _gradientStrength, _pulseSeparation, Eiso);

          E.resize(Eic.size());
          E = Visoperm[c] * Eiso + (1 - Visoperm[c]) * (Vic[b] * Eic + (1 - Vic[b]) * Eec);

          nmse = (E - signal).squaredNorm() / signal.squaredNorm();

          if (nmse < minnmse)
          {
            minnmse = nmse;
            ukfPrecisionType Viso = Visoperm[c];
            tmp_info_state[3] = Vic[b];        // Vic
            tmp_info_state[4] = kappas[a];     // Kappa
            tmp_info_inv_state[3] = Vic[b];    // Vic
            tmp_info_inv_state[4] = kappas[a]; // Kappa
          }
        }
    // xstd::cout <<"nmse of initialization "<< minnmse << "\n";
    if (_num_tensors > 1)
    {
      tmp_info_state.resize(10);
      tmp_info_inv_state.resize(10);
      tmp_info_state[5] = info.start_dir[0];
      tmp_info_state[6] = info.start_dir[1];
      tmp_info_state[7] = info.start_dir[2];
      tmp_info_inv_state[5] = info_inv.start_dir[0];
      tmp_info_inv_state[6] = info_inv.start_dir[1];
      tmp_info_inv_state[7] = info_inv.start_dir[2];
      tmp_info_state[8] = tmp_info_state[3];         // Vic
      tmp_info_state[9] = tmp_info_state[4];         // Kappa
      tmp_info_inv_state[8] = tmp_info_inv_state[3]; // Vic
      tmp_info_inv_state[9] = tmp_info_inv_state[4]; //kappa
    }
  }
  else if (_diffusion_propagator)
  {
    // STEP 0: Find number of branches in one voxel.

    // Compute number of branches at the seed point using spherical ridgelets
    UtilMath<ukfPrecisionType, ukfMatrixType, ukfVectorType> m;

    ukfVectorType HighBSignalValues(signal_mask.size());
    for (int indx = 0; indx < signal_mask.size(); ++indx)
      HighBSignalValues(indx) = signal_values(signal_mask(indx));

    // We can compute ridegelets coefficients
    ukfVectorType C;
    {
      SOLVERS<ukfPrecisionType, ukfMatrixType, ukfVectorType> slv(ARidg, HighBSignalValues, fista_lambda);
      slv.FISTA(C);
    }

    // Now we can compute ODF
    ukfVectorType ODF = QRidg * C;

    // Let's find Maxima of ODF and values in that direction
    ukfMatrixType exe_vol;
    ukfMatrixType dir_vol;
    ukfVectorType ODF_val_at_max(6, 1);
    unsigned n_of_dirs = 0;

    m.FindODFMaxima(exe_vol, dir_vol, ODF, conn, nu, max_odf_thresh, n_of_dirs);
    if (n_of_dirs == 0)
    {
      cout << "exe_vol " << exe_vol << endl;
      cout << "dir_vol " << dir_vol << endl;
    }

    unsigned exe_vol_size = std::min(static_cast<unsigned>(exe_vol.size()), static_cast<unsigned>(6));
    ODF_val_at_max.setZero();
    for (unsigned j = 0; j < exe_vol_size; ++j)
    {
      ODF_val_at_max(j) = ODF(exe_vol(j));
    }

    // STEP 1: Initialise the state based
    mat33_t dir_init;
    dir_init.setZero();

    ukfPrecisionType w1_init = ODF_val_at_max(0);
    dir_init.row(0) = dir_vol.row(0);

    ukfPrecisionType w2_init = 0;
    ukfPrecisionType w3_init = 0;

    //std::cout << "n_of_dirs " << n_of_dirs << std::endl;

    if (n_of_dirs == 1)
    {
      vec3_t orthogonal;
      orthogonal << -dir_vol.row(0)[1], dir_vol.row(0)[0], 0.0;
      orthogonal = orthogonal / orthogonal.norm();
      dir_init.row(1) = orthogonal;

      vec3_t orthogonal2 = dir_init.row(0).cross(orthogonal);
      orthogonal2 = orthogonal2 / orthogonal2.norm();
      dir_init.row(2) = orthogonal2;

      w1_init = 1.0;
    }
    else if (n_of_dirs > 1)
    {
      if (n_of_dirs == 2)
      {
        vec3_t v1 = dir_vol.row(0);
        vec3_t v2 = dir_vol.row(2);
        vec3_t orthogonal = v1.cross(v2);
        orthogonal = orthogonal / orthogonal.norm();

        dir_init.row(1) = dir_vol.row(2);
        dir_init.row(2) = orthogonal;

        w2_init = ODF_val_at_max(2);
        ukfPrecisionType denom = w1_init + w2_init;
        w1_init = w1_init / denom;
        w2_init = w2_init / denom;
      }
      if (n_of_dirs > 2)
      {
        dir_init.row(1) = dir_vol.row(2);
        dir_init.row(2) = dir_vol.row(4);

        w2_init = ODF_val_at_max(2);
        w3_init = ODF_val_at_max(4);
        ukfPrecisionType denom = w1_init + w2_init + w3_init;
        w1_init = w1_init / denom;
        w2_init = w2_init / denom;
        w3_init = w3_init / denom;
      }
    }

    // Diffusion directions, m1 = m2 = m3
    tmp_info_state[0] = dir_init.row(0)[0];
    tmp_info_state[1] = dir_init.row(0)[1];
    tmp_info_state[2] = dir_init.row(0)[2];

    tmp_info_state[7] = dir_init.row(1)[0];
    tmp_info_state[8] = dir_init.row(1)[1];
    tmp_info_state[9] = dir_init.row(1)[2];

    tmp_info_state[14] = dir_init.row(2)[0];
    tmp_info_state[15] = dir_init.row(2)[1];
    tmp_info_state[16] = dir_init.row(2)[2];

    // Fast diffusing component,  lambdas l11, l21 = l1 from the single tensor
    //                            lambdas l12, l21 = (l2 + l3) /2
    // from the single tensor (avg already calculated and stored in l2)
    tmp_info_state[3] = tmp_info_state[10] = tmp_info_state[17] = starting_params[6];
    tmp_info_state[4] = tmp_info_state[11] = tmp_info_state[18] = starting_params[7];

    // Slow diffusing component,  lambdas l13, l23, l33 = 0.2 * l1
    //                            lambdas l14, l24, l34 = 0.2 * (l2 + l3) /2
    tmp_info_state[5] = tmp_info_state[12] = tmp_info_state[19] = 0.7 * starting_params[6];
    tmp_info_state[6] = tmp_info_state[13] = tmp_info_state[20] = 0.7 * starting_params[7];

    tmp_info_state[21] = w1_init;
    tmp_info_state[22] = w2_init;
    tmp_info_state[23] = w3_init;

    // Free water volume fraction
    tmp_info_state[24] = 0.05; // -> as an initial value

    // STEP 2.1: Use L-BFGS-B from ITK library at the same point in space.
    // The UKF is an estimator, and we want to find the estimate with the smallest error through the iterations

    // Set the covariance value
    const int state_dim = tmp_info_state.size();
    info.covariance.resize(state_dim, state_dim);
    info_inv.covariance.resize(state_dim, state_dim);

    // make sure covariances are really empty
    info.covariance.setConstant(ukfZero);
    info_inv.covariance.setConstant(ukfZero);

    // fill the diagonal of the covariance matrix with _p0 (zeros elsewhere)
    for (int local_i = 0; local_i < state_dim; ++local_i)
    {
      info.covariance(local_i, local_i) = _p0;
      info_inv.covariance(local_i, local_i) = _p0;
    }

    // Input of the filter
    State state = ConvertVector<stdVecState, State>(tmp_info_state);
    ukfMatrixType p(info.covariance);

    // Estimate the initial state
    //InitLoopUKF(state, p, signal_values[i], dNormMSE);
    //mtx.Lock();
    NonLinearLeastSquareOptimization(thread_id, state, signal_values);
    //mtx.Unlock();

    // Output of the filter
    tmp_info_state = ConvertVector<State, stdVecState>(state);

    ukfPrecisionType rtopModel = 0.0;
    ukfPrecisionType rtop1 = 0.0;
    ukfPrecisionType rtop2 = 0.0;
    ukfPrecisionType rtop3 = 0.0;
    ukfPrecisionType rtopSignal = 0.0;

    computeRTOPfromState(state, rtopModel, rtop1, rtop2, rtop3);
    computeRTOPfromSignal(rtopSignal, signal_values);

    // These values are stored so that: rtop1 -> fa; rtop2 -> fa2; rtop3 -> fa3; rtop -> trace; rtopSignal -> trace2
    info.fa = rtop1;
    info.fa2 = rtop2;
    info.fa3 = rtop3;
    info.trace = rtopModel;
    info.trace2 = rtopSignal;

    info_inv.fa = rtop1;
    info_inv.fa2 = rtop2;
    info_inv.fa3 = rtop3;
    info_inv.trace = rtopModel;
    info_inv.trace2 = rtopSignal;

    if (rtopSignal >= _rtop_min || !_full_brain)
    {
      // Create the opposite seed
      InverseStateDiffusionPropagator(tmp_info_state, tmp_info_inv_state);

      // Update the original directions
      info.start_dir << tmp_info_state[0], tmp_info_state[1], tmp_info_state[2];
      info_inv.start_dir << -tmp_info_state[0], -tmp_info_state[1], -tmp_info_state[2];

      // Add the primary seeds to the vector
      info.state = ConvertVector<stdVecState, State>(tmp_info_state);
      info_inv.state = ConvertVector<stdVecState, State>(tmp_info_inv_state);
      //mtx.Lock();
      //seed_infos.push_back(info);
      //seed_infos.push_back(info_inv);
      //mtx.Unlock();
      /*
      if (n_of_dirs > 1)
      {
        SwapState3T_BiExp(tmp_info_state, p, 2);
        info.start_dir << tmp_info_state[0], tmp_info_state[1], tmp_info_state[2];
        info.state = ConvertVector<stdVecState, State>(tmp_info_state);
        seed_infos.push_back(info);

        // Create the seed for the opposite direction, keep the other parameters as set for the first direction
        InverseStateDiffusionPropagator(tmp_info_state, tmp_info_inv_state);

        info_inv.state = ConvertVector<stdVecState, State>(tmp_info_inv_state);
        info_inv.start_dir << tmp_info_inv_state[0], tmp_info_inv_state[1], tmp_info_inv_state[2];
        seed_infos.push_back(info_inv);

        if (n_of_dirs > 2)
        {
          SwapState3T_BiExp(tmp_info_state, p, 3);
          info.start_dir << tmp_info_state[0], tmp_info_state[1], tmp_info_state[2];
          info.state = ConvertVector<stdVecState, State>(tmp_info_state);
          seed_infos.push_back(info);

          // Create the seed for the opposite direction, keep the other parameters as set for the first direction
          InverseStateDiffusionPropagator(tmp_info_state, tmp_info_inv_state);

          info_inv.state = ConvertVector<stdVecState, State>(tmp_info_inv_state);
          info_inv.start_dir << tmp_info_inv_state[0], tmp_info_inv_state[1], tmp_info_inv_state[2];
          seed_infos.push_back(info_inv);
        }
      }
      */
    }
  }
  else
  {
    tmp_info_state[3] = starting_params[6];     // l1
    tmp_info_state[4] = starting_params[7];     // l2
    tmp_info_inv_state[3] = starting_params[6]; // l1
    tmp_info_inv_state[4] = starting_params[7]; // l2
  }

  /*
  // Duplicate/tripple states if we have several tensors.
  if (_num_tensors > 1 && !_noddi && !_diffusion_propagator)
  {
    size_t size = tmp_info_state.size();
    for (size_t j = 0; j < size; ++j)
    {
      tmp_info_state.push_back(tmp_info_state[j]);
      tmp_info_inv_state.push_back(tmp_info_state[j]);
    }
    if (_num_tensors > 2)
    {
      for (size_t j = 0; j < size; ++j)
      {
        tmp_info_state.push_back(tmp_info_state[j]);
        tmp_info_inv_state.push_back(tmp_info_state[j]);
      }
    }
  }
  if (_noddi)
  {
    tmp_info_state.push_back(Viso);
    tmp_info_inv_state.push_back(Viso); // add the weight to the state (well was sich rhymt das stiimt)
  }
  else
  {
    if (_free_water && !_diffusion_propagator)
    {
      tmp_info_state.push_back(1);
      tmp_info_inv_state.push_back(1); // add the weight to the state (well was sich rhymt das stiimt)
    }
  }

  if (!_diffusion_propagator)
  {
    const int state_dim = tmp_info_state.size();

    info.covariance.resize(state_dim, state_dim);
    info_inv.covariance.resize(state_dim, state_dim);

    // make sure covariances are really empty
    info.covariance.setConstant(ukfZero);
    info_inv.covariance.setConstant(ukfZero);
    for (int local_i = 0; local_i < state_dim; ++local_i)
    {
      info.covariance(local_i, local_i) = _p0;
      info_inv.covariance(local_i, local_i) = _p0;
    }
    info.state = ConvertVector<stdVecState, State>(tmp_info_state);
    info_inv.state = ConvertVector<stdVecState, State>(tmp_info_inv_state);
    seed_infos.push_back(info);
    seed_infos.push_back(info_inv); // NOTE that the seed in reverse direction is put directly after the seed in
                                    // original direction
  }
*/
  //if (seed_infos.size() > 1)
  //break;
}

bool Tractography::Run()
{
  assert(_signal_data); // The _signal_data is initialized in Tractography::LoadFiles(),
  // Thus Run() must be invoked after LoadFiles()
  // Initialize and prepare seeds.

  std::vector<SeedPointInfo> primary_seed_infos;
  std::vector<SeedPointInfo> branch_seed_infos;                  // The info of branching seeds
  std::vector<BranchingSeedAffiliation> branch_seed_affiliation; // Which fiber originated from the main seeds is this
                                                                 // branch attached
  if (_full_brain)
    std::cout << "No seeding file provided. Starting full brain tractography" << std::endl;
  else
    std::cout << "Seeding file provided. Seeding thresholds will be ignored" << std::endl;

  Init(primary_seed_infos);

  if (primary_seed_infos.size() < 1)
  {
    std::cerr << "No valid seed points available!" << std::endl;
    return false;
  }

  const int num_of_threads = std::min(_num_threads, static_cast<int>(primary_seed_infos.size()));
  assert(num_of_threads > 0);

  _ukf.reserve(num_of_threads); //Allocate, but do not assign
  for (int i = 0; i < num_of_threads; i++)
  {
    _ukf.push_back(new UnscentedKalmanFilter(_model)); // Create one Kalman filter for each thread
  }

  std::vector<UKFFiber> raw_primary;

  // Output directions
  // std::vector<UKFFiber> raw_primary_w1;
  // std::vector<UKFFiber> raw_primary_w2;
  // std::vector<UKFFiber> raw_primary_w3;

  {
    if (this->debug)
      std::cout << "Tracing " << primary_seed_infos.size() << " primary fibers:" << std::endl;

    raw_primary.resize(primary_seed_infos.size());
    // Output directions
    // raw_primary_w1.resize(primary_seed_infos.size() * 2);
    // raw_primary_w2.resize(primary_seed_infos.size() * 2);
    // raw_primary_w3.resize(primary_seed_infos.size() * 2);

    WorkDistribution work_distribution = GenerateWorkDistribution(num_of_threads,
                                                                  static_cast<int>(primary_seed_infos.size()));
#if ITK_VERSION_MAJOR >= 5
    itk::PlatformMultiThreader::Pointer threader = itk::PlatformMultiThreader::New();
    threader->SetNumberOfWorkUnits(num_of_threads);
    std::vector<std::thread> vectorOfThreads;
    vectorOfThreads.reserve(num_of_threads);
#else
    itk::MultiThreader::Pointer threader = itk::MultiThreader::New();
    threader->SetNumberOfThreads(num_of_threads);
#endif
    thread_struct str;
    str.tractography_ = this;
    str.seed_infos_ = &primary_seed_infos;
    str.work_distribution = &work_distribution;
    str.branching_ = _is_branching;
    str.num_tensors_ = _num_tensors;
    str.output_fiber_group_ = &raw_primary;

    // Output directions
    // str.output_fiber_group_1_ = &raw_primary_w1;
    // str.output_fiber_group_2_ = &raw_primary_w2;
    // str.output_fiber_group_3_ = &raw_primary_w3;
    str.branching_seed_info_vec = new std::vector<std::vector<SeedPointInfo>>(num_of_threads);
    str.branching_seed_affiliation_vec = new std::vector<std::vector<BranchingSeedAffiliation>>(num_of_threads);
    for (int i = 0; i < num_of_threads; i++)
    {
#if ITK_VERSION_MAJOR >= 5
      vectorOfThreads.push_back(std::thread(ThreadCallback, i, &str));
#else
      threader->SetMultipleMethod(i, ThreadCallback, &str);
#endif
    }
#if ITK_VERSION_MAJOR < 5
    threader->SetGlobalDefaultNumberOfThreads(num_of_threads);
#else
    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(num_of_threads);
#endif

#if ITK_VERSION_MAJOR >= 5
    for (auto &li : vectorOfThreads)
    {
      if (li.joinable())
      {
        li.join();
      }
    }
#else
    threader->MultipleMethodExecute();
#endif

    // Unpack the branch seeds and their affiliation
    int num_branch_seeds = 0;
    for (int i = 0; i < num_of_threads; i++)
    {
      num_branch_seeds += static_cast<int>(str.branching_seed_info_vec->at(i).size());
    }

    if (this->debug)
      std::cout << "branch_seeds size: " << num_branch_seeds << std::endl;
    branch_seed_infos.resize(num_branch_seeds);
    branch_seed_affiliation.resize(num_branch_seeds);

    int counter = 0;
    for (int i = 0; i < num_of_threads; i++)
    {
      for (size_t j = 0; j < str.branching_seed_info_vec->at(i).size(); j++)
      {
        branch_seed_infos[counter] = str.branching_seed_info_vec->at(i).at(j);
        branch_seed_affiliation[counter] = str.branching_seed_affiliation_vec->at(i).at(j);
        counter++;
      }
    }
  }

  std::vector<UKFFiber> raw_branch;
  if (_is_branching)
  {
    assert(_num_tensors == 2 || _num_tensors == 3);
    if (this->debug)
      std::cout << "Tracing " << branch_seed_infos.size() << " branches:" << std::endl;

    raw_branch.resize(branch_seed_infos.size());

    WorkDistribution work_distribution =
        GenerateWorkDistribution(num_of_threads, static_cast<int>(branch_seed_infos.size()));

    thread_struct str;
    str.tractography_ = this;
    str.work_distribution = &work_distribution;
    str.seed_infos_ = &branch_seed_infos;
    str.branching_ = false;
    str.num_tensors_ = _num_tensors;
    str.output_fiber_group_ = &raw_branch;
    str.branching_seed_info_vec = new std::vector<std::vector<SeedPointInfo>>(num_of_threads);
    str.branching_seed_affiliation_vec = new std::vector<std::vector<BranchingSeedAffiliation>>(num_of_threads);

#if ITK_VERSION_MAJOR >= 5
    itk::PlatformMultiThreader::Pointer threader = itk::PlatformMultiThreader::New();
    threader->SetNumberOfWorkUnits(num_of_threads);
    std::vector<std::thread> vectorOfThreads;
    vectorOfThreads.reserve(num_of_threads);
#else
    itk::MultiThreader::Pointer threader = itk::MultiThreader::New();
    threader->SetNumberOfThreads(num_of_threads);
#endif

    for (int i = 0; i < num_of_threads; i++)
    {
#if ITK_VERSION_MAJOR >= 5
      vectorOfThreads.push_back(std::thread(ThreadCallback, i, &str));
#else
      threader->SetMultipleMethod(i, ThreadCallback, &str);
#endif
    }
#if ITK_VERSION_MAJOR >= 5
    for (auto &li : vectorOfThreads)
    {
      if (li.joinable())
      {
        li.join();
      }
    }
#else
    threader->MultipleMethodExecute();
#endif
  }

  std::vector<UKFFiber> fibers;
  PostProcessFibers(raw_primary, raw_branch, branch_seed_affiliation, _branches_only, fibers);

  if (this->debug)
    std::cout << "fiber size after PostProcessFibers: " << fibers.size() << std::endl;

  if (fibers.size() == 0)
  {
    std::cout << "No fibers! Returning." << fibers.size() << std::endl;
    return EXIT_FAILURE;
  }

  // Write the fiber data to the output vtk file.
  VtkWriter writer(_signal_data, this->_filter_model_type, _record_tensors);
  writer.set_transform_position(_transform_position);

  int writeStatus = EXIT_SUCCESS;
  if (this->_outputPolyData != NULL)
  // TODO refactor this is bad control flow
  {
    writer.PopulateFibersAndTensors(this->_outputPolyData, fibers);
    this->_outputPolyData->Modified();
  }
  else
  {
    vtkNew<vtkPolyData> pd;
    this->SetOutputPolyData(pd.GetPointer());
    writer.PopulateFibersAndTensors(this->_outputPolyData, fibers);
    this->_outputPolyData->Modified();

    // possibly write binary VTK file.
    writer.SetWriteBinary(this->_writeBinary);
    writer.SetWriteCompressed(this->_writeCompressed);

    writeStatus = writer.Write(_output_file, _output_file_with_second_tensor,
                               fibers, _record_state, _store_glyphs, _noddi, _diffusion_propagator);

    // Output directions
    // std::string out_dir = _output_file;
    // out_dir.erase(_output_file.length() - 4);
    // writeStatus = writer.WriteWeight(out_dir + "_dir1.vtk", raw_primary_w1);
    // writeStatus = writer.WriteWeight(out_dir + "_dir2.vtk", raw_primary_w2);
    // writeStatus = writer.WriteWeight(out_dir + "_dir3.vtk", raw_primary_w3);
    // TODO refactor!
    this->SetOutputPolyData(NULL);
  }

  // Clear up the kalman filters
  for (size_t i = 0; i < _ukf.size(); i++)
  {
    delete _ukf[i];
  }
  _ukf.clear();
  return writeStatus;
}

void Tractography::computeRTOPfromSignal(ukfPrecisionType &rtopSignal, const ukfVectorType &signal)
{
  assert(signal.size() > 0);

  rtopSignal = 0.0;

  // The RTOP is the sum of the signal
  // We use signal.size()/2 because the first half of the signal is identical
  // to the second half.

  for (int i = 0; i < signal.size() / 2; ++i)
  {
    rtopSignal += signal[i];

    if (signal[i] < 0)
    {
      std::cout << "Negative signal found when computing the RTOP from the signal, value : " << signal[i] << std::endl;
    }
  }
}

void Tractography::computeRTOPfromState(State &state, ukfPrecisionType &rtop, ukfPrecisionType &rtop1, ukfPrecisionType &rtop2, ukfPrecisionType &rtop3)
{
  state[3] = std::max(state[3], 1.0);
  state[4] = std::max(state[4], 1.0);
  state[5] = std::max(state[5], 0.1);
  state[6] = std::max(state[6], 0.1);

  state[3] = std::min(state[3], 3000.0);
  state[4] = std::min(state[4], 3000.0);
  state[5] = std::min(state[5], 3000.0);
  state[6] = std::min(state[6], 3000.0);

  state[10] = std::max(state[10], 1.0);
  state[11] = std::max(state[11], 1.0);
  state[12] = std::max(state[12], 0.1);
  state[13] = std::max(state[13], 0.1);

  state[10] = std::min(state[10], 3000.0);
  state[11] = std::min(state[11], 3000.0);
  state[12] = std::min(state[12], 3000.0);
  state[13] = std::min(state[13], 3000.0);

  state[17] = std::max(state[17], 1.0);
  state[18] = std::max(state[18], 1.0);
  state[19] = std::max(state[19], 0.1);
  state[20] = std::max(state[20], 0.1);

  state[17] = std::min(state[17], 3000.0);
  state[18] = std::min(state[18], 3000.0);
  state[19] = std::min(state[19], 3000.0);
  state[20] = std::min(state[20], 3000.0);

  state[21] = std::max(state[21], 0.0);
  state[22] = std::max(state[22], 0.0);
  state[23] = std::max(state[23], 0.0);
  state[24] = std::max(state[24], 0.0);

  state[21] = std::min(state[21], 1.0);
  state[22] = std::min(state[22], 1.0);
  state[23] = std::min(state[23], 1.0);
  state[24] = std::min(state[24], 1.0);

  // Control input: state should have 25 rows
  assert(state.size() == 25);

  ukfPrecisionType l11 = state[3] * 1e-6;
  ukfPrecisionType l12 = state[4] * 1e-6;
  ukfPrecisionType l13 = state[5] * 1e-6;
  ukfPrecisionType l14 = state[6] * 1e-6;

  ukfPrecisionType l21 = state[10] * 1e-6;
  ukfPrecisionType l22 = state[11] * 1e-6;
  ukfPrecisionType l23 = state[12] * 1e-6;
  ukfPrecisionType l24 = state[13] * 1e-6;

  ukfPrecisionType l31 = state[17] * 1e-6;
  ukfPrecisionType l32 = state[18] * 1e-6;
  ukfPrecisionType l33 = state[19] * 1e-6;
  ukfPrecisionType l34 = state[20] * 1e-6;

  ukfPrecisionType w1 = state[21];
  ukfPrecisionType w2 = state[22];
  ukfPrecisionType w3 = state[23];
  ukfPrecisionType wiso = state[24];

  ukfPrecisionType det_l1 = l11 * l12;
  ukfPrecisionType det_t1 = l13 * l14;

  ukfPrecisionType det_l2 = l21 * l22;
  ukfPrecisionType det_t2 = l23 * l24;

  ukfPrecisionType det_l3 = l31 * l32;
  ukfPrecisionType det_t3 = l33 * l34;

  ukfPrecisionType det_fw = D_ISO * D_ISO * D_ISO;

  ukfPrecisionType PI_COEFF = std::pow(UKF_PI, 1.5);

  // !!! 0.7 and 0.3 tensor weights are hardcoded...
  rtop1 = PI_COEFF * w1 * (0.7 / std::sqrt(det_l1) + 0.3 / std::sqrt(det_t1));
  rtop2 = PI_COEFF * w2 * (0.7 / std::sqrt(det_l2) + 0.3 / std::sqrt(det_t2));
  rtop3 = PI_COEFF * w3 * (0.7 / std::sqrt(det_l3) + 0.3 / std::sqrt(det_t3));
  rtop = rtop1 + rtop2 + rtop3 + PI_COEFF * (wiso / std::sqrt(det_fw));
}

void Tractography::PrintState(State &state)
{
  std::cout << "State \n";
  std::cout << "\t m1: " << state[0] << " " << state[1] << " " << state[2] << std::endl;
  std::cout << "\t l11 .. l14: " << state[3] << " " << state[4] << " " << state[5] << " " << state[6] << std::endl;
  std::cout << "\t m2: " << state[7] << " " << state[8] << " " << state[9] << std::endl;
  std::cout << "\t l21 .. l24: " << state[10] << " " << state[11] << " " << state[12] << " " << state[13] << std::endl;
  std::cout << "\t m3: " << state[14] << " " << state[15] << " " << state[16] << std::endl;
  std::cout << "\t l31 .. l34: " << state[17] << " " << state[18] << " " << state[19] << " " << state[20] << std::endl;
  std::cout << "\t w1, w2, w3: " << state[21] << " " << state[22] << "" << state[23] << std::endl;
  std::cout << "\t wiso: " << state[24] << std::endl;
  std::cout << " --- " << std::endl;
}

void Tractography::NonLinearLeastSquareOptimization(const int thread_id, State &state, const ukfVectorType &signal)
{
  // Fill in array of parameters we are not intented to optimized
  // We still need to pass this parameters to optimizer because we need to compute
  // estimated signal during optimization and it requireds full state
  ukfVectorType fixed_params(12);
  fixed_params(0) = state(0);
  fixed_params(1) = state(1);
  fixed_params(2) = state(2);
  fixed_params(3) = state(7);
  fixed_params(4) = state(8);
  fixed_params(5) = state(9);
  fixed_params(6) = state(14);
  fixed_params(7) = state(15);
  fixed_params(8) = state(16);

  fixed_params(9) = state(21);
  fixed_params(10) = state(22);
  fixed_params(11) = state(23);

  ukfVectorType state_temp(13);
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

  // init solver with bounds
  
  _lbfgsb[thread_id]->_signal = signal;
  _lbfgsb[thread_id]->_fixed_params = fixed_params;
  // Run solver
  _lbfgsb[thread_id]->Solve(state_temp);
  
  //cout << "" << (*MySolver).XOpt.transpose() << endl << endl;
  //exit(0);

  //MySolver.XOpt;

  // Fill back the state tensor to return it the callee
  state(0) = fixed_params(0);
  state(1) = fixed_params(1);
  state(2) = fixed_params(2);
  state(7) = fixed_params(3);
  state(8) = fixed_params(4);
  state(9) = fixed_params(5);
  state(14) = fixed_params(6);
  state(15) = fixed_params(7);
  state(16) = fixed_params(8);

  state(21) = fixed_params(9);
  state(22) = fixed_params(10);
  state(23) = fixed_params(11);

  state(3) = _lbfgsb[thread_id]->XOpt(0);
  state(4) = _lbfgsb[thread_id]->XOpt(1);
  state(5) = _lbfgsb[thread_id]->XOpt(2);
  state(6) = _lbfgsb[thread_id]->XOpt(3);
  state(10) = _lbfgsb[thread_id]->XOpt(4);
  state(11) = _lbfgsb[thread_id]->XOpt(5);
  state(12) = _lbfgsb[thread_id]->XOpt(6);
  state(13) = _lbfgsb[thread_id]->XOpt(7);
  state(17) = _lbfgsb[thread_id]->XOpt(8);
  state(18) = _lbfgsb[thread_id]->XOpt(9);
  state(19) = _lbfgsb[thread_id]->XOpt(10);
  state(20) = _lbfgsb[thread_id]->XOpt(11);
  state(24) = _lbfgsb[thread_id]->XOpt(12);

  // Second phase of optimization (optional)
  // In this phase only w1, w2, w3 are optimizing
  /*
  CostType::Pointer cost2 = CostType::New();
  OptimizerType::Pointer optimizer2 = OptimizerType::New();

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

  cost2->SetNumberOfParameters(state_temp.size());
  cost2->SetNumberOfFixed(fixed.size());
  cost2->SetNumberOfValues(signal.size());
  cost2->SetSignalValues(signal);
  cost2->SetModel(model);
  cost2->SetFixed(fixed);
  cost2->SetPhase(2);

  optimizer2->SetCostFunction(cost2);

  CostType::ParametersType p2(cost2->GetNumberOfParameters());

  // Fill p
  for (int it = 0; it < state_temp.size(); ++it)
    p2[it] = state_temp[it];

  optimizer2->SetInitialPosition(p2);
  optimizer2->SetProjectedGradientTolerance(1e-12);
  optimizer2->SetMaximumNumberOfIterations(500);
  optimizer2->SetMaximumNumberOfEvaluations(500);
  optimizer2->SetMaximumNumberOfCorrections(10);     // The number of corrections to approximate the inverse hessian matrix
  optimizer2->SetCostFunctionConvergenceFactor(1e1); // Precision of the solution: 1e+12 for low accuracy; 1e+7 for moderate accuracy and 1e+1 for extremely high accuracy.
  optimizer2->SetTrace(false);                       // Print debug info

  // Set bounds
  OptimizerType::BoundSelectionType boundSelect2(cost2->GetNumberOfParameters());
  OptimizerType::BoundValueType upperBound2(cost2->GetNumberOfParameters());
  OptimizerType::BoundValueType lowerBound2(cost2->GetNumberOfParameters());

  boundSelect2.Fill(2); // BOTHBOUNDED = 2
  lowerBound2.Fill(0.0);
  upperBound2.Fill(1.0);

  // Lower bound
  lowerBound2[0] = lowerBound2[1] = lowerBound2[2] = 0.0;

  // Upper bound
  upperBound2[0] = upperBound2[1] = upperBound2[2] = 1.0;
  optimizer2->SetBoundSelection(boundSelect2);
  optimizer2->SetUpperBound(upperBound2);
  optimizer2->SetLowerBound(lowerBound2);
  optimizer2->StartOptimization();

  p2 = optimizer2->GetCurrentPosition();

  // Write back the state
  for (int it = 0; it < state_temp.size(); ++it)
    state_temp[it] = p2[it];

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

  state(21) = state_temp(0);
  state(22) = state_temp(1);
  state(23) = state_temp(2);
  */

  // std::cout << "state after \n"
  //           << state << std::endl;
}

void Tractography::InverseStateDiffusionPropagator(stdVecState &reference, stdVecState &inverted)
{
  assert(reference.size() == 25);
  assert(inverted.size() == 25);

  for (unsigned int it = 0; it < reference.size(); ++it)
  {
    if (it <= 2)
      inverted[it] = -reference[it];
    else
      inverted[it] = reference[it];
  }
}

void Tractography::StateToMatrix(State &state, ukfMatrixType &matrix)
{
  assert(state.size() > 0);

  matrix.resize(state.size(), 1);

  for (int it = 0; it < state.size(); ++it)
    matrix(it, 0) = state[it];
}

void Tractography::MatrixToState(ukfMatrixType &matrix, State &state)
{
  assert(matrix.cols() == 1);
  assert(matrix.rows() > 0);

  state.resize(matrix.rows());

  for (int it = 0; it < matrix.rows(); ++it)
    state[it] = matrix(it, 0);
}

// FIXME: not clear why gradientStrength and pulseSeparation are passed as arguments when
//        they are already class members.
void Tractography::createProtocol(const ukfVectorType &_b_values,
                                  ukfVectorType &l_gradientStrength, ukfVectorType &l_pulseSeparation)
{
  std::vector<double> Bunique, tmpG;
  ukfPrecisionType Bmax = 0;
  ukfPrecisionType tmp, Gmax, GAMMA;

  l_gradientStrength.resize(_b_values.size());
  l_pulseSeparation.resize(_b_values.size());

  // set maximum G = 40 mT/m
  Gmax = 0.04;
  GAMMA = 267598700;

  for (int i = 0; i < _b_values.size(); ++i)
  {
    int unique = 1;
    for (unsigned int j = 0; j < Bunique.size(); ++j)
    {
      if (_b_values[i] == Bunique[j])
      {
        unique = 0;
        break;
      }
    }
    if (unique == 1)
    {
      Bunique.push_back(_b_values[i]);
    }
    if (Bmax < _b_values[i])
    {
      Bmax = _b_values[i];
    }
  }

  tmp = cbrt(3 * Bmax * 1000000 / (2 * GAMMA * GAMMA * Gmax * Gmax));

  for (int i = 0; i < _b_values.size(); ++i)
  {
    l_pulseSeparation[i] = tmp;
  }

  for (unsigned int i = 0; i < Bunique.size(); ++i)
  {
    tmpG.push_back(std::sqrt(Bunique[i] / Bmax) * Gmax);
    // std::cout<< "\n tmpG:" << std::sqrt(Bunique[i]/Bmax) * Gmax;
  }

  for (unsigned int i = 0; i < Bunique.size(); ++i)
  {
    for (int j = 0; j < _b_values.size(); j++)
    {
      if (_b_values[j] == Bunique[i])
      {
        l_gradientStrength[j] = tmpG[i];
      }
    }
  }
}

void Tractography::UnpackTensor(const ukfVectorType &b, // b - bValues
                                const stdVec_t &u,      // u - directions
                                stdEigVec_t &s,         // s = signal values
                                stdEigVec_t &ret)       // starting params [i][j] : i - signal number; j
                                                        // - param
{
  // DEBUGGING
  // std::cout << "b's: ";
  // for (int i=0; i<b.size();++i) {
  //   std::cout << b[i] << ", ";
  // }
  assert(ret.size() == s.size());

  // Build B matrix.
  const int signal_dim = _signal_data->GetSignalDimension();

  /**
  * A special type for holding 6 elements of tensor for each signal
  *  Only used in tractography.cc
  */
  typedef Eigen::Matrix<ukfPrecisionType, Eigen::Dynamic, 6> BMatrixType;
  BMatrixType B(signal_dim * 2, 6); //HACK: Eigen::Matrix<ukfPrecisionType, DYNAMIC, 6> ??

  for (int i = 0; i < signal_dim * 2; ++i)
  {
    const vec3_t &g = u[i];
    B(i, 0) = (-b[i]) * (g[0] * g[0]);
    B(i, 1) = (-b[i]) * (2.0 * g[0] * g[1]);
    B(i, 2) = (-b[i]) * (2.0 * g[0] * g[2]);
    B(i, 3) = (-b[i]) * (g[1] * g[1]);
    B(i, 4) = (-b[i]) * (2.0 * g[1] * g[2]);
    B(i, 5) = (-b[i]) * (g[2] * g[2]);
  }

  // Use QR decomposition to find the matrix representation of the tensor at the
  // seed point of the fiber. Raplacement of the gmm function gmm::least_squares_cg(..)
  Eigen::HouseholderQR<BMatrixType> qr(B);

  // Temporary variables.
  mat33_t D;

  if (this->debug)
    std::cout << "Estimating seed tensors:" << std::endl;

  // Unpack data
  for (stdEigVec_t::size_type i = 0; i < s.size(); ++i)
  {
    // We create a temporary vector to store the signal when the log function is applied
    ukfVectorType log_s;
    log_s.resize(s[i].size());

    for (unsigned int j = 0; j < s[i].size(); ++j)
    {
      if (s[i][j] <= 0)
        s[i][j] = 10e-8;

      //s[i][j] = log(s[i][j]);
      log_s[j] = log(s[i][j]);
    }

    // The six tensor components.
    //TODO: this could be fixed size
    //ukfVectorType d = qr.solve(s[i]);
    ukfVectorType d = qr.solve(log_s);

    // symmetric diffusion tensor
    D(0, 0) = d[0];
    D(0, 1) = d[1];
    D(0, 2) = d[2];
    D(1, 0) = d[1];
    D(1, 1) = d[3];
    D(1, 2) = d[4];
    D(2, 0) = d[2];
    D(2, 1) = d[4];
    D(2, 2) = d[5];
    // Use singular value decomposition to extract the eigenvalues and the
    // rotation matrix (which gives us main direction of the tensor).
    // NOTE that svd can be used here only because D is a normal matrix

    // std::cout << "Tensor test: " << std::endl << D << std::endl;
    Eigen::JacobiSVD<ukfMatrixType> svd_decomp(D, Eigen::ComputeThinU);
    mat33_t Q = svd_decomp.matrixU();
    vec3_t sigma = svd_decomp.singularValues(); // diagonal() returns elements of a diag matrix as a vector.

    assert(sigma[0] >= sigma[1] && sigma[1] >= sigma[2]);
    if (Q.determinant() < ukfZero)
      Q = Q * (-ukfOne);

    assert(Q.determinant() > ukfZero);

    // Extract the three Euler Angles from the rotation matrix.
    ukfPrecisionType phi, psi;
    const ukfPrecisionType theta = std::acos(Q(2, 2));
    ukfPrecisionType epsilon = 1.0e-10;
    if (fabs(theta) > epsilon)
    {
      phi = atan2(Q(1, 2), Q(0, 2));
      psi = atan2(Q(2, 1), -Q(2, 0));
    }
    else
    {
      phi = atan2(-Q(0, 1), Q(1, 1));
      psi = ukfZero;
    }

    ret[i].resize(9);
    ret[i][0] = Q(0, 0);
    ret[i][1] = Q(1, 0);
    ret[i][2] = Q(2, 0);
    ret[i][3] = theta;
    ret[i][4] = phi;
    ret[i][5] = psi;
    sigma = sigma * GLOBAL_TENSOR_PACK_VALUE; // NOTICE this scaling of eigenvalues. The values are scaled back in diffusion_euler()
    ret[i][6] = sigma[0];
    ret[i][7] = sigma[1];
    ret[i][8] = sigma[2];
  }
}

void Tractography::Follow3T(const int thread_id,
                            const SeedPointInfo &fiberStartSeed,
                            UKFFiber &fiber)
{
  // For ridgelets bi-exp model only!
  int fiber_size = 100;
  int fiber_length = 0;
  assert(_model->signal_dim() == _signal_data->GetSignalDimension() * 2);

  // Unpack the fiberStartSeed information.
  vec3_t x = fiberStartSeed.point;
  State state = fiberStartSeed.state;
  ukfMatrixType p(fiberStartSeed.covariance);
  /* 
  I don't create new variables in a SeedPointInfo for rtop values and keep
  everything in fa variables just to keep consistency with other models 
  and also, to avoid making new variables which are used only in Bi-exp model.
  I think think the idea of creating more new variables just for one model
  is a pure redundancy. 
  
  To simplify code 'reading' and understanding I make a local rtop variables
  in Follow3T and Step3T functions.
  */
  ukfPrecisionType rtop1 = fiberStartSeed.fa;
  ukfPrecisionType rtop2 = fiberStartSeed.fa2;
  ukfPrecisionType rtop3 = fiberStartSeed.fa3;
  ukfPrecisionType rtopModel = fiberStartSeed.trace;
  ukfPrecisionType rtopSignal = fiberStartSeed.trace2;

  ukfPrecisionType dNormMSE = 0; // no error at the fiberStartSeed

  //std::cout << "For seed point \n " << fiberStartSeed.state << std::endl;

  //  Reserving fiber array memory so as to avoid resizing at every step
  FiberReserve(fiber, fiber_size);

  // Record start point.
  Record(x, rtop1, rtop2, rtop3, state, p, fiber, dNormMSE, rtopModel, rtopSignal);

  vec3_t m1 = fiberStartSeed.start_dir;
  vec3_t m2, m3;

  // Tract the fiber.
  ukfMatrixType signal_tmp(_model->signal_dim(), 1);
  ukfMatrixType state_tmp(_model->state_dim(), 1);

  int stepnr = 0;
  while (true)
  {
    //std::cout << "step " << stepnr << std::endl;
    ++stepnr;

    Step3T(thread_id, x, m1, m2, m3, state, p, dNormMSE, rtop1, rtop2, rtop3, rtopModel, rtopSignal);

    //cout << "w's " << state(21) << " " << state(22) << " " << state(23) << endl;
    // Check if we should abort following this fiber. We abort if we reach the
    // CSF, if FA or GA get too small, if the curvature get's too high or if
    // the fiber gets too long.
    const bool is_brain = _signal_data->ScalarMaskValue(x) > 0; //_signal_data->Interp3ScalarMask(x) > 0.1;

    state_tmp.col(0) = state;

    _model->H(state_tmp, signal_tmp);

    const ukfPrecisionType mean_signal = s2adc(signal_tmp);
    bool in_csf = (mean_signal < _mean_signal_min);

    // ukfPrecisionType rtopSignal = trace2; // rtopSignal is stored in trace2

    // in_csf = rtopSignal < _rtop_min;
    bool in_rtop1 = rtop1 < 1000;

    bool is_high_fw = state(24) > 0.7;
    bool in_rtop = rtopModel < 15000; // means 'in rtop' threshold
    bool dNormMSE_too_high = dNormMSE > _max_nmse;
    bool is_curving = curve_radius(fiber.position) < _min_radius;

    // stepnr > _max_length // Stop if the fiber is too long - Do we need this???
    if (!is_brain || in_rtop || in_rtop1 || is_high_fw || in_csf || is_curving || dNormMSE_too_high)
    {
      break;
    }

    if (fiber_length >= fiber_size)
    {
      // If fibersize is more than initally allocated size resizing further
      fiber_size += 100;
      FiberReserve(fiber, fiber_size);
    }

    if ((stepnr + 1) % _steps_per_record == 0)
    {
      fiber_length++;
      Record(x, rtop1, rtop2, rtop3, state, p, fiber, dNormMSE, rtopModel, rtopSignal);
    }
  }
  FiberReserve(fiber, fiber_length);
}

void Tractography::Follow3T(const int thread_id,
                            const SeedPointInfo &fiberStartSeed,
                            UKFFiber &fiber,
                            UKFFiber &fiber1,
                            UKFFiber &fiber2,
                            UKFFiber &fiber3)
{
  // For ridgelets bi-exp model only!
  // Debugging version of bi-exp model. Provides functionality
  // to output every (out of 3) directions from the state vector
  int fiber_size = 100;
  int fiber_weight_size = fiber_size * 2 / 3;
  int fiber_length = 0;
  assert(_model->signal_dim() == _signal_data->GetSignalDimension() * 2);

  // Unpack the fiberStartSeed information.
  vec3_t x = fiberStartSeed.point;
  State state = fiberStartSeed.state;
  ukfMatrixType p(fiberStartSeed.covariance);
  /* 
  I don't create new variables in a SeedPointInfo for rtop values and keep
  everything in fa variables just to keep consistency with other models 
  and also, to avoid making new variables which are used only in Bi-exp model.
  I think think the idea of creating more new variables just for one model
  is a pure redundancy. 
  
  To simplify code 'reading' and understanding I make a local rtop variables
  in Follow3T and Step3T functions.
  */
  ukfPrecisionType rtop1 = fiberStartSeed.fa;
  ukfPrecisionType rtop2 = fiberStartSeed.fa2;
  ukfPrecisionType rtop3 = fiberStartSeed.fa3;
  ukfPrecisionType rtopModel = fiberStartSeed.trace;
  ukfPrecisionType rtopSignal = fiberStartSeed.trace2;

  ukfPrecisionType dNormMSE = 0; // no error at the fiberStartSeed
  //std::cout << "For seed point \n " << fiberStartSeed.state << std::endl;

  // Reserving fiber array memory so as to avoid resizing at every step
  FiberReserve(fiber, fiber_size);
  FiberReserveWeightTrack(fiber1, fiber_weight_size);
  FiberReserveWeightTrack(fiber2, fiber_weight_size);
  FiberReserveWeightTrack(fiber3, fiber_weight_size);

  // Record start point
  Record(x, rtop1, rtop2, rtop3, state, p, fiber, dNormMSE, rtopModel, rtopSignal);
  RecordWeightTrack(x, fiber1, state(0), state(1), state(2));
  RecordWeightTrack(x, fiber2, state(7), state(8), state(9));
  RecordWeightTrack(x, fiber3, state(14), state(15), state(16));

  vec3_t m1 = fiberStartSeed.start_dir;
  vec3_t m2, m3;

  // Tract the fiber.
  ukfMatrixType signal_tmp(_model->signal_dim(), 1);
  ukfMatrixType state_tmp(_model->state_dim(), 1);

  int stepnr = 0;
  while (true)
  {
    // std::cout << "step " << stepnr << std::endl;
    ++stepnr;

    Step3T(thread_id, x, m1, m2, m3, state, p, dNormMSE, rtop1, rtop2, rtop3, rtopModel, rtopSignal);

    // cout << "w's " << state(21) << " " << state(22) << " " << state(23) << endl;
    // Check if we should abort following this fiber. We abort if we reach the
    // CSF, if FA or GA get too small, if the curvature get's too high or if
    // the fiber gets too long.
    const bool is_brain = _signal_data->ScalarMaskValue(x) > 0; //_signal_data->Interp3ScalarMask(x) > 0.1;

    state_tmp.col(0) = state;

    _model->H(state_tmp, signal_tmp);

    const ukfPrecisionType mean_signal = s2adc(signal_tmp);
    bool in_csf = (mean_signal < _mean_signal_min);

    // ukfPrecisionType rtopSignal = trace2; // rtopSignal is stored in trace2

    // in_csf = rtopSignal < _rtop_min;
    bool in_rtop1 = rtop1 < 4000;

    bool is_high_fw = state(24) > 0.7;
    bool in_rtop = rtopModel < 15000; // means 'in rtop' threshold
    bool dNormMSE_too_high = dNormMSE > _max_nmse;
    bool is_curving = curve_radius(fiber.position) < _min_radius;

    // stepnr > _max_length // Stop if the fiber is too long - Do we need this???
    if (!is_brain || in_rtop || in_rtop1 || is_high_fw || in_csf || is_curving || dNormMSE_too_high)
    {
      break;
    }

    if (fiber_length >= fiber_size)
    {
      // If fibersize is more than initally allocated size resizing further
      fiber_size += 100;
      FiberReserve(fiber, fiber_size);
      FiberReserveWeightTrack(fiber1, fiber_weight_size);
      FiberReserveWeightTrack(fiber2, fiber_weight_size);
      FiberReserveWeightTrack(fiber3, fiber_weight_size);
    }

    if ((stepnr + 1) % _steps_per_record == 0)
    {
      fiber_length++;
      Record(x, rtop1, rtop2, rtop3, state, p, fiber, dNormMSE, rtopModel, rtopSignal);
      if ((stepnr + 1) % 3 == 0)
      {
        RecordWeightTrack(x, fiber1, state(0), state(1), state(2));
        RecordWeightTrack(x, fiber2, state(7), state(8), state(9));
        RecordWeightTrack(x, fiber3, state(14), state(15), state(16));
      }
    }
  }
  FiberReserve(fiber, fiber_length);
  FiberReserveWeightTrack(fiber1, fiber_weight_size);
  FiberReserveWeightTrack(fiber2, fiber_weight_size);
  FiberReserveWeightTrack(fiber3, fiber_weight_size);
}

void Tractography::Follow3T(const int thread_id,
                            const size_t seed_index,
                            const SeedPointInfo &fiberStartSeed,
                            UKFFiber &fiber,
                            bool is_branching,
                            std::vector<SeedPointInfo> &branching_seeds,
                            std::vector<BranchingSeedAffiliation> &branching_seed_affiliation)
{
  int fiber_size = 100;
  int fiber_length = 0;
  assert(_model->signal_dim() == _signal_data->GetSignalDimension() * 2);

  // Unpack the fiberStartSeed information.
  vec3_t x = fiberStartSeed.point;
  State state = fiberStartSeed.state;
  ukfMatrixType p(fiberStartSeed.covariance);
  ukfPrecisionType fa = fiberStartSeed.fa;
  ukfPrecisionType fa2 = fiberStartSeed.fa2;
  ukfPrecisionType fa3 = fiberStartSeed.fa3;
  ukfPrecisionType dNormMSE = 0; // no error at the fiberStartSeed
  ukfPrecisionType trace = fiberStartSeed.trace;
  ukfPrecisionType trace2 = fiberStartSeed.trace2;

  //  Reserving fiber array memory so as to avoid resizing at every step
  FiberReserve(fiber, fiber_size);

  // Record start point.
  Record(x, fa, fa2, fa3, state, p, fiber, dNormMSE, trace, trace2);

  vec3_t m1 = fiberStartSeed.start_dir;
  vec3_t l1, m2, l2, m3, l3;

  // Tract the fiber.
  ukfMatrixType signal_tmp(_model->signal_dim(), 1);
  ukfMatrixType state_tmp(_model->state_dim(), 1);

  int stepnr = 0;
  while (true)
  {
    ++stepnr;

    Step3T(thread_id, x, m1, l1, m2, l2, m3, l3, fa, fa2, fa3, state, p, dNormMSE, trace, trace2);

    // Check if we should abort following this fiber. We abort if we reach the
    // CSF, if FA or GA get too small, if the curvature get's too high or if
    // the fiber gets too long.
    const bool is_brain = _signal_data->ScalarMaskValue(x) > 0; //_signal_data->Interp3ScalarMask(x) > 0.1;

    state_tmp.col(0) = state;
    _model->H(state_tmp, signal_tmp);

    const ukfPrecisionType mean_signal = s2adc(signal_tmp);
    const bool in_csf = (mean_signal < _mean_signal_min) ||
                        (fa < _fa_min);

    bool is_curving = curve_radius(fiber.position) < _min_radius;

    if (!is_brain || in_csf || stepnr > _max_length // Stop if the fiber is too long
        || is_curving)
    {
      break;
    }

    if (fiber_length >= fiber_size)
    {
      // If fibersize is more than initally allocated size resizing further
      fiber_size += 100;
      FiberReserve(fiber, fiber_size);
    }

    if ((stepnr + 1) % _steps_per_record == 0)
    {
      fiber_length++;
      Record(x, fa, fa2, fa3, state, p, fiber, dNormMSE, trace, trace2);
    }

    // Record branch if necessary.
    if (is_branching)
    {
      const ukfPrecisionType fa_1 = l2fa(l1[0], l1[1], l1[2]);
      const bool is_one = (l1[0] > l1[1]) && (l1[0] > l1[2]) && (fa_1 > _fa_min);
      if (is_one)
      {
        bool add_m2 = false;
        bool add_m3 = false;

        const ukfPrecisionType fa_2 = l2fa(l2[0], l2[1], l2[2]);
        const ukfPrecisionType fa_3 = l2fa(l3[0], l3[1], l3[2]);

        const bool is_two = (l2[0] > l2[1]) && (l2[0] > l2[2]) && (fa_2 > _fa_min);
        const bool is_three = (l3[0] > l3[1]) && (l3[0] > l3[2]) && (fa_3 > _fa_min);

        ukfPrecisionType dotval = m1.dot(m2);
        const bool is_branch1 = dotval < _cos_theta_min && dotval > _cos_theta_max;
        dotval = m1.dot(m3);
        const bool is_branch2 = dotval < _cos_theta_min && dotval > _cos_theta_max;
        dotval = m2.dot(m3);
        const bool is_branch3 = dotval < _cos_theta_min;

        int state_dim = _model->state_dim();
        // If there is a branch between m1 and m2.
        if (is_two && is_branch1)
        {
          // If there is a branch between m1 and m3 we have to check if we
          // branch twice or only once.
          if (is_three && is_branch2)
          {
            // If angle between m2 and m3 is big enough we have 2 branches.
            if (is_branch3)
            {
              add_m2 = true;
              add_m3 = true;
            }
            else
            {
              // Otherwise we only follow m2 or m3, and we follow the one
              // tensor where the FA is bigger.
              if (fa_2 > fa_3)
              {
                add_m2 = true;
              }
              else
              {
                add_m3 = true;
              }
            }
          }
          else
          {
            // If it's not possible for m3 to branch we are sure that m2
            // branches.
            add_m2 = true;
          }
        }
        else if (is_three && is_branch2)
        {
          // If m2 is not branching we only check if m3 can branch.
          add_m3 = true;
        }

        // If we have to tensors and the angle between them is large enough we
        // create a new seed for the branch. Since this branch is following the
        // second tensor we swap the state and covariance.
        if (add_m2)
        {
          branching_seeds.push_back(SeedPointInfo());
          SeedPointInfo &local_seed = branching_seeds[branching_seeds.size() - 1];
          branching_seed_affiliation.push_back(BranchingSeedAffiliation());
          BranchingSeedAffiliation &affiliation = branching_seed_affiliation[branching_seed_affiliation.size() - 1];

          affiliation.fiber_index_ = seed_index;
          affiliation.position_on_fiber_ = stepnr;

          local_seed.state.resize(state_dim);
          local_seed.state = state;

          local_seed.covariance.resize(state_dim, state_dim);
          local_seed.covariance = p;

          SwapState3T(local_seed.state, local_seed.covariance, 2);
          local_seed.point = x;
          local_seed.start_dir = m2;
          local_seed.fa = fa_2;
        }
        // Same for the third tensor.
        if (add_m3)
        {
          branching_seeds.push_back(SeedPointInfo());
          SeedPointInfo &local_seed = branching_seeds[branching_seeds.size() - 1];
          branching_seed_affiliation.push_back(BranchingSeedAffiliation());
          BranchingSeedAffiliation &affiliation = branching_seed_affiliation[branching_seed_affiliation.size() - 1];

          affiliation.fiber_index_ = seed_index;
          affiliation.position_on_fiber_ = stepnr;

          local_seed.state.resize(state_dim);
          local_seed.state = state;
          local_seed.covariance.resize(state_dim, state_dim);
          local_seed.covariance = p;
          SwapState3T(local_seed.state, local_seed.covariance, 3);
          local_seed.point = x;
          local_seed.start_dir = m3;
          local_seed.fa = fa_3;
        }
      }
    }
  }
  FiberReserve(fiber, fiber_length);
}

void Tractography::Follow2T(const int thread_id,
                            const size_t seed_index,
                            const SeedPointInfo &fiberStartSeed,
                            UKFFiber &fiber,
                            bool is_branching,
                            std::vector<SeedPointInfo> &branching_seeds,
                            std::vector<BranchingSeedAffiliation> &branching_seed_affiliation)
{
  int fiber_size = 100;
  int fiber_length = 0;
  // Unpack the fiberStartSeed information.
  vec3_t x = fiberStartSeed.point; // NOTICE that the x here is in ijk coordinate system
  State state = fiberStartSeed.state;

  ukfMatrixType p(fiberStartSeed.covariance);

  ukfPrecisionType fa = fiberStartSeed.fa;
  ukfPrecisionType fa2 = fiberStartSeed.fa2;
  ukfPrecisionType fa3 = fiberStartSeed.fa3;
  ukfPrecisionType dNormMSE = 0; // no error at the fiberStartSeed
  ukfPrecisionType trace = fiberStartSeed.trace;
  ukfPrecisionType trace2 = fiberStartSeed.trace2;

  //  Reserving fiber array memory so as to avoid resizing at every step
  FiberReserve(fiber, fiber_size);

  // Record start point.
  Record(x, fa, fa2, fa3, state, p, fiber, dNormMSE, trace, trace2); // writes state to fiber.state

  vec3_t m1, l1, m2, l2;
  m1 = fiberStartSeed.start_dir;

  // Track the fiber.
  ukfMatrixType signal_tmp(_model->signal_dim(), 1);
  ukfMatrixType state_tmp(_model->state_dim(), 1);

  int stepnr = 0;

  // useful for debuggingo
  //   std::ofstream stateFile;
  //   stateFile.open("states.txt", std::ios::app);

  while (true)
  {
    ++stepnr;

    Step2T(thread_id, x, m1, l1, m2, l2, fa, fa2, state, p, dNormMSE, trace, trace2);

    // Check if we should abort following this fiber. We abort if we reach the
    // CSF, if FA or GA get too small, if the curvature get's too high or if
    // the fiber gets too long.
    const bool is_brain = _signal_data->ScalarMaskValue(x) > 0; // _signal_data->Interp3ScalarMask(x) > 0.1; // is this 0.1 correct? yes

    // after here state does not change until next step.

    state_tmp.col(0) = state;

    _model->H(state_tmp, signal_tmp); // signal_tmp is written, but only used to calculate mean signal

    const ukfPrecisionType mean_signal = s2adc(signal_tmp);
    const bool in_csf = (_noddi) ? (mean_signal < _mean_signal_min) : (mean_signal < _mean_signal_min || fa < _fa_min);

    const bool is_curving = curve_radius(fiber.position) < _min_radius;

    if (!is_brain || in_csf || stepnr > _max_length // Stop when the fiber is too long
        || is_curving)
    {
      break;
    }

    if (_noddi)
      if (state[4] < 0.6 || state[9] < 0.6) // kappa1 and kappa2 break conditions
        break;

    if (fiber_length >= fiber_size)
    {
      // If fibersize is more than initally allocated size resizing further
      fiber_size += 100;
      FiberReserve(fiber, fiber_size);
    }

    if ((stepnr + 1) % _steps_per_record == 0)
    {
      fiber_length++;
      if (_noddi)
        Record(x, state[3], state[8], state[8], state, p, fiber, dNormMSE, state[4], state[9]); // state[8] duplicated just for Record function
      else
        Record(x, fa, fa2, fa3, state, p, fiber, dNormMSE, trace, trace2);
    }

    // Record branch if necessary.
    if (is_branching)
    {
      const ukfPrecisionType fa_1 = l2fa(l1[0], l1[1], l1[2]);
      const ukfPrecisionType fa_2 = l2fa(l2[0], l2[1], l2[2]);
      const bool is_two = (l1[0] > l1[1]) && (l1[0] > l1[2]) && (l2[0] > l2[1]) && (l2[0] > l2[2]) && (fa_1 > _fa_min) && (fa_2 > _fa_min);
      ukfPrecisionType theta = m1.dot(m2);
      bool is_branch = theta < _cos_theta_min && theta > _cos_theta_max;

      // If we have two tensors and the angle between them is large enough we
      // create a new seed for the branch. Since this branch is following the
      // second tensor we swap the state and covariance.
      if (is_two && is_branch)
      {
        branching_seeds.push_back(SeedPointInfo());
        SeedPointInfo &local_seed = branching_seeds[branching_seeds.size() - 1];
        branching_seed_affiliation.push_back(BranchingSeedAffiliation());
        BranchingSeedAffiliation &affiliation = branching_seed_affiliation[branching_seed_affiliation.size() - 1];

        affiliation.fiber_index_ = seed_index;
        affiliation.position_on_fiber_ = stepnr;

        int state_dim = _model->state_dim();
        local_seed.state.resize(state_dim);
        local_seed.state = state;
        local_seed.covariance.resize(state_dim, state_dim);
        local_seed.covariance = p;
        SwapState2T(local_seed.state, local_seed.covariance);
        local_seed.point = x;
        local_seed.start_dir = m2;
        local_seed.fa = fa_2;
      }
    }
  }
  FiberReserve(fiber, fiber_length);
  //   stateFile.close();
}

// Also read the comments to Follow2T above, it's documented better than this
// function here.
void Tractography::Follow1T(const int thread_id,
                            const SeedPointInfo &fiberStartSeed,
                            UKFFiber &fiber)
{
  int fiber_size = 100;
  int fiber_length = 0;
  assert(_model->signal_dim() == _signal_data->GetSignalDimension() * 2);

  vec3_t x = fiberStartSeed.point;
  State state = fiberStartSeed.state;

  // DEBUG
  //   std::cout << "fiberStartSeed state:\n";
  //   for (int i=0;i<state.size();++i) {
  //     std::cout << state[i] << " ";
  //   }
  //   std::cout << std::endl;

  ukfMatrixType p(fiberStartSeed.covariance);

  ukfPrecisionType fa = fiberStartSeed.fa;
  ukfPrecisionType fa2 = fiberStartSeed.fa2; // just needed for record
  ukfPrecisionType fa3 = fiberStartSeed.fa3; // just needed for record
  ukfPrecisionType trace = fiberStartSeed.trace;
  ukfPrecisionType trace2 = fiberStartSeed.trace2;

  ukfPrecisionType dNormMSE = 0; // no error at the fiberStartSeed

  //  Reserving fiber array memory so as to avoid resizing at every step
  FiberReserve(fiber, fiber_size);

  // Record start point.
  Record(x, fa, fa2, fa3, state, p, fiber, dNormMSE, trace, trace2);

  // Tract the fiber.
  ukfMatrixType signal_tmp(_model->signal_dim(), 1);
  ukfMatrixType state_tmp(_model->state_dim(), 1);

  int stepnr = 0;
  while (true)
  {
    ++stepnr;

    Step1T(thread_id, x, fa, state, p, dNormMSE, trace);

    // Terminate if off brain or in CSF.
    const bool is_brain = _signal_data->ScalarMaskValue(x) > 0; //_signal_data->Interp3ScalarMask(x) > 0.1; // x is the seed point
    state_tmp.col(0) = state;

    _model->H(state_tmp, signal_tmp);

    // Check mean_signal threshold
    const ukfPrecisionType mean_signal = s2adc(signal_tmp);
    bool in_csf;
    if (_noddi)
      in_csf = mean_signal < _mean_signal_min;
    else
      in_csf = mean_signal < _mean_signal_min || fa < _fa_min;

    bool is_curving = curve_radius(fiber.position) < _min_radius;

    if (!is_brain || in_csf || stepnr > _max_length // Stop when fiber is too long
        || is_curving)
    {
      break;
    }
    if (_noddi)
      if (state[4] < 1.2) // checking kappa
        break;

    if (fiber_length >= fiber_size)
    {
      // If fibersize is more than initally allocated size resizing further
      fiber_size += 100;
      FiberReserve(fiber, fiber_size);
    }

    if ((stepnr + 1) % _steps_per_record == 0)
    {
      fiber_length++;
      if (_noddi)
        Record(x, state[3], fa2, fa3, state, p, fiber, dNormMSE, state[4], trace2);
      else
        Record(x, fa, fa2, fa3, state, p, fiber, dNormMSE, trace, trace2);
    }
  }
  FiberReserve(fiber, fiber_length);
}

void Tractography::Step3T(const int thread_id,
                          vec3_t &x,
                          vec3_t &m1,
                          vec3_t &m2,
                          vec3_t &m3,
                          State &state,
                          ukfMatrixType &covariance,
                          ukfPrecisionType &dNormMSE,
                          ukfPrecisionType &rtop1,
                          ukfPrecisionType &rtop2,
                          ukfPrecisionType &rtop3,
                          ukfPrecisionType &rtopModel,
                          ukfPrecisionType &rtopSignal)
{
  // For ridgelets bi-exp model
  assert(static_cast<int>(covariance.cols()) == _model->state_dim() &&
         static_cast<int>(covariance.rows()) == _model->state_dim());
  assert(static_cast<int>(state.size()) == _model->state_dim());
  State state_new(_model->state_dim());
  State state_prev(_model->state_dim());
  state_prev = state;

  ukfMatrixType covariance_new(_model->state_dim(), _model->state_dim());
  covariance_new.setConstant(ukfZero);

  // Use the Unscented Kalman Filter to get the next estimate.
  ukfVectorType signal(_signal_data->GetSignalDimension() * 2);
  _signal_data->Interp3Signal(x, signal);

  LoopUKF(thread_id, state, covariance, signal, state_new, covariance_new, dNormMSE);
  // cout << "ukf loop end" << endl;
  // cout << "m1 state " << state(0) << " " << state(1) << " " << state(2) << endl;
  // cout << "m1 " << m1.transpose() << endl;

  vec3_t old_dir = m1;

  _model->State2Tensor3T(state, old_dir, m1, m2, m3);
  // ukfPrecisionType degre = std::acos(std::min(std::max(m1.dot(old_dir) / (m1.norm() * old_dir.norm()), -1.0), 1.0)) * 180 / Pi;
  // if (degre > 70) {
  //   cout << "degre " << degre << endl;
  //   cout << "state " << state.transpose() << endl;
  //   cout << "old s " << state_prev.transpose() << endl;
  //   cout << "m1 " << m1.transpose() << endl;
  //   cout << "m1 old " << old_dir << endl;
  // }
  // cout << "m1 state " << state(0) << " " << state(1) << " " << state(2) << endl;
  // cout << "m1 " << m1.transpose() << endl;

  ukfPrecisionType _rtop1, _rtop2, _rtop3, _rtopModel, _rtopSignal;
  //stdVecState local_state = ConvertVector<State, stdVecState>(state);

  computeRTOPfromState(state, _rtopModel, _rtop1, _rtop2, _rtop3);
  computeRTOPfromSignal(_rtopSignal, signal);

  rtop1 = _rtop1;
  rtop2 = _rtop2;
  rtop3 = _rtop3;
  rtopModel = _rtopModel;
  rtopSignal = _rtopSignal;

  // BELOW is a huge pile of code which might be useful in future or for
  // debugging

  // ukfPrecisionType dot1 = m1.dot(old_dir);
  // ukfPrecisionType dot2 = m2.dot(old_dir);
  // ukfPrecisionType dot3 = m3.dot(old_dir);

  // NEED TO FIX
  // if (dot1 < dot2 && dot3 < dot2)
  // {
  //   // std::cout << "swap 2 triggered" << std::endl;
  //   // std::cout << "angle m2 m1 " << RadToDeg(std::acos(m2.dot(m1))) << std::endl;
  //   // std::cout << "angle dot1 " << RadToDeg(std::acos(dot1)) << std::endl;
  //   // std::cout << "angle dot2 " << RadToDeg(std::acos(dot2)) << std::endl;
  //   // std::cout << "angle dot3 " << RadToDeg(std::acos(dot3)) << std::endl;
  //   // std::cout << "old dir " << old_dir << std::endl;
  //   // std::cout << "m1 " << m1 << std::endl;
  //   // std::cout << "m2 " << m2 << std::endl;
  //   // std::cout << "dot1 " << dot1 << std::endl;
  //   // std::cout << "dot2 " << dot2 << std::endl;
  //   // std::cout << "dot3 " << dot3 << std::endl;

  //   // std::cout << "w1 " << state(21) << " w2 " << state(22) << std::endl;

  //   // Switch dirs and lambdas.
  //   vec3_t tmp = m1;
  //   m1 = m2;
  //   m2 = tmp;
  //   // std::cout << "cov before det = " << covariance.determinant() << " diff norm " << (covariance - covariance.transpose()).norm() << std::endl;
  //   // Swap state.
  //   std::cout << "swap 2 " << std::endl;
  //   SwapState3T_BiExp(state, covariance, 2);
  //   // std::cout << "cov after det =  " << covariance.determinant()  << " diff norm " << (covariance - covariance.transpose()).norm() << std::endl;

  //   //w11 w12 w13 w14
  //   //w21 w22 w23 w24
  //   //w31 w32 w33 w43
  //   //w41 w42 w43 w44
  // }

  // // NEED TO FIX
  // else if (dot1 < dot3)
  // {
  //   // Switch dirs and lambdas.
  //   vec3_t tmp = m1;
  //   m1 = m3;
  //   m3 = tmp;

  //   // Swap state.
  //   // std::cout << "swap 3 triggered" << std::endl;
  //   // std::cout << "normed? " << m1.norm() << " " << m3.norm() << std::endl;
  //   // std::cout << "angle " << std::acos(m3.dot(m1)) << std::endl;
  //   //std::cout << "state before\n " << state << std::endl;
  //   //std::cout << "covariance before\n " << covariance << std::endl;
  //   std::cout << "swap 3 " << std::endl;
  //   SwapState3T_BiExp(state, covariance, 3);
  //   //std::cout << "state after\n " << state << std::endl;
  //   // std::cout << "covariance after\n " << covariance << std::endl;
  // }

  vec3_t dx;
  {
    const vec3_t dir = m1; // The dir is a unit vector in ijk coordinate system indicating the direction of step
    const vec3_t voxel = _signal_data->voxel();
    dx << dir[2] / voxel[0], // By dividing by the voxel size, it's guaranteed that the step
        // represented by dx is 1mm in RAS coordinate system, no matter whether
        // the voxel is isotropic or not
        dir[1] / voxel[1], // The value is scaled back during the ijk->RAS transformation when
        // outputted
        dir[0] / voxel[2];

    x = x + dx * _stepLength; // The x here is in ijk coordinate system.
  }
}

void Tractography::Step3T(const int thread_id,
                          vec3_t &x,
                          vec3_t &m1,
                          vec3_t &l1,
                          vec3_t &m2,
                          vec3_t &l2,
                          vec3_t &m3,
                          vec3_t &l3,
                          ukfPrecisionType &fa,
                          ukfPrecisionType &fa2,
                          ukfPrecisionType &fa3,
                          State &state,
                          ukfMatrixType &covariance,
                          ukfPrecisionType &dNormMSE,
                          ukfPrecisionType &trace,
                          ukfPrecisionType &trace2)
{

  assert(static_cast<int>(covariance.cols()) == _model->state_dim() &&
         static_cast<int>(covariance.rows()) == _model->state_dim());
  assert(static_cast<int>(state.size()) == _model->state_dim());
  State state_new(_model->state_dim());

  ukfMatrixType covariance_new(_model->state_dim(), _model->state_dim());

  // Use the Unscented Kalman Filter to get the next estimate.
  ukfVectorType signal(_signal_data->GetSignalDimension() * 2);
  _signal_data->Interp3Signal(x, signal);
  _ukf[thread_id]->Filter(state, covariance, signal, state_new, covariance_new, dNormMSE);

  state = state_new;
  covariance = covariance_new;

  vec3_t old_dir = m1;

  _model->State2Tensor3T(state, old_dir, m1, l1, m2, l2, m3, l3);
  trace = l1[0] + l1[1] + l1[2];
  trace2 = l2[0] + l2[1] + l2[2];

  ukfPrecisionType dot1 = m1.dot(old_dir);
  ukfPrecisionType dot2 = m2.dot(old_dir);
  ukfPrecisionType dot3 = m3.dot(old_dir);
  if (dot1 < dot2 && dot3 < dot2)
  {
    // Switch dirs and lambdas.
    vec3_t tmp = m1;
    m1 = m2;
    m2 = tmp;
    tmp = l1;
    l1 = l2;
    l2 = tmp;

    // Swap state.

    SwapState3T(state, covariance, 2);
  }
  else if (dot1 < dot3)
  {
    // Switch dirs and lambdas.
    vec3_t tmp = m1;
    m1 = m3;
    m3 = tmp;
    tmp = l1;
    l1 = l3;
    l3 = tmp;

    // Swap state.
    SwapState3T(state, covariance, 3);
  }

  // Update FA. If the first lamba is not the largest anymore the FA is set to
  // 0, and the 0 FA value will lead to abortion in the tractography loop.
  if (l1[0] < l1[1] || l1[0] < l1[2])
  {
    fa = ukfZero;
  }
  else
  {
    fa = l2fa(l1[0], l1[1], l1[2]);
    fa2 = l2fa(l2[0], l2[1], l2[2]);
    fa3 = l2fa(l3[0], l3[1], l3[2]);
  }

  const vec3_t &voxel = _signal_data->voxel();

  // CB: Bug corrected, dir[i] should be divided by voxel[i]
  vec3_t dx;
  dx << m1[2] / voxel[0],
      m1[1] / voxel[1],
      m1[0] / voxel[2];
  x = x + dx * _stepLength;
}

void Tractography::LoopUKF(const int thread_id,
                           State &state,
                           ukfMatrixType &covariance,
                           ukfVectorType &signal,
                           State &state_new,
                           ukfMatrixType &covariance_new,
                           ukfPrecisionType &dNormMSE)
{
  _ukf[thread_id]->Filter(state, covariance, signal, state_new, covariance_new, dNormMSE);

  state = state_new;
  covariance = covariance_new;

  ukfPrecisionType er_org = dNormMSE;
  ukfPrecisionType er = er_org;

  State state_prev = state;

  for (int jj = 0; jj < _maxUKFIterations; ++jj)
  {
    _ukf[thread_id]->Filter(state, covariance, signal, state_new, covariance_new, dNormMSE);
    state = state_new;

    er_org = er;
    er = dNormMSE;

    if (er_org - er < 0.001)
      break;

    state_prev = state;
  }

  state = state_prev;
}

void Tractography::Step2T(const int thread_id,
                          vec3_t &x,
                          vec3_t &m1,
                          vec3_t &l1,
                          vec3_t &m2,
                          vec3_t &l2,
                          ukfPrecisionType &fa,
                          ukfPrecisionType &fa2,
                          State &state,
                          ukfMatrixType &covariance,
                          ukfPrecisionType &dNormMSE,
                          ukfPrecisionType &trace,
                          ukfPrecisionType &trace2)
{
  assert(static_cast<int>(covariance.cols()) == _model->state_dim() &&
         static_cast<int>(covariance.rows()) == _model->state_dim());
  assert(static_cast<int>(state.size()) == _model->state_dim());

  State state_new(_model->state_dim());
  ukfMatrixType covariance_new(_model->state_dim(), _model->state_dim());
  covariance_new.setConstant(ukfZero);

  // Use the Unscented Kalman Filter to get the next estimate.
  ukfVectorType signal(_signal_data->GetSignalDimension() * 2);
  _signal_data->Interp3Signal(x, signal);

  _ukf[thread_id]->Filter(state, covariance, signal, state_new, covariance_new, dNormMSE);

  state = state_new;
  covariance = covariance_new;

  const vec3_t old_dir = m1; // Direction in last step
  ukfPrecisionType fa_tensor_1 = ukfZero;
  ukfPrecisionType fa_tensor_2 = ukfZero;
  if (_noddi)
  {
    initNormalized(m1, state[0], state[1], state[2]);
    initNormalized(m2, state[5], state[6], state[7]);
    if (m1[0] * old_dir[0] + m1[1] * old_dir[1] + m1[2] * old_dir[2] < 0)
    {
      m1 = -m1;
    }
    if (m2[0] * old_dir[0] + m2[1] * old_dir[1] + m2[2] * old_dir[2] < 0)
    {
      m2 = -m2;
    }
  }
  else
  {
    _model->State2Tensor2T(state, old_dir, m1, l1, m2, l2); // The returned m1 and m2 are unit vector here
    trace = l1[0] + l1[1] + l1[2];
    trace2 = l2[0] + l2[1] + l2[2];
    fa_tensor_1 = l2fa(l1[0], l1[1], l1[2]);
    fa_tensor_2 = l2fa(l2[0], l2[1], l2[2]);
  }
  const ukfPrecisionType tensor_angle = RadToDeg(std::acos(m1.dot(m2)));

  if (m1.dot(old_dir) < m2.dot(old_dir))
  {
    // Switch dirs and lambdas.
    vec3_t tmp = m1;
    m1 = m2;
    m2 = tmp;
    tmp = l1;
    l1 = l2;
    l2 = tmp;
    // Need to swap scalar measures too.
    ukfPrecisionType tmpScalar = fa_tensor_1;
    fa_tensor_1 = fa_tensor_2;
    fa_tensor_2 = tmpScalar;
    tmpScalar = trace;
    trace = trace2;
    trace2 = tmpScalar;
    ukfMatrixType old = covariance;
    SwapState2T(state, covariance); // Swap the two tensors
  }

  if (tensor_angle <= 20)
  {
    if (_noddi)
    {
      vec3_t tmp = m1;
      m1 = m2;
      m2 = tmp;
      ukfMatrixType old = covariance;

      SwapState2T(state, covariance); // Swap the two tensors
    }
    else if (std::min(fa_tensor_1, fa_tensor_2) <= 0.2)
    {
      if (fa_tensor_1 > 0.2)
      {
        // do nothing
        // i.e. keep m1 as principal direction
      }
      else
      {
        // switch directions, note: FA will be re-calculated after
        vec3_t tmp = m1;
        m1 = m2;
        m2 = tmp;
        tmp = l1;
        l1 = l2;
        l2 = tmp;

        ukfMatrixType old = covariance;

        SwapState2T(state, covariance); // Swap the two tensors
      }
    }
  }
  // Update FA. If the first lamba is not the largest anymore the FA is set to
  // 0 what will lead to abortion in the tractography loop.
  if (l1[0] < l1[1] || l1[0] < l1[2])
  {
    fa = ukfZero;
    fa2 = ukfZero;
  }
  else
  {
    fa = l2fa(l1[0], l1[1], l1[2]);
    fa2 = l2fa(l2[0], l2[1], l2[2]);
  }
  vec3_t dx;
  {
    const vec3_t dir = m1; // The dir is a unit vector in ijk coordinate system indicating the direction of step
    const vec3_t voxel = _signal_data->voxel();
    dx << dir[2] / voxel[0], // By dividing by the voxel size, it's guaranteed that the step
        // represented by dx is 1mm in RAS coordinate system, no matter whether
        // the voxel is isotropic or not
        dir[1] / voxel[1], // The value is scaled back during the ijk->RAS transformation when
        // outputted
        dir[0] / voxel[2];

    x = x + dx * _stepLength; // The x here is in ijk coordinate system.
  }
  // NOTICE that the coordinate order of x is in reverse order with respect to the axis order in the original signal
  // file.
  // This coordinate order is filpped back during output
  // The step length is in World space
  // throw;
}

void Tractography::Step1T(const int thread_id,
                          vec3_t &x,
                          ukfPrecisionType &fa,
                          State &state,
                          ukfMatrixType &covariance,
                          ukfPrecisionType &dNormMSE,
                          ukfPrecisionType &trace)
{

  assert(static_cast<int>(covariance.cols()) == _model->state_dim() &&
         static_cast<int>(covariance.rows()) == _model->state_dim());
  assert(static_cast<int>(state.size()) == _model->state_dim());
  State state_new(_model->state_dim());
  ukfMatrixType covariance_new(_model->state_dim(), _model->state_dim());

  ukfVectorType signal(_signal_data->GetSignalDimension() * 2);
  _signal_data->Interp3Signal(x, signal);

  _ukf[thread_id]->Filter(state, covariance, signal, state_new, covariance_new, dNormMSE);

  state = state_new;
  covariance = covariance_new;

  vec3_t dir;
  if (_noddi)
  {
    dir << state[0], state[1], state[2];
  }
  else
  {
    vec3_t l;
    _model->State2Tensor1T(state, dir, l);

    trace = l[0] + l[1] + l[2];

    // Update FA. If the first lamba is not the largest anymore the FA is set to
    // 0 what will lead to abortion in the tractography loop.
    if (l[0] < l[1] || l[0] < l[2])
    {
      fa = ukfZero;
    }
    else
    {
      fa = l2fa(l[0], l[1], l[2]);
    }
  }
  vec3_t voxel = _signal_data->voxel();

  vec3_t dx;
  dx << dir[2] / voxel[0],
      dir[1] / voxel[1],
      dir[0] / voxel[2];
  x = x + dx * _stepLength;
}

void Tractography::SwapState3T(stdVecState &state,
                               ukfMatrixType &covariance,
                               int i)
{
  State tmp_state = ConvertVector<stdVecState, State>(state);
  SwapState3T(tmp_state, covariance, i);
  state = ConvertVector<State, stdVecState>(tmp_state);
}

void Tractography::SwapState3T(State &state,
                               ukfMatrixType &covariance,
                               int i)
{
  // This function is only for 3T. No free water
  assert(i == 2 || i == 3);

  int state_dim = _model->state_dim();
  ukfMatrixType tmp(state_dim, state_dim);
  state_dim /= 3;
  assert(state_dim == 5 || state_dim == 6);
  --i;
  int j = i == 1 ? 2 : 1;
  i *= state_dim;
  j *= state_dim;

  tmp.setConstant(ukfZero);
  tmp = covariance;
  covariance.block(i, i, state_dim, state_dim) = tmp.block(0, 0, state_dim, state_dim);
  covariance.block(0, 0, state_dim, state_dim) = tmp.block(i, i, state_dim, state_dim);
  covariance.block(0, i, state_dim, state_dim) = tmp.block(i, 0, state_dim, state_dim);
  covariance.block(i, 0, state_dim, state_dim) = tmp.block(0, i, state_dim, state_dim);

  covariance.block(j, i, state_dim, state_dim) = tmp.block(j, 0, state_dim, state_dim);
  covariance.block(j, 0, state_dim, state_dim) = tmp.block(j, i, state_dim, state_dim);
  covariance.block(i, j, state_dim, state_dim) = tmp.block(0, j, state_dim, state_dim);
  covariance.block(0, j, state_dim, state_dim) = tmp.block(i, j, state_dim, state_dim);

  // Swap the state.
  const ukfVectorType tmp_vec = state;
  state.segment(i, state_dim) = tmp_vec.segment(0, state_dim);
  state.segment(0, state_dim) = tmp_vec.segment(i, state_dim);
}

void Tractography::SwapState3T_BiExp(stdVecState &state,
                                     ukfMatrixType &covariance,
                                     int i)
{
  State tmp_state = ConvertVector<stdVecState, State>(state);
  SwapState3T_BiExp(tmp_state, covariance, i);
  state = ConvertVector<State, stdVecState>(tmp_state);
}

void Tractography::SwapState3T_BiExp(State &state,
                                     ukfMatrixType &covariance,
                                     int i)
{
  // This function is only for Bi-exp model
  assert(i == 2 || i == 3);
  int ishift = i - 1;

  int state_dim = _model->state_dim();
  assert(state_dim == 25);

  ukfMatrixType tmp(state_dim, state_dim);
  state_dim = 7;
  --i;
  int j = i == 1 ? 2 : 1;
  i *= state_dim;
  j *= state_dim;

  int tshift = 3 * state_dim;
  int mshift = ishift * state_dim;

  tmp = covariance;
  covariance.block(i, i, state_dim, state_dim) = tmp.block(0, 0, state_dim, state_dim);
  covariance.block(0, 0, state_dim, state_dim) = tmp.block(i, i, state_dim, state_dim);

  covariance.block(0, i, state_dim, state_dim) = tmp.block(i, 0, state_dim, state_dim);
  covariance.block(i, 0, state_dim, state_dim) = tmp.block(0, i, state_dim, state_dim);

  covariance.block(j, i, state_dim, state_dim) = tmp.block(j, 0, state_dim, state_dim);
  covariance.block(j, 0, state_dim, state_dim) = tmp.block(j, i, state_dim, state_dim);

  covariance.block(i, j, state_dim, state_dim) = tmp.block(0, j, state_dim, state_dim);
  covariance.block(0, j, state_dim, state_dim) = tmp.block(i, j, state_dim, state_dim);

  // Swap weights in covariance matrix
  // Lower parp
  covariance.block(tshift, mshift, 4, state_dim) = tmp.block(tshift, 0, 4, state_dim);
  covariance.block(tshift, 0, 4, state_dim) = tmp.block(tshift, mshift, 4, state_dim);
  // Right part
  covariance.block(mshift, tshift, state_dim, 4) = tmp.block(0, tshift, state_dim, 4);
  covariance.block(0, tshift, state_dim, 4) = tmp.block(mshift, tshift, state_dim, 4);

  // Lower right 4x4 matrix
  // !!!Need to check this!!!!
  int corn_shift = tshift + ishift;
  covariance(corn_shift, corn_shift) = tmp(tshift, tshift);
  covariance(tshift, tshift) = tmp(corn_shift, corn_shift);

  covariance(tshift, corn_shift) = tmp(corn_shift, tshift);
  covariance(corn_shift, tshift) = tmp(tshift, corn_shift);

  int oneshift = tshift + 1;
  int twoshift = tshift + 2;

  if (ishift == 1)
  {
    covariance.block(twoshift, tshift, 2, 1) = tmp.block(twoshift, oneshift, 2, 1);
    covariance.block(twoshift, oneshift, 2, 1) = tmp.block(twoshift, tshift, 2, 1);

    covariance.block(tshift, twoshift, 1, 2) = tmp.block(oneshift, twoshift, 1, 2);
    covariance.block(oneshift, twoshift, 1, 2) = tmp.block(tshift, twoshift, 1, 2);
  }
  else if (ishift == 2)
  {
    // Horizontal mid
    covariance(oneshift, tshift) = tmp(oneshift, twoshift);
    covariance(oneshift, twoshift) = tmp(oneshift, tshift);

    // Vertical mid
    covariance(twoshift, oneshift) = tmp(tshift, oneshift);
    covariance(tshift, oneshift) = tmp(twoshift, oneshift);

    int threeshift = tshift + 3;
    // Horizontal bottom
    covariance(threeshift, twoshift) = tmp(threeshift, tshift);
    covariance(threeshift, tshift) = tmp(threeshift, twoshift);
    // Vertical right
    covariance(twoshift, threeshift) = tmp(tshift, threeshift);
    covariance(tshift, threeshift) = tmp(twoshift, threeshift);
  }
  else
  {
    std::cout << "Error: BiExp swap state function works only for 3 Tensors.\n";
    throw;
  }

  // Swap the state
  const ukfVectorType tmp_vec = state;
  state.segment(i, state_dim) = tmp_vec.segment(0, state_dim);
  state.segment(0, state_dim) = tmp_vec.segment(i, state_dim);

  const ukfPrecisionType tmp_weight = state(21);
  int iw = 21 + ishift;
  state(21) = state(iw);
  state(iw) = tmp_weight;
}

void Tractography::SwapState2T(State &state,
                               ukfMatrixType &covariance)
{
  // This function is only for 2T.
  int state_dim = _model->state_dim();

  ukfMatrixType tmp(state_dim, state_dim);
  bool bUnevenState = false;

  if (state_dim % 2 != 0)
  {
    bUnevenState = true; // there is a weight term in the end of the state
  }
  state_dim = state_dim >> 1; // for uneven state (fw) rounds down, thats good

  tmp.setConstant(ukfZero);
  tmp = covariance;

  covariance.block(state_dim, state_dim, state_dim, state_dim) = tmp.block(0, 0, state_dim, state_dim);
  covariance.block(0, 0, state_dim, state_dim) = tmp.block(state_dim, state_dim, state_dim, state_dim);
  covariance.block(0, state_dim, state_dim, state_dim) = tmp.block(state_dim, 0, state_dim, state_dim);
  covariance.block(state_dim, 0, state_dim, state_dim) = tmp.block(0, state_dim, state_dim, state_dim);

  if (bUnevenState) // change covariances of weights and state so they match the state again
  {
    covariance.block(state_dim * 2, state_dim, 1, state_dim) = tmp.block(state_dim * 2, 0, 1, state_dim);
    covariance.block(state_dim * 2, 0, 1, state_dim) = tmp.block(state_dim * 2, state_dim, 1, state_dim);

    covariance.block(state_dim, state_dim * 2, state_dim, 1) = tmp.block(0, state_dim * 2, state_dim, 1);
    covariance.block(0, state_dim * 2, state_dim, 1) = tmp.block(state_dim, state_dim * 2, state_dim, 1);
  }

  // Swap the state.
  const ukfVectorType tmp_vec = state;
  state.segment(state_dim, state_dim) = tmp_vec.segment(0, state_dim);
  state.segment(0, state_dim) = tmp_vec.segment(state_dim, state_dim);
}

void Tractography::Record(const vec3_t &x, const ukfPrecisionType fa, const ukfPrecisionType fa2, const ukfPrecisionType fa3, const State &state,
                          const ukfMatrixType p, UKFFiber &fiber, const ukfPrecisionType dNormMSE, const ukfPrecisionType trace, const ukfPrecisionType trace2)
{
  // if Noddi model is used Kappa is stored in trace, Vic in fa and Viso in freewater
  assert(_model->state_dim() == static_cast<int>(state.size()));
  assert(p.rows() == static_cast<unsigned int>(state.size()) &&
         p.cols() == static_cast<unsigned int>(state.size()));

  // std::cout << "x: " << x[0] << " " << x[1] << " " << x[2] << std::endl;
  fiber.position.push_back(x);
  fiber.norm.push_back(p.norm());

  if (_record_nmse)
  {
    fiber.normMSE.push_back(dNormMSE);
  }

  if ((_record_trace || _record_kappa) && !_record_rtop && !_diffusion_propagator)
  {
    fiber.trace.push_back(2 * (atan(1 / trace) / 3.14));
    if (_num_tensors >= 2)
    {
      fiber.trace2.push_back(2 * (atan(1 / trace2) / 3.14));
    }
  }

  if (_record_rtop)
  {
    fiber.trace.push_back(trace);
    fiber.trace2.push_back(trace2);
  }

  if (_record_fa || _record_Vic || _record_rtop)
  {
    fiber.fa.push_back(fa);
    if (_num_tensors >= 2)
    {
      fiber.fa2.push_back(fa2);
    }
    if (_num_tensors == 3)
    {
      fiber.fa3.push_back(fa3);
    }
  }

  if (_record_weights)
  {
    ukfPrecisionType w1 = state[21];
    ukfPrecisionType w2 = state[22];
    ukfPrecisionType w3 = state[23];
    ukfPrecisionType wiso = state[24];

    fiber.w1.push_back(w1);
    fiber.w2.push_back(w2);
    fiber.w3.push_back(w3);
    fiber.free_water.push_back(wiso);

    /* Angles */
    State store_state(state);
    vec3_t dir1;
    initNormalized(dir1, store_state[0], store_state[1], store_state[2]);
    vec3_t dir2;
    initNormalized(dir2, store_state[7], store_state[8], store_state[9]);
    vec3_t dir3;
    initNormalized(dir3, store_state[14], store_state[15], store_state[16]);

    ukfPrecisionType d1d2 = std::min(RadToDeg(std::acos(dir1.dot(dir2))), RadToDeg(std::acos(dir1.dot(-dir2))));
    ukfPrecisionType d1d3 = std::min(RadToDeg(std::acos(dir1.dot(dir3))), RadToDeg(std::acos(dir1.dot(-dir3))));

    fiber.w1w2angle.push_back(d1d2);
    fiber.w1w3angle.push_back(d1d3);
  }

  if (_record_Viso)
  {
    ukfPrecisionType viso = state[_nPosFreeWater];
    if (viso < 0)
    {
      if (viso >= -1.0e-4) // for small errors just round it to 0
      {
        viso = 0;
      }
      else // for too big errors exit with exception.
      {
        std::cout << "Error: program produced negative free water.\n";
        throw;
      }
    }
    fiber.free_water.push_back(viso);
  }

  if (_record_free_water)
  {
    ukfPrecisionType fw = 1 - state[_nPosFreeWater];
    // sometimes QP produces slightly negative results due to numerical errors in Quadratic Programming, the weight is
    // clipped in F() and H() but its still possible that
    // a slightly negative weight gets here, because the filter ends with a constrain step.
    if (fw < 0)
    {
      if (fw >= -1.0e-4) // for small errors just round it to 0
      {
        fw = 0;
      }
      else // for too big errors exit with exception.
      {
        std::cout << "Error: program produced negative free water.\n";
        throw;
      }
    }
    fiber.free_water.push_back(fw);
  }

  // Record the state
  if ((state.size() == 5 || state.size() == 10 || state.size() == 15) && !_diffusion_propagator) // i.e. simple model
  {                                                                                              // Normalize direction before storing it;
    State store_state(state);
    vec3_t dir;
    initNormalized(dir, store_state[0], store_state[1], store_state[2]);
    store_state[0] = dir[0];
    store_state[1] = dir[1];
    store_state[2] = dir[2];

    if (state.size() == 10)
    {
      initNormalized(dir, store_state[5], store_state[6], store_state[7]);
      store_state[5] = dir[0];
      store_state[6] = dir[1];
      store_state[7] = dir[2];
    }
    if (state.size() == 15)
    {
      initNormalized(dir, store_state[10], store_state[11], store_state[12]);
      store_state[10] = dir[0];
      store_state[11] = dir[1];
      store_state[12] = dir[2];
    }
    fiber.state.push_back(store_state);
  }
  else if (_diffusion_propagator)
  {
    State store_state(state);
    vec3_t dir;

    // normalize m1
    initNormalized(dir, store_state[0], store_state[1], store_state[2]);
    store_state[0] = dir[0];
    store_state[1] = dir[1];
    store_state[2] = dir[2];

    // normalize m2
    initNormalized(dir, store_state[7], store_state[8], store_state[9]);
    store_state[7] = dir[0];
    store_state[8] = dir[1];
    store_state[9] = dir[2];

    // normalize m2
    initNormalized(dir, store_state[14], store_state[15], store_state[16]);
    store_state[14] = dir[0];
    store_state[15] = dir[1];
    store_state[16] = dir[2];

    fiber.state.push_back(store_state);
  }
  else
  {
    // Normal state
    fiber.state.push_back(state);
  }

  if (_record_cov)
  {
    fiber.covariance.push_back(p);
  }
}

void Tractography::RecordWeightTrack(const vec3_t &x, UKFFiber &fiber, ukfPrecisionType d1, ukfPrecisionType d2, ukfPrecisionType d3)
{
  vec3_t x1;
  vec3_t x2;

  vec3_t dx;
  {
    const vec3_t voxel = _signal_data->voxel();
    dx << d3 / voxel[0],
        d2 / voxel[1],
        d1 / voxel[2];

    x1 = x - dx * _stepLength;
    x2 = x + dx * _stepLength;
  }
  fiber.position.push_back(x1);
  fiber.position.push_back(x2);
}

void Tractography::FiberReserve(UKFFiber &fiber, int fiber_size)
{
  // Reserving space for fiber
  fiber.position.reserve(fiber_size);
  fiber.norm.reserve(fiber_size);
  fiber.state.reserve(fiber_size);
  if (_record_nmse)
  {
    fiber.normMSE.reserve(fiber_size);
  }

  if (_record_trace || _record_kappa || _record_rtop)
  {
    fiber.trace.reserve(fiber_size);
    if (_num_tensors >= 2)
    {
      fiber.trace2.reserve(fiber_size);
    }
  }

  if (_record_fa || _record_Vic || _record_rtop)
  {
    fiber.fa.reserve(fiber_size);
    if (_num_tensors >= 2)
    {
      fiber.fa2.reserve(fiber_size);
    }
    if (_num_tensors >= 3)
    {
      fiber.fa3.reserve(fiber_size);
    }
  }
  if (_record_free_water || _record_Viso)
  {
    fiber.free_water.reserve(fiber_size);
  }
  if (_record_weights)
  {
    fiber.w1.reserve(fiber_size);
    fiber.w2.reserve(fiber_size);
    fiber.w3.reserve(fiber_size);
    fiber.free_water.reserve(fiber_size);
    fiber.w1w2angle.reserve(fiber_size);
    fiber.w1w3angle.reserve(fiber_size);
  }
  if (_record_cov)
  {
    fiber.covariance.reserve(fiber_size);
  }
}

void Tractography::FiberReserveWeightTrack(UKFFiber &fiber, int fiber_size)
{
  // Reserving space for fiber
  fiber.position.reserve(fiber_size);
}
