#include "filter_ridg.h"

// 2T Bi-Exponential model with spherical ridgelets //
// Functions for 3-tensor bi-exponential simple model.
void Ridg_BiExp_FW::F(ukfMatrixType &X, ukfVectorType s, const ukfMatrixType &covMatrix) const
{
	assert(_signal_dim > 0);
	assert(X.rows() == static_cast<unsigned int>(_state_dim) &&
		   (X.cols() == static_cast<unsigned int>(2 * _state_dim + 1) ||
			X.cols() == 1));

	UtilMath<ukfPrecisionType, ukfMatrixType, ukfVectorType> m;

	ukfVectorType HighBSignalValues(signal_mask.size());
	for (int indx = 0; indx < signal_mask.size(); ++indx)
		HighBSignalValues(indx) = s(signal_mask(indx));

	ukfVectorType C;
	{
		SOLVERS<ukfPrecisionType, ukfMatrixType, ukfVectorType> slv(A, HighBSignalValues, fista_lambda);
		slv.FISTA(C);
	}

	ukfVectorType ODF = Q * C;

	ukfMatrixType exe_vol;
	ukfMatrixType dir_vol;
	unsigned int n_of_dirs;
	m.FindODFMaxima(exe_vol, dir_vol, ODF, conn, nu, max_odf_thresh, n_of_dirs);

	ukfVectorType closest(6);
	vec3_t m1;
	vec3_t m_temp;
	vec3_t o;
	vec3_t o_a;
	vec3_t o1;
	vec3_t m2;
	vec3_t o2;
	vec3_t m3;
	vec3_t o3;
	vec3_t max_odf;

	ukfPrecisionType x1;
	ukfPrecisionType x2;
	ukfPrecisionType x3;

	ukfPrecisionType w1;
	ukfPrecisionType w2;
	ukfPrecisionType w3;

	for (unsigned int i = 0; i < X.cols(); ++i)
	{
		max_odf.setZero();
		o1.setZero();
		o2.setZero();
		o3.setZero();

		m1 << X(0, i), X(1, i), X(2, i);
		m2 << X(7, i), X(8, i), X(9, i);
		m3 << X(14, i), X(15, i), X(16, i);
		m1.normalize();
		m2.normalize();
		m3.normalize();

		ukfPrecisionType denom = X(21, i) + X(22, i) + X(23, i);
		X(21, i) = X(21, i) / denom;
		X(22, i) = X(22, i) / denom;
		X(23, i) = X(23, i) / denom;

		m_temp = m1;

		// Check that the eigenvalues are greater or equal to the minimum value
		// and less or equal to the maximum value
		// Tensor 1
		X(3, i) = std::max(X(3, i), _lambda_min_fast_diffusion);
		X(4, i) = std::max(X(4, i), _lambda_min_fast_diffusion);
		X(5, i) = std::max(X(5, i), _lambda_min_slow_diffusion);
		X(6, i) = std::max(X(6, i), _lambda_min_slow_diffusion);

		X(3, i) = std::min(X(3, i), _lambda_max_diffusion);
		X(4, i) = std::min(X(4, i), _lambda_max_diffusion);
		X(5, i) = std::min(X(5, i), _lambda_max_diffusion);
		X(6, i) = std::min(X(6, i), _lambda_max_diffusion);

		// Tensor 2
		X(10, i) = std::max(X(10, i), _lambda_min_fast_diffusion);
		X(11, i) = std::max(X(11, i), _lambda_min_fast_diffusion);
		X(12, i) = std::max(X(12, i), _lambda_min_slow_diffusion);
		X(13, i) = std::max(X(13, i), _lambda_min_slow_diffusion);

		X(10, i) = std::min(X(10, i), _lambda_max_diffusion);
		X(11, i) = std::min(X(11, i), _lambda_max_diffusion);
		X(12, i) = std::min(X(12, i), _lambda_max_diffusion);
		X(13, i) = std::min(X(13, i), _lambda_max_diffusion);

		// Tensor 3
		X(17, i) = std::max(X(17, i), _lambda_min_fast_diffusion);
		X(18, i) = std::max(X(18, i), _lambda_min_fast_diffusion);
		X(19, i) = std::max(X(19, i), _lambda_min_slow_diffusion);
		X(20, i) = std::max(X(20, i), _lambda_min_slow_diffusion);

		X(17, i) = std::min(X(17, i), _lambda_max_diffusion);
		X(18, i) = std::min(X(18, i), _lambda_max_diffusion);
		X(19, i) = std::min(X(19, i), _lambda_max_diffusion);
		X(20, i) = std::min(X(20, i), _lambda_max_diffusion);

		// Weights
		X(21, i) = CheckZero(X(21, i));
		X(22, i) = CheckZero(X(22, i));
		X(23, i) = CheckZero(X(23, i));

		// Free water
		X(24, i) = CheckZero(X(24, i));

		//ukfPrecisionType w1_temp = X(21, i);

		//
		for (unsigned int v = 0; v < exe_vol.rows() / 2; v += 2)
		{
			o = dir_vol.row(v);
			o_a = dir_vol.row(v + 1);
			closest(0) = cosine_similarity(m1, o);
			closest(1) = cosine_similarity(m2, o);
			closest(2) = cosine_similarity(m3, o);

			closest(3) = cosine_similarity(m1, o_a);
			closest(4) = cosine_similarity(m2, o_a);
			closest(5) = cosine_similarity(m3, o_a);

			ukfVectorType::Index maxInd;
			closest.maxCoeff(&maxInd);
			if (maxInd == 0)
			{
				o1 = o;
				max_odf(0) = ODF(exe_vol(v));
			}
			else if (maxInd == 1)
			{
				o2 = o;
				max_odf(1) = ODF(exe_vol(v));
			}
			else if (maxInd == 2)
			{
				o3 = o;
				max_odf(2) = ODF(exe_vol(v));
			}
			else if (maxInd == 3)
			{
				o1 = o_a;
				max_odf(0) = ODF(exe_vol(v + 1));
			}
			else if (maxInd == 4)
			{
				o2 = o_a;
				max_odf(1) = ODF(exe_vol(v + 1));
			}
			else if (maxInd == 5)
			{
				o3 = o_a;
				max_odf(2) = ODF(exe_vol(v + 1));
			}
		}

		denom = max_odf.sum();
		max_odf = max_odf / denom;

		x1 = 0;
		w1 = 0;
		x2 = 0;
		w2 = 0;
		x3 = 0;
		w3 = 0;

		if (max_odf(0) > 0)
		{
			x1 = BhattacharyyaCoeff(o1, m1, covMatrix.block(0, 0, 3, 3));
			w1 = BhattacharyyaCoeff(max_odf(0), X(21, i), covMatrix(21, 21));
		}
		if (max_odf(1) > 0)
		{
			x2 = BhattacharyyaCoeff(o2, m2, covMatrix.block(7, 7, 3, 3));
			w2 = BhattacharyyaCoeff(max_odf(1), X(22, i), covMatrix(22, 22));
		}
		if (max_odf(2) > 0)
		{
			x3 = BhattacharyyaCoeff(o3, m3, covMatrix.block(14, 14, 3, 3));
			w3 = BhattacharyyaCoeff(max_odf(2), X(23, i), covMatrix(23, 23));
		}

		// if (n_of_dirs == 1 && w1 < 0.1) {
		//     cout << "before X(21) " << X(21, i) << " X(22) " << X(22, i) << " X(23)  " << X(23, i) << endl;
		//     cout << "m1 before " << m1.transpose() << endl;
		//     cout << "m2 before " << m2.transpose() << endl;
		//     cout << "m3 before " << m3.transpose() << endl;
		// }

		// Average of direction from state and ridgelets for 1st tensor
		m1 = (1.0 - x1) * m1 + x1 * o1;

		// Average of direction from state and ridgelets for 2st tensor
		m2 = (1.0 - x2) * m2 + x2 * o2;

		// Average of direction from state and ridgelets for 3st tensor
		m3 = (1.0 - x3) * m3 + x3 * o3;

		X(21, i) = (1.0 - w1) * X(21, i) + w1 * max_odf(0);
		X(22, i) = (1.0 - w2) * X(22, i) + w2 * max_odf(1);
		X(23, i) = (1.0 - w3) * X(23, i) + w3 * max_odf(2);

		// vec3_t test1;
		// test1(0) = 0.5;
		// test1(1) = 0.5;
		// test1(2) = 0.5;
		// test1 = test1 / test1.norm();
		// vec3_t test2;
		// test2(0) = -0.3;
		// test2(1) = 0.1;
		// test2(2) = -test1(2);
		// test2 = test2 / test2.norm();
		// ukfMatrixType cov2;
		// cov2.resize(3, 3);
		// cov2 = covMatrix.block(0, 0, 3, 3);

		// //cout << "test bhat " << BhattacharyyaCoeff(test1, test2, covMatrix.block(0, 0, 3, 3), cov2) << std::endl;
		// cout << "vec 1 " << test1.transpose() << " vec 2 " << test2.transpose() << endl;
		// cout << "test bhat " << BhattacharyyaCoeff(test1, test2, covMatrix.block(0, 0, 3, 3)) << " cov\n " << covMatrix.block(0, 0, 3, 3);
		// exit(0);

		//hack
		// if (X(21, i) < 0.05)
		// {
		//     vec3_t orthogonal = m2.cross(m3);
		//     m1 = orthogonal / orthogonal.norm();
		// }
		// else if (X(22, i) < 0.05)
		// {
		//     vec3_t orthogonal = m1.cross(m3);
		//     m2 = orthogonal / orthogonal.norm();
		// }
		// else if (X(23, i) < 0.05)
		// {
		//     vec3_t orthogonal = m1.cross(m2);
		//     m3 = orthogonal / orthogonal.norm();
		// }

		m1.normalize();
		m2.normalize();
		m3.normalize();

		X(0, i) = m1(0);
		X(1, i) = m1(1);
		X(2, i) = m1(2);

		X(7, i) = m2(0);
		X(8, i) = m2(1);
		X(9, i) = m2(2);

		X(14, i) = m3(0);
		X(15, i) = m3(1);
		X(16, i) = m3(2);

		// Weights
		denom = X(21, i) + X(22, i) + X(23, i);
		X(21, i) = X(21, i) / denom;
		X(22, i) = X(22, i) / denom;
		X(23, i) = X(23, i) / denom;

		X(21, i) = CheckZero(X(21, i));
		X(22, i) = CheckZero(X(22, i));
		X(23, i) = CheckZero(X(23, i));

		// ukfPrecisionType degre = std::acos(std::min(std::max(m1.dot(m_temp) / (m1.norm() * m_temp.norm()), -1.0), 1.0)) * 180 / Pi;

		// if (X(22, i) < X(21, i) && X(22, i) < X(23, i) && n_of_dirs != 1)
		// {
		//     cout << "dot " << m1.dot(m_temp) << endl;
		//     cout << "normed dot " << m1.dot(m_temp) / (m1.norm() * m_temp.norm()) << endl;
		//     cout << "after n dirs " << n_of_dirs << " w1 " << X(21, i) << " w1 prev " << w1_temp << endl;
		//     cout << "after " << " w2 " << X(22, i) << " w3 " << X(23, i) << endl;
		//     cout << "x1 " << x1 << " x2 " << x2 << " x3 " << x3 << endl;
		//     cout << "dir_vol \n " << dir_vol << endl;
		//     cout << "m1 before " << m_temp.transpose() << endl;
		//     cout << "m1 after " << m1.transpose() << endl;
		//     cout << "o1 " << o1.transpose() << endl;
		//     cout << "o2 " << o2.transpose() << endl;
		//     cout << "o3 " << o3.transpose() << endl;
		// }

		// if (X(21, i) + X(22, i) + X(23, i) > 1)
		// {
		//     ukfPrecisionType denom = X(21, i) + X(22, i) + X(23, i);
		//     X(21, i) = X(21, i) / denom;
		//     X(22, i) = X(22, i) / denom;
		//     X(23, i) = X(23, i) / denom;
		//     if (X(21, i) + X(22, i) + X(23, i) > 1)
		//     {
		//         cout << "sum " << X(21, i) + X(22, i) + X(23, i) << " w's " << X(21, i) << " " << X(22, i) << " " << X(23, i) << endl;
		//     }
		// }
	} //for X.cols()
}

ukfPrecisionType Ridg_BiExp_FW::cosine_similarity(vec3_t &First, vec3_t &Second) const
{
	ukfPrecisionType dot = First.dot(Second);
	ukfPrecisionType den_a = First.norm();
	ukfPrecisionType den_b = Second.norm();

	if (den_a == 0.0 || den_b == 0.0)
	{
		throw std::logic_error(
			"cosine similarity is not defined whenever one or both "
			"input vectors are zero-vectors.");
	}

	return dot / (den_a * den_b);
}

void Ridg_BiExp_FW::F(ukfMatrixType &X) const
{
	// Identity version of state-transition function
	assert(_signal_dim > 0);
	assert(X.rows() == static_cast<unsigned int>(_state_dim) &&
		   (X.cols() == static_cast<unsigned int>(2 * _state_dim + 1) ||
			X.cols() == 1));
	for (unsigned int i = 0; i < X.cols(); ++i)
	{
		// Directions
		ukfPrecisionType norm_inv = ukfZero;
		norm_inv += X(0, i) * X(0, i);
		norm_inv += X(1, i) * X(1, i);
		norm_inv += X(2, i) * X(2, i);

		norm_inv = ukfOne / sqrt(norm_inv);
		X(0, i) *= norm_inv;
		X(1, i) *= norm_inv;
		X(2, i) *= norm_inv;

		norm_inv = ukfZero;
		norm_inv += X(7, i) * X(7, i);
		norm_inv += X(8, i) * X(8, i);
		norm_inv += X(9, i) * X(9, i);

		norm_inv = ukfOne / sqrt(norm_inv);
		X(7, i) *= norm_inv;
		X(8, i) *= norm_inv;
		X(9, i) *= norm_inv;

		norm_inv = ukfZero;
		norm_inv += X(14, i) * X(14, i);
		norm_inv += X(15, i) * X(15, i);
		norm_inv += X(16, i) * X(16, i);

		norm_inv = ukfOne / sqrt(norm_inv);
		X(14, i) *= norm_inv;
		X(15, i) *= norm_inv;
		X(16, i) *= norm_inv;

		// Lambdas
		X(3, i) = std::max(X(3, i), _lambda_min_fast_diffusion);
		X(4, i) = std::max(X(4, i), _lambda_min_fast_diffusion);
		X(5, i) = std::max(X(5, i), _lambda_min_slow_diffusion);
		X(6, i) = std::max(X(6, i), _lambda_min_slow_diffusion);

		X(3, i) = std::min(X(3, i), _lambda_max_diffusion);
		X(4, i) = std::min(X(4, i), _lambda_max_diffusion);
		X(5, i) = std::min(X(5, i), _lambda_max_diffusion);
		X(6, i) = std::min(X(6, i), _lambda_max_diffusion);

		// Tensor 2
		X(10, i) = std::max(X(10, i), _lambda_min_fast_diffusion);
		X(11, i) = std::max(X(11, i), _lambda_min_fast_diffusion);
		X(12, i) = std::max(X(12, i), _lambda_min_slow_diffusion);
		X(13, i) = std::max(X(13, i), _lambda_min_slow_diffusion);

		X(10, i) = std::min(X(10, i), _lambda_max_diffusion);
		X(11, i) = std::min(X(11, i), _lambda_max_diffusion);
		X(12, i) = std::min(X(12, i), _lambda_max_diffusion);
		X(13, i) = std::min(X(13, i), _lambda_max_diffusion);

		// Tensor 3
		X(17, i) = std::max(X(17, i), _lambda_min_fast_diffusion);
		X(18, i) = std::max(X(18, i), _lambda_min_fast_diffusion);
		X(19, i) = std::max(X(19, i), _lambda_min_slow_diffusion);
		X(20, i) = std::max(X(20, i), _lambda_min_slow_diffusion);

		X(17, i) = std::min(X(17, i), _lambda_max_diffusion);
		X(18, i) = std::min(X(18, i), _lambda_max_diffusion);
		X(19, i) = std::min(X(19, i), _lambda_max_diffusion);
		X(20, i) = std::min(X(20, i), _lambda_max_diffusion);

		// Weights
		X(21, i) = CheckZero(X(21, i));
		X(22, i) = CheckZero(X(22, i));
		X(23, i) = CheckZero(X(23, i));

		// Free water
		X(24, i) = CheckZero(X(24, i));
	}
};

void Ridg_BiExp_FW::H(const ukfMatrixType &X,
					  ukfMatrixType &Y) const
{
	assert(_signal_dim > 0);
	assert(X.rows() == static_cast<unsigned int>(_state_dim) &&
		   (X.cols() == static_cast<unsigned int>(2 * _state_dim + 1) ||
			X.cols() == 1));
	assert(Y.rows() == static_cast<unsigned int>(_signal_dim) &&
		   (Y.cols() == static_cast<unsigned int>(2 * _state_dim + 1) ||
			Y.cols() == 1));
	assert(_signal_data);

	const stdVec_t &gradients = _signal_data->gradients();
	const ukfVectorType &b = _signal_data->GetBValues();

	for (unsigned int i = 0; i < X.cols(); ++i)
	{
		// Normalize directions.
		vec3_t m1;
		initNormalized(m1, X(0, i), X(1, i), X(2, i));
		vec3_t m2;
		initNormalized(m2, X(7, i), X(8, i), X(9, i));
		vec3_t m3;
		initNormalized(m3, X(14, i), X(15, i), X(16, i));

		// Tensor 1 lambdas
		ukfPrecisionType l11 = std::max(X(3, i), _lambda_min_fast_diffusion);
		ukfPrecisionType l12 = std::max(X(4, i), _lambda_min_fast_diffusion);
		ukfPrecisionType l13 = std::max(X(5, i), _lambda_min_slow_diffusion);
		ukfPrecisionType l14 = std::max(X(6, i), _lambda_min_slow_diffusion);

		l11 = std::min(l11, _lambda_max_diffusion);
		l12 = std::min(l12, _lambda_max_diffusion);
		l13 = std::min(l13, _lambda_max_diffusion);
		l14 = std::min(l14, _lambda_max_diffusion);

		// Tensor 2 lambdas
		ukfPrecisionType l21 = std::max(X(10, i), _lambda_min_fast_diffusion);
		ukfPrecisionType l22 = std::max(X(11, i), _lambda_min_fast_diffusion);
		ukfPrecisionType l23 = std::max(X(12, i), _lambda_min_slow_diffusion);
		ukfPrecisionType l24 = std::max(X(13, i), _lambda_min_slow_diffusion);

		l21 = std::min(l21, _lambda_max_diffusion);
		l22 = std::min(l22, _lambda_max_diffusion);
		l23 = std::min(l23, _lambda_max_diffusion);
		l24 = std::min(l24, _lambda_max_diffusion);

		// Tensor 3 lambdas
		ukfPrecisionType l31 = std::max(X(17, i), _lambda_min_fast_diffusion);
		ukfPrecisionType l32 = std::max(X(18, i), _lambda_min_fast_diffusion);
		ukfPrecisionType l33 = std::max(X(19, i), _lambda_min_slow_diffusion);
		ukfPrecisionType l34 = std::max(X(20, i), _lambda_min_slow_diffusion);

		l31 = std::min(l31, _lambda_max_diffusion);
		l32 = std::min(l32, _lambda_max_diffusion);
		l33 = std::min(l33, _lambda_max_diffusion);
		l34 = std::min(l34, _lambda_max_diffusion);

		// Flip if necessary.
		// if (m1[0] < 0)
		// {
		//     m1 = -m1;
		// }
		// if (m2[0] < 0)
		// {
		//     m2 = -m2;
		// }
		// if (m3[0] < 0)
		// {
		//     m3 = -m3;
		// }

		// Get compartments weights
		const ukfPrecisionType w1 = CheckZero(X(21, i));
		const ukfPrecisionType w2 = CheckZero(X(22, i));
		const ukfPrecisionType w3 = CheckZero(X(23, i));

		// Get free water weight from state
		const ukfPrecisionType w = CheckZero(X(24, i));

		// Fill in lambdas matricies
		diagmat3_t lambdas11, lambdas12, lambdas21, lambdas22, lambdas31, lambdas32;
		lambdas11.diagonal()[0] = l11;
		lambdas11.diagonal()[1] = l12;
		lambdas11.diagonal()[2] = l12;

		lambdas12.diagonal()[0] = l13;
		lambdas12.diagonal()[1] = l14;
		lambdas12.diagonal()[2] = l14;

		lambdas21.diagonal()[0] = l21;
		lambdas21.diagonal()[1] = l22;
		lambdas21.diagonal()[2] = l22;

		lambdas22.diagonal()[0] = l23;
		lambdas22.diagonal()[1] = l24;
		lambdas22.diagonal()[2] = l24;

		lambdas31.diagonal()[0] = l31;
		lambdas31.diagonal()[1] = l32;
		lambdas31.diagonal()[2] = l32;

		lambdas32.diagonal()[0] = l33;
		lambdas32.diagonal()[1] = l34;
		lambdas32.diagonal()[2] = l34;

		// Calculate diffusion matrix.
		const mat33_t &D1 = diffusion(m1, lambdas11);
		const mat33_t &D1t = diffusion(m1, lambdas12);
		const mat33_t &D2 = diffusion(m2, lambdas21);
		const mat33_t &D2t = diffusion(m2, lambdas22);
		const mat33_t &D3 = diffusion(m3, lambdas31);
		const mat33_t &D3t = diffusion(m3, lambdas32);

		ukfPrecisionType _w_slow_diffusion = 1.0 - _w_fast_diffusion;
		ukfPrecisionType _not_w = 1.0 - w;
		// Reconstruct signal by the means of the model
		for (int j = 0; j < _signal_dim; ++j)
		{
			// u = gradient direction considered
			const vec3_t &u = gradients[j];

			Y(j, i) =
				_not_w * (w1 * (_w_fast_diffusion * std::exp(-b[j] * u.dot(D1 * u)) + _w_slow_diffusion * std::exp(-b[j] * u.dot(D1t * u))) +
						  w2 * (_w_fast_diffusion * std::exp(-b[j] * u.dot(D2 * u)) + _w_slow_diffusion * std::exp(-b[j] * u.dot(D2t * u))) +
						  w3 * (_w_fast_diffusion * std::exp(-b[j] * u.dot(D3 * u)) + _w_slow_diffusion * std::exp(-b[j] * u.dot(D3t * u)))) +
				w * std::exp(-b[j] * u.dot(m_D_iso * u));
		}
	}
}

void Ridg_BiExp_FW::State2Tensor3T(const State &x, const vec3_t &old_m, vec3_t &m1, vec3_t &l1, vec3_t &m2, vec3_t &l2, vec3_t &m3, vec3_t &l3)
{
	// Orientations;
	initNormalized(m1, x[0], x[1], x[2]);
	initNormalized(m2, x[7], x[8], x[9]);
	initNormalized(m3, x[14], x[15], x[16]);

	// Tensor 1
	// Lambda fast diffusion
	l1[0] = std::max(x(3), _lambda_min_fast_diffusion);
	l1[1] = std::max(x(4), _lambda_min_fast_diffusion);
	l1[2] = l1[1];

	// Tensor 2
	// Lambda fast diffusion
	l2[0] = std::max(x(10), _lambda_min_fast_diffusion);
	l2[1] = std::max(x(11), _lambda_min_fast_diffusion);
	l2[2] = l2[1];

	// Tensor 3
	// Lambda fast diffusion
	l3[0] = std::max(x(17), _lambda_min_fast_diffusion);
	l3[1] = std::max(x(18), _lambda_min_fast_diffusion);
	l3[2] = l3[1];

	// Flip orientations if necessary.
	if (m1[0] * old_m[0] + m1[1] * old_m[1] + m1[2] * old_m[2] < 0)
	{
		m1 = -m1;
	}
	if (m2[0] * old_m[0] + m2[1] * old_m[1] + m2[2] * old_m[2] < 0)
	{
		m2 = -m2;
	}
	if (m3[0] * old_m[0] + m3[1] * old_m[1] + m3[2] * old_m[2] < 0)
	{
		m3 = -m3;
	}
}

void Ridg_BiExp_FW::State2Tensor3T(const State &x, const vec3_t &old_m, vec3_t &m1, vec3_t &m2, vec3_t &m3)
{
	// Orientations;
	initNormalized(m1, x[0], x[1], x[2]);
	initNormalized(m2, x[7], x[8], x[9]);
	initNormalized(m3, x[14], x[15], x[16]);

	// Flip orientations if necessary.
	if (m1.dot(old_m) < 0)
	{
		m1 = -m1;
	}
	if (m2.dot(old_m) < 0)
	{
		m2 = -m2;
	}
	if (m3.dot(old_m) < 0)
	{
		m3 = -m3;
	}
}
