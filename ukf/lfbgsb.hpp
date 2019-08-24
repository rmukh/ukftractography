/**
 *           c++11-only implementation of the L-BFGS-B algorithm
 *
 * Copyright (c) 2014 Patrick Wieschollek
 *               https://github.com/PatWie/LBFGSB
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/* 
 * based on the paper
 * A LIMITED MEMORY ALGORITHM FOR BOUND CONSTRAINED OPTIMIZATION
 * (Byrd, Lu, Nocedal, Zhu)
 */
/*
   Redesigned, improved, integrated  by Rinat Mukhometzianov, 2019
*/

#ifndef LBFGSB_H_
#define LBFGSB_H_

#include "ukf_types.h"
#include "linalg.h"

#include <stdexcept>
#include <cmath>
#include <list>
#include <stdio.h>
#include <iostream>
#include <functional>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#define INF HUGE_VAL
#define Assert(x, m)                  \
    if (!(x))                         \
    {                                 \
        throw(std::runtime_error(m)); \
    }

class LFBGSB
{
    ukfMatrixType W, M;
    ukfVectorType lb, ub;

public:
    ukfVectorType XOpt;

    ukfVectorType _fixed_params;
    ukfVectorType _signal;
    const ukfPrecisionType EPS = 2.2204e-16;

    LFBGSB(const ukfVectorType &l, const ukfVectorType &u, const stdVec_t &grads, const ukfVectorType &b, const mat33_t &diso, ukfPrecisionType w_fast)
        : lb(l), ub(u), tol(1e-10), maxIter(500), m(10), theta(1.0), gradients(grads), b_vals(b), m_D_iso(diso), _w_fast_diffusion(w_fast), line_search_flag(true)
    {
        W = ukfMatrixType::Zero(l.rows(), 0);
        M = ukfMatrixType::Zero(0, 0);
    }

    std::vector<int> sort_indexes(const std::vector<std::pair<int, ukfPrecisionType>> &v)
    {
        std::vector<int> idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i)
            idx[i] = v[i].first;
        sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1].second < v[i2].second; });
        return idx;
    }

    void H(const ukfVectorType &X, ukfVectorType &Y)
    {
        // Normalize directions.
        vec3_t m1;
        initNormalized(m1, X(0), X(1), X(2));
        vec3_t m2;
        initNormalized(m2, X(7), X(8), X(9));
        vec3_t m3;
        initNormalized(m3, X(14), X(15), X(16));

        // Tensor 1 lambdas
        ukfPrecisionType l11 = X(3);
        ukfPrecisionType l12 = X(4);
        ukfPrecisionType l13 = X(5);
        ukfPrecisionType l14 = X(6);

        // Tensor 2 lambdas
        ukfPrecisionType l21 = X(10);
        ukfPrecisionType l22 = X(11);
        ukfPrecisionType l23 = X(12);
        ukfPrecisionType l24 = X(13);

        // Tensor 3 lambdas
        ukfPrecisionType l31 = X(17);
        ukfPrecisionType l32 = X(18);
        ukfPrecisionType l33 = X(19);
        ukfPrecisionType l34 = X(20);

        // Get compartments weights
        const ukfPrecisionType w1 = X(21);
        const ukfPrecisionType w2 = X(22);
        const ukfPrecisionType w3 = X(23);

        // Get free water weight from state
        const ukfPrecisionType w = X(24);

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
        for (int j = 0; j < _signal.size(); ++j)
        {
            // u = gradient direction considered
            const vec3_t &u = gradients[j];

            Y(j) =
                _not_w * (w1 * (_w_fast_diffusion * std::exp(-b_vals[j] * u.dot(D1 * u)) + _w_slow_diffusion * std::exp(-b_vals[j] * u.dot(D1t * u))) +
                          w2 * (_w_fast_diffusion * std::exp(-b_vals[j] * u.dot(D2 * u)) + _w_slow_diffusion * std::exp(-b_vals[j] * u.dot(D2t * u))) +
                          w3 * (_w_fast_diffusion * std::exp(-b_vals[j] * u.dot(D3 * u)) + _w_slow_diffusion * std::exp(-b_vals[j] * u.dot(D3t * u)))) +
                w * std::exp(-b_vals[j] * u.dot(m_D_iso * u));
        }
    }

    void computeError(const ukfVectorType &signal_estimate, const ukfVectorType &signal, ukfPrecisionType &err)
    {
        ukfPrecisionType sum = 0.0;
        ukfPrecisionType norm_sq_signal = 0.0;
        unsigned int N = signal.size() / 2;

        for (unsigned int i = 0; i < N; ++i)
        {
            ukfPrecisionType diff = signal[i] - signal_estimate(i, 0);
            sum += diff * diff;
            norm_sq_signal += signal[i] * signal[i];
        }

        err = sum / norm_sq_signal;
    }

    // ukfPrecisionType functionValue(const ukfVectorType &x)
    // {
    //     ukfPrecisionType residual = 0.0;

    //     // Convert the parameter to the ukfMtarixType
    //     ukfVectorType localState(x.size() + _fixed_params.size());
    //     if (1)
    //     {
    //         localState(0) = _fixed_params(0);
    //         localState(1) = _fixed_params(1);
    //         localState(2) = _fixed_params(2);
    //         localState(7) = _fixed_params(3);
    //         localState(8) = _fixed_params(4);
    //         localState(9) = _fixed_params(5);
    //         localState(14) = _fixed_params(6);
    //         localState(15) = _fixed_params(7);
    //         localState(16) = _fixed_params(8);
    //         localState(21) = _fixed_params(9);
    //         localState(22) = _fixed_params(10);
    //         localState(23) = _fixed_params(11);

    //         localState(3) = x(0);
    //         localState(4) = x(1);
    //         localState(5) = x(2);
    //         localState(6) = x(3);
    //         localState(10) = x(4);
    //         localState(11) = x(5);
    //         localState(12) = x(6);
    //         localState(13) = x(7);
    //         localState(17) = x(8);
    //         localState(18) = x(9);
    //         localState(19) = x(10);
    //         localState(20) = x(11);
    //         localState(24) = x(12);
    //     }
    //     else if (0)
    //     {
    //         localState(0) = _fixed_params(0);
    //         localState(1) = _fixed_params(1);
    //         localState(2) = _fixed_params(2);
    //         localState(3) = _fixed_params(3);
    //         localState(4) = _fixed_params(4);
    //         localState(5) = _fixed_params(5);
    //         localState(6) = _fixed_params(6);
    //         localState(7) = _fixed_params(7);
    //         localState(8) = _fixed_params(8);
    //         localState(9) = _fixed_params(9);
    //         localState(10) = _fixed_params(10);
    //         localState(11) = _fixed_params(11);
    //         localState(12) = _fixed_params(12);
    //         localState(13) = _fixed_params(13);
    //         localState(14) = _fixed_params(14);
    //         localState(15) = _fixed_params(15);
    //         localState(16) = _fixed_params(16);
    //         localState(17) = _fixed_params(17);
    //         localState(18) = _fixed_params(18);
    //         localState(19) = _fixed_params(19);
    //         localState(20) = _fixed_params(20);
    //         localState(24) = _fixed_params(21);

    //         localState(21) = x(0);
    //         localState(22) = x(1);
    //         localState(23) = x(2);
    //     }
    //     else
    //     {
    //         std::cout << "You have not specified the phase!";
    //         throw;
    //     }

    //     // Estimate the signal
    //     ukfVectorType estimatedSignal(_signal.size());

    //     H(localState, estimatedSignal);

    //     // Compute the error between the estimated signal and the acquired one
    //     ukfPrecisionType err = 0.0;
    //     computeError(estimatedSignal, _signal, err);
    //     //cout << err << " ";

    //     // Return the result
    //     residual = err;
    //     return residual;
    // }

    // void functionGradient(const ukfVectorType &x, ukfVectorType &grad)
    // {
    //     // We use numerical derivative
    //     // slope = [f(x+h) - f(x-h)] / (2h)

    //     unsigned int x_size = x.size();
    //     ukfVectorType p_h(x_size);  // for f(x+h)
    //     ukfVectorType p_hh(x_size); // for f(x-h)

    //     // The size of the derivative is not set by default,
    //     // so we have to do it manually
    //     grad.resize(x_size);

    //     // Set parameters
    //     p_h = x;
    //     p_hh = x;
    //     /*
    //     // Calculate derivative for each parameter (reference to the wikipedia page: Numerical Differentiation)
    //     for (unsigned it = 0; it < x_size; ++it)
    //     {
    //         // // Optimal h is sqrt(epsilon machine) * x
    //         // ukfPrecisionType h;
    //         // if (x(it) == 0)
    //         //     h = std::sqrt(EPS);
    //         // else
    //         //     h = std::sqrt(EPS) * x(it);

    //         // //ukfPrecisionType h = std::sqrt(EPS) * std::max(std::abs(x(it)), 1.0);

    //         // // Volatile, otherwise compiler will optimize the value for dx
    //         // volatile ukfPrecisionType xph = x(it) + h;

    //         // // For taking into account the rounding error
    //         // ukfPrecisionType dx = xph - x(it);

    //         // // Compute the slope
    //         // p_h(it) = xph;

    //         // //p_hh[it] = parameters[it] - h;
    //         // grad(it) = (functionValue(p_h) - functionValue(p_hh)) / dx;

    //         // Estimating derivatives using Richardson's Extrapolation

    //         //ukfPrecisionType h = std::sqrt(EPS) * std::max(std::abs(x(it)), 1.0);
    //         ukfPrecisionType h = 0.001;
    //         // Compute d/dx[func(*first)] using a three-point
    //         // central difference rule of O(dx^6).

    //         const ukfPrecisionType dx1 = h;
    //         const ukfPrecisionType dx2 = dx1 * 2;
    //         const ukfPrecisionType dx3 = dx1 * 3;

    //         p_h(it) = x(it) + dx1;
    //         p_hh(it) = x(it) - dx1;
    //         const ukfPrecisionType m1 = (functionValue(p_h) - functionValue(p_hh)) / 2;

    //         p_h(it) = x(it) + dx2;
    //         p_hh(it) = x(it) - dx2;
    //         const ukfPrecisionType m2 = (functionValue(p_h) - functionValue(p_hh)) / 4;

    //         p_h(it) = x(it) + dx3;
    //         p_hh(it) = x(it) - dx3;
    //         const ukfPrecisionType m3 = (functionValue(p_h) - functionValue(p_hh)) / 6;

    //         const ukfPrecisionType fifteen_m1 = 15 * m1;
    //         const ukfPrecisionType six_m2 = 6 * m2;
    //         const ukfPrecisionType ten_dx1 = 10 * dx1;

    //         grad(it) = ((fifteen_m1 - six_m2) + m3) / ten_dx1;

    //         // Set parameters back for next iteration
    //         p_h(it) = x(it);
    //         p_hh(it) = x(it);
    //     }
    //     */
    //     //original version
    //     for (unsigned it = 0; it < x_size; ++it)
    //     {
    //         // Optimal h is sqrt(epsilon machine) * x
    //         double h = std::sqrt(2.2204e-16) * std::max(std::abs(x(it)), 1e-7);

    //         // Volatile, otherwise compiler will optimize the value for dx
    //         volatile double xph = x(it) + h;

    //         // For taking into account the rounding error
    //         double dx = xph - x(it);

    //         // Compute the slope
    //         p_h[it] = xph;

    //         //p_hh[it] = parameters[it] - h;
    //         grad(it) = (functionValue(p_h) - functionValue(p_hh)) / dx;

    //         // Set parameters back for next iteration
    //         p_h(it) = x(it);
    //         p_hh(it) = x(it);
    //     }
    // }

    void functionGradient(ukfVectorType &x, ukfVectorType &grad)
    {
        grad.resize(x.size());
        grad(0) = -400 * (x(1) - x(0) * x(0)) * x(0) - 2 * (1 - x(0));
        grad(1) = 200 * (x(1) - x(0) * x(0));
    }

    ukfPrecisionType functionValue(ukfVectorType &x)
    {
        const double t1 = (1 - x[0]);
        const double t2 = (x[1] - x[0] * x[0]);
        return t1 * t1 + 100 * t2 * t2;
    }
    // Find cauchy point in x
    // start in x
    void GetGeneralizedCauchyPoint(ukfVectorType &x, ukfVectorType &g, ukfVectorType &x_cauchy, ukfVectorType &c)
    {
        const unsigned DIM = x.rows();
        // PAGE 8
        // Algorithm CP: Computation of the generalized Cauchy point
        // Given x,l,u,g, and B = \theta I-WMW

        // {all t_i} = { (idx,value), ... }
        std::vector<std::pair<int, ukfPrecisionType>> SetOfT;
        // the feasible set is implicitly given by "SetOfT - {t_i==0}"
        ukfVectorType d = -g;

        for (unsigned j = 0; j < DIM; ++j)
        {
            ukfPrecisionType tmp = 0.0;

            if (g(j) == 0.0)
            {
                SetOfT.push_back(std::make_pair(j, INF));
            }
            else
            {
                if (g(j) < 0.0)
                    tmp = (x(j) - ub(j)) / g(j);
                else if (g(j) > 0.0)
                    tmp = (x(j) - lb(j)) / g(j);

                SetOfT.push_back(std::make_pair(j, tmp));
            }

            if (tmp < EPS)
                d(j) = 0.0;
        }

        // sortedindices [1,0,2] means the minimal element is on the 1th entry
        std::vector<int> SortedIndices = sort_indexes(SetOfT); // using heapsort

        // Initialize
        x_cauchy = x;
        // p := 	W^T*p
        ukfVectorType p = W.transpose() * d; // (2mn operations)
        // c := 	0
        c = ukfVectorType::Zero(W.cols());
        // f' := 	g^T*d = -d^Td
        ukfPrecisionType f_prime = -d.dot(d); // (n operations)
        // f'' :=	\theta*d^T*d-d^T*W*M*W^T*d = -\theta*f' - p^T*M*p
        ukfPrecisionType f_doubleprime = (ukfPrecisionType)(-theta) * f_prime - p.dot(M * p); // (O(m^2) operations)
        f_doubleprime = std::max(EPS, f_doubleprime);
        ukfPrecisionType f_primezero = -theta * f_prime;
        // \delta t_min :=	-f'/f''
        ukfPrecisionType dt_min = -f_prime / f_doubleprime;
        // t_old := 	0
        ukfPrecisionType t_old = 0;

        unsigned i = 0;
        for (unsigned j = 0; j < DIM; j++)
        {
            i = j;
            if (SetOfT[SortedIndices[j]].second > 0)
                break;
        }

        // b := i s.t t_i = t
        int b = SortedIndices[i];
        // t := min{ti : i in F}
        ukfPrecisionType t = SetOfT[b].second;

        // \delta t := 	t - 0
        ukfPrecisionType dt = t - t_old;

        // examination of subsequent segments
        while (dt_min >= dt && i < DIM)
        {
            if (d(b) > 0)
                x_cauchy(b) = ub(b);
            else if (d(b) < 0)
                x_cauchy(b) = lb(b);

            // z_b := x_p^{cp} - x_b
            ukfPrecisionType zb = x_cauchy(b) - x(b);
            // c :=  c +\delta t*p
            c += dt * p;

            ukfVectorType wbt = W.row(b);

            ukfPrecisionType gb = g(b);
            f_prime += dt * f_doubleprime + gb * gb + theta * gb * zb - gb * wbt.transpose() * (M * c);
            f_doubleprime += -theta * gb * gb - 2.0 * (gb * (wbt.dot(M * p))) - gb * gb * wbt.transpose() * (M * wbt);
            f_doubleprime = std::max(EPS * f_primezero, f_doubleprime);

            p += gb * wbt.transpose();
            d(b) = 0;
            dt_min = -f_prime / f_doubleprime;
            t_old = t;
            ++i;
            if (i < DIM)
            {
                b = SortedIndices[i];
                t = SetOfT[b].second;
                dt = t - t_old;
            }
        }

        dt_min = std::max(dt_min, 0.0);
        t_old += dt_min;

        for (unsigned ii = i; ii < x_cauchy.rows(); ++ii)
        {
            x_cauchy(SortedIndices[ii]) = x(SortedIndices[ii]) + t_old * d(SortedIndices[ii]);
        }

        c += dt_min * p;
    }

    // find valid alpha for (8.5)
    // x_cp cauchy point
    // du unconstrained solution of subspace minimization
    // FreeVariables flag (1 if is free variable and 0 if is not free variable)
    ukfPrecisionType FindAlpha(ukfVectorType &x_cp, ukfVectorType &du, std::vector<int> &FreeVariables)
    {
        /* 
         * this returns
		 * a* = max {a : a <= 1 and  l_i-xc_i <= a*d_i <= u_i-xc_i}
		 */
        ukfPrecisionType alphastar = 1;
        const unsigned int n = FreeVariables.size();
        for (unsigned int i = 0; i < n; i++)
        {
            if (du(i) > 0)
                alphastar = std::min(alphastar, (ub(FreeVariables[i]) - x_cp(FreeVariables[i])) / du(i));
            else
                alphastar = std::min(alphastar, (lb(FreeVariables[i]) - x_cp(FreeVariables[i])) / du(i));
        }
        return alphastar;
    }

    ukfPrecisionType zoomAlpha(ukfVectorType &x0, ukfPrecisionType f0, ukfVectorType &g0, const ukfVectorType &p, ukfPrecisionType alpha_lo, ukfPrecisionType alpha_hi)
    {
        ukfPrecisionType c1 = 1e-4;
        ukfPrecisionType c2 = 0.9;
        unsigned i = 0;
        unsigned max_iter = 20;
        ukfPrecisionType dphi0 = g0.dot(p);
        ukfPrecisionType dphi = 0.0;

        ukfPrecisionType alpha = 0.0;
        ukfPrecisionType alpha_i = 0.0;
        ukfPrecisionType f_i = 0.0;
        ukfPrecisionType f_lo = 0.0;
        ukfVectorType x;
        ukfVectorType x_lo;
        ukfVectorType g_i;

        while (true)
        {
            alpha_i = 0.5 * (alpha_lo + alpha_hi);
            alpha = alpha_i;
            x = x0 + alpha_i * p;
            f_i = functionValue(x);

            functionGradient(x, g_i);
            x_lo = x0 + alpha_lo * p;
            f_lo = functionValue(x_lo);
            if (f_i > f0 + c1 * alpha_i * dphi0 || f_i >= f_lo)
            {
                alpha_hi = alpha_i;
            }
            else
            {
                dphi = g_i.dot(p);
                if (std::abs(dphi) <= -c2 * dphi0)
                {
                    alpha = alpha_i;
                    break;
                }
                if (dphi * (alpha_hi - alpha_lo) >= 0)
                    alpha_hi = alpha_lo;

                alpha_lo = alpha_i;
            }
            i++;
            if (i > max_iter)
            {
                alpha = alpha_i;
                break;
            }
        }
        return alpha;
    }

    ukfPrecisionType strongWolfeConditions(ukfVectorType &x0, ukfPrecisionType f0, ukfVectorType &g0, const ukfVectorType &p)
    {
        // Init all necessary variables
        ukfPrecisionType c1 = 1e-4;
        ukfPrecisionType c2 = 0.9;
        ukfPrecisionType alpha = 1.0;
        ukfPrecisionType alpha_max = 2.5;
        ukfPrecisionType alpha_im1 = 0.0;
        ukfPrecisionType alpha_i = 1.0;
        ukfPrecisionType f_im1 = f0;
        ukfPrecisionType f_i = 0;
        ukfPrecisionType dphi0 = g0.dot(p);
        ukfPrecisionType dphi = 0.0;
        unsigned i = 0;
        unsigned max_iter = 20;
        ukfVectorType x;
        ukfVectorType g_i;

        // Perfome search for the alpha value which satisfy strong Wolfe conditions
        while (true)
        {
            x = x0 + alpha_i * p;

            f_i = functionValue(x);
            functionGradient(x, g_i);
            if ((f_i > f0 + c1 * dphi0) || ((i > 1) && (f_i >= f_im1)))
            {
                alpha = zoomAlpha(x0, f0, g0, p, alpha_im1, alpha_i);
                break;
            }

            dphi = g_i.dot(p);

            if (std::abs(dphi) <= -c2 * dphi0)
            {
                alpha = alpha_i;
                break;
            }
            if (dphi >= 0)
            {
                alpha = zoomAlpha(x0, f0, g0, p, alpha_i, alpha_im1);
                break;
            }

            // update
            alpha_im1 = alpha_i;
            f_im1 = f_i;
            alpha_i = alpha_i + 0.8 * (alpha_max - alpha_i);

            if (i > max_iter)
            {
                alpha = alpha_i;
                break;
            }

            i++;
        }

        return alpha;
    }

    // Using linesearch to determine step width
    // x start in x
    // dx direction
    // f current value of objective (will be changed)
    // f current gradient of objective (will be changed)
    // t step width (will be changed)
    void LineSearch(ukfVectorType &x, ukfVectorType dx, ukfPrecisionType &f, ukfVectorType &g)
    {
        ukfPrecisionType alpha = 1.0;
        if (line_search_flag)
            alpha = strongWolfeConditions(x, f, g, dx);

        x += alpha * dx;
    }

    // direct primal approach
    void SubspaceMinimization(ukfVectorType &x_cauchy, ukfVectorType &x, ukfVectorType &c, ukfVectorType &g, ukfVectorType &SubspaceMin)
    {
        line_search_flag = true;

        // cached value: ThetaInverse=1/theta;
        ukfPrecisionType theta_inverse = 1.0 / theta;

        // size of "t"
        std::vector<int> FreeVariablesIndex;
        for (unsigned i = 0; i < x_cauchy.rows(); ++i)
        {
            if ((x_cauchy(i) != ub(i)) && (x_cauchy(i) != lb(i)))
            {
                FreeVariablesIndex.push_back(i);
            }
        }
        const unsigned FreeVarCount = FreeVariablesIndex.size();

        if (FreeVarCount == 0)
        {
            line_search_flag = false;
            SubspaceMin = x_cauchy;
            return;
        }

        ukfMatrixType WZ = ukfMatrixType::Zero(W.cols(), FreeVarCount);
        for (unsigned i = 0; i < FreeVarCount; i++)
            WZ.col(i) = W.row(FreeVariablesIndex[i]);
        ukfVectorType rr = (g + theta * (x_cauchy - x) - W * (M * c));
        // r=r(FreeVariables);
        ukfMatrixType r = ukfMatrixType::Zero(FreeVarCount, 1);
        for (unsigned i = 0; i < FreeVarCount; i++)
            r.row(i) = rr.row(FreeVariablesIndex[i]);
        // STEP 2: "v = w^T*Z*r" and STEP 3: "v = M*v"
        ukfVectorType v = M * (WZ * r);
        // STEP 4: N = 1/theta*W^T*Z*(W^T*Z)^T
        ukfMatrixType N = theta_inverse * WZ * WZ.transpose();
        // N = I - MN
        N = ukfMatrixType::Identity(N.rows(), N.rows()) - M * N;
        // STEP: 5
        // v = N^{-1}*v
        if (v.size() > 0)
            v = N.lu().solve(v);
        // STEP: 6
        // HERE IS A MISTAKE IN THE ORIGINAL PAPER!
        ukfVectorType du = -theta_inverse * r - theta_inverse * theta_inverse * WZ.transpose() * v;

        // STEP: 7
        ukfPrecisionType alpha_star = FindAlpha(x_cauchy, du, FreeVariablesIndex);

        // STEP: 8
        ukfVectorType dStar = alpha_star * du;

        SubspaceMin = x_cauchy;
        for (unsigned i = 0; i < FreeVarCount; ++i)
        {
            SubspaceMin(FreeVariablesIndex[i]) = SubspaceMin(FreeVariablesIndex[i]) + dStar(i);
        }
    }

    bool isOptimal(ukfVectorType &x, ukfVectorType &g)
    {
        ukfVectorType projGrad = x - g;

        for (int i = 0; i < x.size(); ++i)
        {
            if (projGrad(i) < lb(i))
                projGrad(i) = lb(i);
            else if (projGrad(i) > ub(i))
                projGrad(i) = ub(i);
        }

        projGrad = projGrad - x;

        return (projGrad.cwiseAbs()).maxCoeff() > tol;
    }

    void Solve(ukfVectorType &x0)
    {
        Assert(x0.rows() == lb.rows(), "lower bound size incorrect");
        Assert(x0.rows() == ub.rows(), "upper bound size incorrect");
        const unsigned DIM = x0.rows();

        ukfMatrixType yHistory = ukfMatrixType::Zero(DIM, 0);
        ukfMatrixType sHistory = ukfMatrixType::Zero(DIM, 0);

        ukfVectorType x = x0;
        ukfVectorType g;

        ukfPrecisionType f = functionValue(x);
        functionGradient(x, g);

        unsigned k = 0;
        theta = 1.0;
        W = ukfMatrixType::Zero(DIM, 1);
        M = ukfMatrixType::Zero(1, 1);

        while (isOptimal(x, g) && (k < maxIter))
        {
            double f_old = f;
            ukfVectorType x_old = x;
            ukfVectorType g_old = g;

            // STEP 2: compute the cauchy point by algorithm CP
            ukfVectorType CauchyPoint = ukfMatrixType::Zero(DIM, 1);
            ukfVectorType c = ukfMatrixType::Zero(DIM, 1);
            GetGeneralizedCauchyPoint(x, g, CauchyPoint, c);

            // STEP 3: compute a search direction d_k by the primal method
            ukfVectorType SubspaceMin;
            SubspaceMinimization(CauchyPoint, x, c, g, SubspaceMin);

            ukfMatrixType H;

            // STEP 4: perform linesearch
            LineSearch(x, SubspaceMin - x, f, g);

            // STEP 5: compute gradient of the function
            f = functionValue(x);
            functionGradient(x, g);

            // prepare for next iteration
            ukfVectorType newY = g - g_old;
            ukfVectorType newS = x - x_old;

            // STEP 6:
            ukfPrecisionType skipping = std::abs(newS.dot(newY));

            if (skipping < EPS)
            {
                k++;
                continue;
            }
            if (k < m)
            {
                yHistory.conservativeResize(DIM, k + 1);
                sHistory.conservativeResize(DIM, k + 1);
            }
            else
            {
                yHistory.leftCols(m - 1) = yHistory.rightCols(m - 1).eval();
                sHistory.leftCols(m - 1) = sHistory.rightCols(m - 1).eval();
            }
            yHistory.rightCols(1) = newY;
            sHistory.rightCols(1) = newS;

            // STEP 7:
            theta = (ukfPrecisionType)(newY.dot(newY)) / (newY.dot(newS));

            W = ukfMatrixType::Zero(yHistory.rows(), yHistory.cols() + sHistory.cols());
            W << yHistory, theta * sHistory;
            ukfMatrixType A = sHistory.transpose() * yHistory;
            ukfMatrixType L = A.triangularView<Eigen::StrictlyLower>();
            ukfMatrixType MM(A.rows() + L.rows(), A.rows() + L.cols());
            ukfMatrixType D = -1 * A.diagonal().asDiagonal();
            MM << D, L.transpose(), L, ((sHistory.transpose() * sHistory) * theta);
            M = MM.inverse();

            if (std::abs(f_old - f) < tol)
                break;

            k++;
        }

        XOpt = x;
    }

private:
    ukfPrecisionType tol;
    unsigned maxIter;
    unsigned m;
    ukfPrecisionType theta;
    const stdVec_t &gradients;
    const ukfVectorType &b_vals;
    const mat33_t &m_D_iso;
    ukfPrecisionType _w_fast_diffusion;
    bool line_search_flag;
};

#endif /* LBFGSB_H_ */