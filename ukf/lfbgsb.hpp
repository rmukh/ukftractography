/*
  ################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
  ##
  ##   The part of the code of this file is part of the OptimLib C++ library.
  ##   The original code you can find:
  ##   https://github.com/kthohr/optim/blob/48c657ae29a0daf17ba1a363b461a28717c97cfd/src/line_search/more_thuente.cpp
  ##   https://github.com/kthohr/optim/blob/48c657ae29a0daf17ba1a363b461a28717c97cfd/src/unconstrained/lbfgs.cpp
  ##   The code was changed by Rinat Mukhometzianov, 2019 (C)
  ##   
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################
  */

/* 
 * based on the paper
 * A LIMITED MEMORY ALGORITHM FOR BOUND CONSTRAINED OPTIMIZATION
 * (Byrd, Lu, Nocedal, Zhu)
 */
/*
* Redesigned, improved, and integrated by Rinat Mukhometzianov, 2019
*/

#ifndef LBFGSB_H_
#define LBFGSB_H_

#include "ukf_types.h"
#include "linalg.h"
#include "filter_model.h"

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

class SignalModel;

class LFBGSB
{

public:
    ukfVectorType XOpt;

    ukfVectorType _fixed_params;
    ukfVectorType _signal;
    ukfVectorType lb, ub;

    LFBGSB(SignalModel *model) : tol(1e-12), maxIter(2000), m(10), wolfe1(1e-04), wolfe2(0.9), local_model(model), EPS(2.2204e-16) {}

    /* helper functions */

    void setPhase(unsigned val)
    {
        phase = val;
    }

    void setSignal(const ukfVectorType &val)
    {
        _signal = val;
    }

    void setFixed(const ukfVectorType &val)
    {
        _fixed_params = val;
    }

    void setLowerBound(const ukfVectorType &val)
    {
        lb = val;
    }

    void setUpperBound(const ukfVectorType &val)
    {
        ub = val;
    }

    /* Main 'computational' functions */
    void computeError(const ukfMatrixType &signal_estimate, const ukfVectorType &signal, ukfPrecisionType &err)
    {
        ukfPrecisionType sum = 0.0;
        ukfPrecisionType norm_sq_signal = 0.0;
        unsigned int N = signal.size() / 2;

        for (unsigned int i = 0; i < N; ++i)
        {
            ukfPrecisionType diff = signal(i) - signal_estimate(i, 0);
            sum += diff * diff;
            norm_sq_signal += signal(i) * signal(i);
        }

        err = sum / norm_sq_signal;
    }

    ukfPrecisionType functionValue(const ukfVectorType &x)
    {
        ukfPrecisionType residual = 0.0;

        // Convert the parameter to the ukfMtarixType
        ukfVectorType localState(x.size() + _fixed_params.size());
        if (phase == 1)
        {
            localState(0) = _fixed_params(0);
            localState(1) = _fixed_params(1);
            localState(2) = _fixed_params(2);
            localState(7) = _fixed_params(3);
            localState(8) = _fixed_params(4);
            localState(9) = _fixed_params(5);
            localState(14) = _fixed_params(6);
            localState(15) = _fixed_params(7);
            localState(16) = _fixed_params(8);
            localState(21) = _fixed_params(9);
            localState(22) = _fixed_params(10);
            localState(23) = _fixed_params(11);

            localState(3) = x(0);
            localState(4) = x(1);
            localState(5) = x(2);
            localState(6) = x(3);
            localState(10) = x(4);
            localState(11) = x(5);
            localState(12) = x(6);
            localState(13) = x(7);
            localState(17) = x(8);
            localState(18) = x(9);
            localState(19) = x(10);
            localState(20) = x(11);
            localState(24) = x(12);
        }
        else if (phase == 2)
        {
            localState(0) = _fixed_params(0);
            localState(1) = _fixed_params(1);
            localState(2) = _fixed_params(2);
            localState(3) = _fixed_params(3);
            localState(4) = _fixed_params(4);
            localState(5) = _fixed_params(5);
            localState(6) = _fixed_params(6);
            localState(7) = _fixed_params(7);
            localState(8) = _fixed_params(8);
            localState(9) = _fixed_params(9);
            localState(10) = _fixed_params(10);
            localState(11) = _fixed_params(11);
            localState(12) = _fixed_params(12);
            localState(13) = _fixed_params(13);
            localState(14) = _fixed_params(14);
            localState(15) = _fixed_params(15);
            localState(16) = _fixed_params(16);
            localState(17) = _fixed_params(17);
            localState(18) = _fixed_params(18);
            localState(19) = _fixed_params(19);
            localState(20) = _fixed_params(20);
            localState(24) = _fixed_params(21);

            localState(21) = x(0);
            localState(22) = x(1);
            localState(23) = x(2);
        }
        else
        {
            std::cout << "You have specified incorrect phase!";
            throw;
        }

        // Estimate the signal
        ukfMatrixType estimatedSignal(_signal.size(), 1);

        local_model->H(localState, estimatedSignal);

        // Compute the error between the estimated signal and the acquired one
        ukfPrecisionType err = 0.0;
        computeError(estimatedSignal, _signal, err);
        //cout << err << " ";

        // Return the result
        residual = err;
        return residual;
    }

    void functionGradientMSE(const ukfVectorType &x, ukfVectorType &grad)
    {
        // We use numerical derivative
        // slope = [f(x+h) - f(x-h)] / (2h)

        unsigned int x_size = x.size();

        ukfVectorType p_h(x_size);  // for f(x+h)
        ukfVectorType p_hh(x_size); // for f(x-h)

        // The size of the derivative is not set by default,
        // so we have to do it manually
        grad.conservativeResize(x_size);

        // Set parameters
        p_h = x;
        p_hh = x;

        //original version
        for (unsigned it = 0; it < x_size; ++it)
        {
            // Optimal h is sqrt(epsilon machine) * x
            double h = std::sqrt(2.2204e-16) * std::max(std::abs(x(it)), 1e-7);

            // Volatile, otherwise compiler will optimize the value for dx
            volatile double xph = x(it) + h;

            // For taking into account the rounding error
            double dx = xph - x(it);

            // Compute the slope
            p_h(it) = xph;

            grad(it) = (functionValue(p_h) - functionValue(p_hh)) / dx;

            // Set parameters back for next iteration
            p_h(it) = x(it);
            p_hh(it) = x(it);
        }
    }

    ukfPrecisionType objFunc(ukfVectorType &x, ukfVectorType &grad)
    {
        ukfVectorType x_inv;
        invTransform(x, x_inv);

        ukfVectorType vals_grad;
        ukfVectorType jacobian;

        functionGradientMSE(x_inv, vals_grad);
        JacobAdjust(x, jacobian);

        grad = jacobian.array() * vals_grad.array();

        return functionValue(x_inv);
    }

    // Supremum norm
    ukfPrecisionType sup_norm(const ukfPrecisionType a, const ukfPrecisionType b, const ukfPrecisionType c)
    {
        return std::max(std::max(std::abs(a), std::abs(b)), std::abs(c));
    }

    // Update the interval of uncertainty
    unsigned interv_uncert(ukfPrecisionType &st_best, ukfPrecisionType &f_best, ukfPrecisionType &d_best,
                           ukfPrecisionType &st_other, ukfPrecisionType &f_other, ukfPrecisionType &d_other,
                           ukfPrecisionType &step, ukfPrecisionType &f_step, ukfPrecisionType &d_step,
                           bool &bracket, ukfPrecisionType step_min, ukfPrecisionType step_max)
    {
        bool bound = false;
        unsigned info = 0;
        ukfPrecisionType sgnd = d_step * (d_best / std::abs(d_best));

        ukfPrecisionType theta, s, gamma, p, q, r, step_c, step_q, step_f;

        if (f_step > f_best)
        {
            info = 1;
            bound = true;

            theta = 3 * (f_best - f_step) / (step - st_best) + d_best + d_step;
            s = sup_norm(theta, d_best, d_step);

            gamma = s * std::sqrt(std::pow(theta / s, 2) - (d_best / s) * (d_step / s));
            if (step < st_best)
            {
                gamma = -gamma;
            }

            p = (gamma - d_best) + theta;
            q = ((gamma - d_best) + gamma) + d_step;
            r = p / q;

            step_c = st_best + r * (step - st_best);
            step_q = st_best + ((d_best / ((f_best - f_step) / (step - st_best) + d_best)) / 2.0) * (step - st_best);

            if (std::abs(step_c - st_best) < std::abs(step_q - st_best))
            {
                step_f = step_c;
            }
            else
            {
                step_f = step_c + (step_q - step_c) / 2;
            }

            bracket = true;
        }
        else if (sgnd < 0.0)
        {
            info = 2;
            bound = false;

            theta = 3 * (f_best - f_step) / (step - st_best) + d_best + d_step;
            s = sup_norm(theta, d_best, d_step);

            gamma = s * std::sqrt(std::pow(theta / s, 2) - (d_best / s) * (d_step / s));
            if (step > st_best)
            {
                gamma = -gamma;
            }

            p = (gamma - d_step) + theta;
            q = ((gamma - d_step) + gamma) + d_best;
            r = p / q;

            step_c = step + r * (st_best - step);
            step_q = step + (d_step / (d_step - d_best)) * (st_best - step);

            if (std::abs(step_c - step) > std::abs(step_q - step))
            {
                step_f = step_c;
            }
            else
            {
                step_f = step_q;
            }

            bracket = true;
        }
        else if (std::abs(d_step) < std::abs(d_best))
        {
            info = 3;
            bound = true;

            theta = 3 * (f_best - f_step) / (step - st_best) + d_best + d_step;
            s = sup_norm(theta, d_best, d_step);

            gamma = s * std::sqrt(std::max(0.0, std::pow(theta / s, 2) - (d_best / s) * (d_step / s)));
            if (step > st_best)
            {
                gamma = -gamma;
            }

            p = (gamma - d_step) + theta;
            q = (gamma + (d_best - d_step)) + gamma;
            r = p / q;

            if (r < 0.0 && gamma != 0.0)
            {
                step_c = step + r * (st_best - step);
            }
            else if (step > st_best)
            {
                step_c = step_max;
            }
            else
            {
                step_c = step_min;
            }

            step_q = step + (d_step / (d_step - d_best)) * (st_best - step);

            if (bracket)
            {
                if (std::abs(step - step_c) < std::abs(step - step_q))
                {
                    step_f = step_c;
                }
                else
                {
                    step_f = step_q;
                }
            }
            else
            {
                if (std::abs(step - step_c) > std::abs(step - step_q))
                {
                    step_f = step_c;
                }
                else
                {
                    step_f = step_q;
                }
            }
        }
        else
        {
            info = 4;
            bound = false;

            if (bracket)
            {
                theta = 3 * (f_step - f_other) / (st_other - step) + d_other + d_step;
                s = sup_norm(theta, d_other, d_step);

                gamma = s * std::sqrt(std::pow(theta / s, 2) - (d_other / s) * (d_step / s));
                if (step > st_other)
                {
                    gamma = -gamma;
                }

                p = (gamma - d_step) + theta;
                q = ((gamma - d_step) + gamma) + d_other;
                r = p / q;

                step_c = step + r * (st_other - step);
                step_f = step_c;
            }
            else if (step > st_best)
            {
                step_f = step_max;
            }
            else
            {
                step_f = step_min;
            }
        }

        // Actually perform update of the interval
        if (f_step > f_best)
        {
            st_other = step;
            f_other = f_step;
            d_other = d_step;
        }
        else
        {
            if (sgnd < 0.0)
            {
                st_other = st_best;
                f_other = f_best;
                d_other = d_best;
            }

            st_best = step;
            f_best = f_step;
            d_best = d_step;
        }

        // Compute new step
        step_f = std::min(step_max, step_f);
        step_f = std::max(step_min, step_f);
        step = step_f;

        if (bracket && bound)
        {
            if (st_other > st_best)
            {
                step = std::min(st_best + 0.66 * (st_other - st_best), step);
            }
            else
            {
                step = std::max(st_best + 0.66 * (st_other - st_best), step);
            }
        }

        return info;
    }

    // Using linesearch to determine step width
    // x start in x
    ukfPrecisionType LineSearch(ukfVectorType &x, ukfVectorType &grad, ukfVectorType &dir)
    {
        // Reimplemented from MINPACK Fortran utility and Matlab's port of MINPACK
        ukfPrecisionType step = 1.0;
        const unsigned iter_max = 100;

        const ukfPrecisionType step_min = 0.0;
        const ukfPrecisionType step_max = 10.0;
        const ukfPrecisionType x_tol = 1e-4;

        unsigned info = 0, infoc = 1;
        const ukfPrecisionType extra_delta = 4;

        ukfVectorType x_0 = x;

        ukfPrecisionType f_step = objFunc(x, grad);

        ukfPrecisionType dgrad_init = grad.dot(dir);

        if (dgrad_init >= 0)
        {
            return step;
        }

        ukfPrecisionType dgrad = dgrad_init;
        unsigned iter = 0;

        bool bracket = false, stage_1 = true;

        ukfPrecisionType f_init = f_step, dgrad_test = wolfe1 * dgrad_init;
        ukfPrecisionType width = step_max - step_min, width_old = 2 * width;

        ukfPrecisionType st_best = 0.0, f_best = f_init, dgrad_best = dgrad_init;
        ukfPrecisionType st_other = 0.0, f_other = f_init, dgrad_other = dgrad_init;

        while (1)
        {
            iter++;

            ukfPrecisionType st_min, st_max;

            if (bracket)
            {
                st_min = std::min(st_best, st_other);
                st_max = std::max(st_best, st_other);
            }
            else
            {
                st_min = st_best;
                st_max = step + extra_delta * (step - st_best);
            }

            step = std::min(std::max(step, step_min), step_max);

            if ((bracket && (step <= st_min || step >= st_max)) || iter >= iter_max - 1 || infoc == 0 || (bracket && st_max - st_min <= x_tol * st_max))
            {
                step = st_best;
            }

            x = x_0 + step * dir;
            f_step = objFunc(x, grad);
            dgrad = grad.dot(dir);
            ukfPrecisionType armijo_check_val = f_init + step * dgrad_test;

            // check stop conditions
            if ((bracket && (step <= st_min || step >= st_max)) || infoc == 0)
            {
                info = 6;
            }
            if (step == step_max && f_step <= armijo_check_val && dgrad <= dgrad_test)
            {
                info = 5;
            }
            if (step == step_min && (f_step > armijo_check_val || dgrad >= dgrad_test))
            {
                info = 4;
            }
            if (iter >= iter_max)
            {
                info = 3;
            }
            if (bracket && st_max - st_min <= x_tol * st_max)
            {
                info = 2;
            }

            // strong Wolfe conditions
            if (f_step <= armijo_check_val && std::abs(dgrad) <= wolfe2 * (-dgrad_init))
            {
                info = 1;
            }

            if (info != 0)
            {
                return step;
            }

            if (stage_1 && f_step <= armijo_check_val && dgrad >= std::min(wolfe1, wolfe2) * dgrad_init)
            {
                stage_1 = false;
            }

            if (stage_1 && f_step <= f_best && f_step > armijo_check_val)
            {
                double f_mod = f_step - step * dgrad_test;
                double f_best_mod = f_best - st_best * dgrad_test;
                double f_other_mod = f_other - st_other * dgrad_test;

                double dgrad_mod = dgrad - dgrad_test;
                double dgrad_best_mod = dgrad_best - dgrad_test;
                double dgrad_other_mod = dgrad_other - dgrad_test;

                infoc = interv_uncert(st_best, f_best_mod, dgrad_best_mod, st_other, f_other_mod, dgrad_other_mod, step, f_mod, dgrad_mod, bracket, st_min, st_max);

                f_best = f_best_mod + st_best * dgrad_test;
                f_other = f_other_mod + st_other * dgrad_test;

                dgrad_best = dgrad_best_mod + dgrad_test;
                dgrad_other = dgrad_other_mod + dgrad_test;
            }
            else
            {
                infoc = interv_uncert(st_best, f_best, dgrad_best, st_other, f_other, dgrad_other, step, f_step, dgrad, bracket, st_min, st_max);
            }

            if (bracket)
            {
                if (std::abs(st_other - st_best) >= 0.66 * width_old)
                {
                    step = st_best + 0.5 * (st_other - st_best);
                }

                width_old = width;
                width = std::abs(st_other - st_best);
            }
        }

        return step;
    }

    void JacobAdjust(ukfVectorType &x, ukfVectorType &output)
    {
        Eigen::Array<ukfPrecisionType, Dynamic, 1> x_exp = x.array().exp();
        output = x_exp * (ub - lb).array() / (x_exp + 1).pow(2);
    }

    void transform(ukfVectorType &in, ukfVectorType &out)
    {
        out.resizeLike(in);
        out = ((in - lb).array() + EPS).log() - ((ub - in).array() + EPS).log();
    }

    void invTransform(ukfVectorType &in, ukfVectorType &out)
    {
        const unsigned DIM = in.size();
        out.resizeLike(in);

        for (unsigned i = 0; i < DIM; ++i)
        {
            if (!std::isfinite(in(i)))
            {
                if (std::isnan(in(i)))
                {
                    out(i) = (ub(i) - lb(i)) / 2.0;
                }
                else if (in(i) < 0.0)
                {
                    out(i) = lb(i) + EPS;
                }
                else
                {
                    out(i) = ub(i) - EPS;
                }
            }
            else
            {
                ukfPrecisionType in_exp = std::exp(in(i));
                out(i) = (lb(i) + EPS + (ub(i) - EPS) * in_exp) / (1.0 + in_exp);

                if (!std::isfinite(out(i)))
                {
                    out(i) = ub(i) - EPS;
                }
            }
        }
    }

    void step(const ukfVectorType &g, ukfMatrixType &s_mat, ukfMatrixType &y_mat, const unsigned M, ukfVectorType &r)
    {
        ukfVectorType q(g);
        ukfVectorType alpha;
        alpha.resize(M);

        for (unsigned i = 0; i < M; i++)
        {
            ukfPrecisionType rho = 1.0 / y_mat.col(i).dot(s_mat.col(i));
            alpha(i) = rho * s_mat.col(i).dot(q);

            q -= alpha(i) * y_mat.col(i);
        }

        r = q * (s_mat.col(0).dot(y_mat.col(0))) / y_mat.col(0).dot(y_mat.col(0));

        for (int i = M - 1; i >= 0; i--)
        {
            ukfPrecisionType rho = 1.0 / y_mat.col(i).dot(s_mat.col(i));
            ukfPrecisionType beta = rho * y_mat.col(i).dot(r);

            r += (alpha(i) - beta) * s_mat.col(i);
        }
    }

    void Solve(ukfVectorType &x0)
    {
        Assert(x0.rows() == lb.rows(), "lower bound size incorrect");
        Assert(x0.rows() == ub.rows(), "upper bound size incorrect");

        XOpt.resize(x0.size());
        const unsigned DIM = x0.rows();

        ukfVectorType x;
        transform(x0, x);

        // Gradient vector
        ukfVectorType g;
        objFunc(x, g);

        double err = g.norm();
        if (err <= tol)
        {
            XOpt = x0;
            return;
        }

        // Search direction
        ukfVectorType d = -g;

        ukfVectorType x_prev = x;
        ukfVectorType g_prev = g;

        LineSearch(x_prev, g_prev, d);

        err = g.norm();
        if (err <= tol)
        {
            XOpt = x_prev;
            return;
        }

        ukfVectorType s = x_prev - x;
        ukfVectorType y = g_prev - g;

        // Declare and init matricies and vectors
        ukfMatrixType sHistory;
        sHistory.resize(DIM, m);
        ukfMatrixType yHistory;
        yHistory.resize(DIM, m);

        sHistory.col(0) = s;
        yHistory.col(0) = y;

        g = g_prev;

        // Init loop variables
        unsigned k = 0;
        ukfVectorType r;

        // Start loop
        while (err > tol && k < maxIter)
        {
            k++;

            step(g, sHistory, yHistory, std::min(k, m), r);
            d = -r;

            LineSearch(x_prev, g_prev, d);

            // Stop searching minimum if L2-norm became less than user defined tolerance
            err = g_prev.norm();

            if (err <= tol)
                break;

            if (g_prev.array().isNaN().any())
            {
                x_prev = x;
                break;
            }

            s = x_prev - x;
            y = g_prev - g;

            err = s.norm();

            x = x_prev;
            g = g_prev;

            sHistory.rightCols(m - 1) = sHistory.leftCols(m - 2).eval();
            yHistory.rightCols(m - 1) = yHistory.leftCols(m - 2).eval();

            sHistory.col(0) = s;
            yHistory.col(0) = y;
        }

        invTransform(x_prev, XOpt);
    }

private:
    ukfPrecisionType tol;
    unsigned maxIter;
    unsigned m;
    ukfPrecisionType wolfe1;
    ukfPrecisionType wolfe2;
    const SignalModel *const local_model;
    ukfPrecisionType EPS;
    unsigned phase;
};

#endif /* LBFGSB_H_ */
