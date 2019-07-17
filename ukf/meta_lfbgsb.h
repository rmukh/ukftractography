/**
 *           c++11-only implementation of the L-BFGS-B algorithm
 *
 * Copyright (c) 2014 Patrick Wieschollek
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

#ifndef LBFGSB_DEFINITIONS_H_
#define LBFGSB_DEFINITIONS_H_

#include <stdexcept>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <vector>

struct Options
{
    double tol;
    double functol;
    double constrtol;
    int maxIter;
    int m;
    Options()
    {
        tol = 1e-9;
        functol = 1e-10;
        constrtol = 1e-2;
        maxIter = 500;
        m = 6;
    }
};

const double EPS = 2.2204e-016;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::VectorXd::Scalar Scalar;

#define INF HUGE_VAL
#define Assert(x, m)                  \
    if (!(x))                         \
    {                                 \
        throw(std::runtime_error(m)); \
    }

#define FAST

#ifdef FAST
#define Debug(x)
#else
#define Debug(x)                     \
    if (false)                       \
    {                                \
        std::cout << "DEBUG: " << x; \
        std::cout << std::endl;      \
    }
#endif

#endif /* LBFGSB_DEFINITIONS_H_ */