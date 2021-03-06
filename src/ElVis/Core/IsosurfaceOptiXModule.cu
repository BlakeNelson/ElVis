///////////////////////////////////////////////////////////////////////////////
//
// The MIT License
//
// Copyright (c) 2006 Scientific Computing and Imaging Institute,
// University of Utah (USA)
//
// License for the specific language governing rights and limitations under
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ELVIS_ISOSURFACE_MODULE_CU
#define ELVIS_ISOSURFACE_MODULE_CU

#include <ElVis/Core/PrimaryRayGenerator.cu>
#include <ElVis/Core/VolumeRenderingPayload.cu>
#include <ElVis/Core/ConvertToColor.cu>
#include <ElVis/Core/FindElement.cu>
#include <ElVis/Core/Interval.hpp>
#include <ElVis/Core/IntervalPoint.cu>
#include <ElVis/Core/IntervalMatrix.cu>
#include <ElVis/Core/ElementId.h>
#include <ElVis/Core/ElementTraversal.cu>

#include <ElVis/Math/TrapezoidalIntegration.hpp>
#include <ElVis/Core/VolumeRenderingOptiXModule.cu>

// This file is meant to be included from a higher level .cu file which has
// already managed
// the many of the necessary header inclusions.

#include <ElVis/Core/Jacobi.hpp>
#include <ElVis/Core/matrix.cu>
#include <ElVis/Core/Cuda.h>

class OrthogonalLegendreBasis
{
public:
  __device__ static ElVisFloat Eval(unsigned int i, const ElVisFloat& x)
  {
    return Sqrtf((MAKE_FLOAT(2.0) * i + MAKE_FLOAT(1.0)) / MAKE_FLOAT(2.0)) *
           ElVis::OrthoPoly::P(i, 0, 0, x);
  }
};

/// \brief Project a function onto a polynomial of a given order.
/// \param[in] order the order of the projected polynomial
/// \param[in] allNodes the numeric integration nodes.
/// \param[in] allWeights the numeric integration weights.
/// \param[in] f The original function to be approximated.
/// \param[out] coefss The coefficients of the projected polynomial.
template <typename FuncType>
__device__ void GenerateLeastSquaresPolynomialProjection(
  unsigned int order,
  const ElVisFloat* __restrict__ allNodes,
  const ElVisFloat* __restrict__ allWeights,
  const FuncType& f,
  ElVisFloat* coeffs)
{
  // Nodes and weights start with two point rules
  unsigned int index = (order - 1) * (order);
  index = index >> 1;
  index += order - 1;

  // ELVIS_PRINTF("Index %d\n", index);
  const ElVisFloat* __restrict__ nodes = allNodes + index;
  const ElVisFloat* __restrict__ weights = allWeights + index;

  for (unsigned int j = 0; j <= order; ++j)
  {
    coeffs[j] = MAKE_FLOAT(0.0);
  }

  for (unsigned int k = 0; k <= order; ++k)
  {
    //            ELVIS_PRINTF("K %d, node %2.15f, weight %2.15f, sample
    //            %2.15f, basis %2.15f\n",
    //                     k, nodes[k], weights[k], workspace[k],
    //                     OrthogonalLegendreBasis::Eval(c_index, nodes[k]));
    ElVisFloat sample = f(nodes[k]) * weights[k];
    for (unsigned int c_index = 0; c_index <= order; ++c_index)
    {
      ElVisFloat scale = OrthogonalLegendreBasis::Eval(c_index, nodes[k]);
      coeffs[c_index] += sample * scale;
    }
  }
}

template <typename T1, typename T2>
__device__ T1 SIGN(const T1& a, const T2& b)
{
  return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

__device__ void balance(SquareMatrix& a)
{
  int n = a.GetSize();
  const ElVisFloat RADIX = 2;
  bool done = false;
  ElVisFloat sqrdx = RADIX * RADIX;
  while (!done)
  {
    done = true;
    for (int i = 0; i < n; i++)
    {
      ElVisFloat r = 0.0, c = 0.0;
      for (int j = 0; j < n; j++)
      {
        if (j != i)
        {
          c += abs(a(j, i));
          r += abs(a(i, j));
        }
      }
      if (c != 0.0 && r != 0.0)
      {
        ElVisFloat g = r / RADIX;
        ElVisFloat f = 1.0;
        ElVisFloat s = c + r;
        while (c < g)
        {
          f *= RADIX;
          c *= sqrdx;
        }
        g = r * RADIX;
        while (c > g)
        {
          f /= RADIX;
          c /= sqrdx;
        }
        if ((c + r) / f < 0.95 * s)
        {
          done = false;
          g = 1.0 / f;
          // scale[i] *= f;
          for (int j = 0; j < n; j++)
            a(i, j) *= g;
          for (int j = 0; j < n; j++)
            a(j, i) *= f;
        }
      }
    }
  }
}

// returns roots in wri.  Since we don't care about complex roots, they are just
// set to -1.0
__device__ void hqr(SquareMatrix& a, int n, ElVisFloat* wri)
{
  int nn, m, l, k, j, its, i, mmin;
  ElVisFloat z, y, x, w, v, u, t, s, r, q, p, anorm = MAKE_FLOAT(0.0);

  const ElVisFloat EPS = MAKE_FLOAT(1e-8);

  for (i = 0; i < n; i++)
  {
    for (j = max(i - 1, 0); j < n; j++)
    {
      anorm += abs(a(i, j));
    }
  }

  nn = n - 1;
  t = 0.0;
  while (nn >= 0)
  {
    its = 0;
    do
    {
      for (l = nn; l > 0; l--)
      {
        s = abs(a(l - 1, l - 1)) + abs(a(l, l));
        if (s == 0.0) s = anorm;
        if (abs(a(l, l - 1)) <= EPS * s)
        {
          a(l, l - 1) = 0.0;
          break;
        }
      }
      x = a(nn, nn);
      if (l == nn)
      {
        wri[nn--] = x + t;
      }
      else
      {
        y = a(nn - 1, nn - 1);
        w = a(nn, nn - 1) * a(nn - 1, nn);
        if (l == nn - 1)
        {
          p = 0.5 * (y - x);
          q = p * p + w;
          z = sqrt(abs(q));
          x += t;
          if (q >= 0.0)
          {
            z = p + SIGN(z, p);
            wri[nn - 1] = wri[nn] = x + z;
            if (z != 0.0) wri[nn] = x - w / z;
          }
          else
          {
            // wri[nn]=Complex(x+p,-z);
            // wri[nn-1]=conj(wri[nn]);
            wri[nn] = MAKE_FLOAT(-10.0);
            wri[nn - 1] = MAKE_FLOAT(-10.0);
          }
          nn -= 2;
        }
        else
        {
          if (its == 30) return;
          if (its == 10 || its == 20)
          {
            t += x;
            for (i = 0; i < nn + 1; i++)
              a(i, i) -= x;
            s = abs(a(nn, nn - 1)) + abs(a(nn - 1, nn - 2));
            y = x = 0.75 * s;
            w = -0.4375 * s * s;
          }
          ++its;
          for (m = nn - 2; m >= l; m--)
          {
            z = a(m, m);
            r = x - z;
            s = y - z;
            p = (r * s - w) / a(m + 1, m) + a(m, m + 1);
            q = a(m + 1, m + 1) - z - r - s;
            r = a(m + 2, m + 1);
            s = abs(p) + abs(q) + abs(r);
            p /= s;
            q /= s;
            r /= s;
            if (m == l) break;
            u = abs(a(m, m - 1)) * (abs(q) + abs(r));
            v = abs(p) * (abs(a(m - 1, m - 1)) + abs(z) + abs(a(m + 1, m + 1)));
            if (u <= EPS * v) break;
          }
          for (i = m; i < nn - 1; i++)
          {
            a(i + 2, i) = 0.0;
            if (i != m) a(i + 2, i - 1) = 0.0;
          }
          for (k = m; k < nn; k++)
          {
            if (k != m)
            {
              p = a(k, k - 1);
              q = a(k + 1, k - 1);
              r = 0.0;
              if (k + 1 != nn) r = a(k + 2, k - 1);
              if ((x = abs(p) + abs(q) + abs(r)) != 0.0)
              {
                p /= x;
                q /= x;
                r /= x;
              }
            }
            if ((s = SIGN(sqrt(p * p + q * q + r * r), p)) != 0.0)
            {
              if (k == m)
              {
                if (l != m) a(k, k - 1) = -a(k, k - 1);
              }
              else
              {
                a(k, k - 1) = -s * x;
              }
              p += s;
              x = p / s;
              y = q / s;
              z = r / s;
              q /= p;
              r /= p;
              for (j = k; j < nn + 1; j++)
              {
                p = a(k, j) + q * a(k + 1, j);
                if (k + 1 != nn)
                {
                  p += r * a(k + 2, j);
                  a(k + 2, j) -= p * z;
                }
                a(k + 1, j) -= p * y;
                a(k, j) -= p * x;
              }
              mmin = nn < k + 3 ? nn : k + 3;
              for (i = l; i < mmin + 1; i++)
              {
                p = x * a(i, k) + y * a(i, k + 1);
                if (k + 1 != nn)
                {
                  p += z * a(i, k + 2);
                  a(i, k + 2) -= p * r;
                }
                a(i, k + 1) -= p * q;
                a(i, k) -= p;
              }
            }
          }
        }
      }
    } while (l + 1 < nn);
  }
}

// Field evaluator used when the t values are in [-1..1], rather than world
// space.
struct IsosurfaceFieldEvaluator
{
public:
  ELVIS_DEVICE IsosurfaceFieldEvaluator()
    : Origin(),
      Direction(),
      A(),
      B(),
      ElementId(0),
      ElementType(0),
      ReferencePointType(ElVis::eReferencePointIsInvalid),
      InitialGuess()
  {
  }

  // Some versions of OptiX don't support constructors with parameters,
  // so we use two stage initialization.
  __device__ void initialize(const ElVisFloat3& origin,
                             const ElVisFloat3& direction,
                             ElVisFloat segmentStartT,
                             ElVisFloat segmentEndT,
                             unsigned int elementId,
                             unsigned int elementType,
                             int fieldId)
  {
    Origin = origin;
    Direction = direction;
    A = segmentStartT;
    B = segmentEndT;
    ElementId = elementId;
    ElementType = elementType;
    FieldId = fieldId;
  }

  __device__ ElVisFloat operator()(const ElVisFloat& t) const
  {
    // Incoming t is [-1..1], we need to scale to [A,B]
    ElVisFloat scaledT = (t + MAKE_FLOAT(1.0)) / MAKE_FLOAT(2.0) * (B - A) + A;
    ElVisFloat3 p = Origin + scaledT * Direction;
    ElVisFloat s = EvaluateFieldOptiX(
      ElementId, ElementType, FieldId, p, ReferencePointType, InitialGuess);
    ReferencePointType = ElVis::eReferencePointIsInitialGuess;
    return s;
  }

  ElVisFloat3 Origin;
  ElVisFloat3 Direction;
  ElVisFloat A;
  ElVisFloat B;
  unsigned int ElementId;
  unsigned int ElementType;
  int FieldId;
  mutable ElVis::ReferencePointParameterType ReferencePointType;
  mutable ElVisFloat3 InitialGuess;

private:
  IsosurfaceFieldEvaluator(const IsosurfaceFieldEvaluator& rhs);
  IsosurfaceFieldEvaluator& operator=(const IsosurfaceFieldEvaluator& rhs);
};

__device__ void GenerateRowMajorHessenbergMatrix(
  const ElVisFloat* monomialCoefficients, int n, SquareMatrix& h)
{

  // First row
  for (int column = 0; column < n - 1; ++column)
  {
    h(0, column) = MAKE_FLOAT(0.0);
  }

  for (int row = 1; row < n; ++row)
  {
    for (int column = 0; column < n - 1; ++column)
    {
      if (row == column + 1)
      {
        h(row, column) = MAKE_FLOAT(1.0);
      }
      else
      {
        h(row, column) = MAKE_FLOAT(0.0);
      }
    }
  }

  ElVisFloat inverse = MAKE_FLOAT(-1.0) / monomialCoefficients[n];
  for (int row = 0; row < n; ++row)
  {
    h(row, n - 1) = monomialCoefficients[row] * inverse;
  }
}

__device__ void PrintMatrix(SquareMatrix& m)
{
  for (unsigned int row = 0; row < m.GetSize(); ++row)
  {
    for (unsigned int column = 0; column < m.GetSize(); ++column)
    {
      ELVIS_PRINTF("%2.15f, ", m(row, column));
    }
    ELVIS_PRINTF("\n");
  }
}

__device__ void ConvertToMonomial(unsigned int order,
                                  ElVisFloat* monomialConversionBuffer,
                                  const ElVisFloat* legendreCoeffs,
                                  ElVisFloat* monomialCoeffs)
{
  int tableIndex = 0;
  for (int i = 2; i <= order; ++i)
  {
    tableIndex += i * i;
  }
  // ELVIS_PRINTF("Table Index %d\n", tableIndex);
  SquareMatrix m(&monomialConversionBuffer[tableIndex], order + 1);

  // ELVIS_PRINTF("Monomial conversion matrix.\n");
  // PrintMatrix(m);

  // Now that we have the coefficient table we can convert.
  for (unsigned int coeffIndex = 0; coeffIndex <= order; ++coeffIndex)
  {
    monomialCoeffs[coeffIndex] = MAKE_FLOAT(0.0);
    for (unsigned int legCoeffIndex = 0; legCoeffIndex <= order;
         ++legCoeffIndex)
    {
      monomialCoeffs[coeffIndex] +=
        legendreCoeffs[legCoeffIndex] * m(legCoeffIndex, coeffIndex);
    }
  }
}

rtDeclareVariable(uint, NumIsosurfaces, , );
rtBuffer<ElVisFloat> SurfaceIsovalues;
rtDeclareVariable(int, isosurfaceProjectionOrder, , );
rtDeclareVariable(ElVisFloat, isosurfaceEpsilon, , );
rtBuffer<ElVisFloat> Nodes;
rtBuffer<ElVisFloat> Weights;
rtBuffer<ElVisFloat> MonomialConversionTable;

#define WORKSPACE_SIZE 32

__device__ bool FindIsosurfaceInSegment(const Segment& seg,
                                        const ElVisFloat3& origin)
{
  bool result = false;

  // ELVIS_PRINTF("Find Isosurface in Segment\n");
  //  ELVIS_PRINTF("FindIsosurfaceInSegment: Projection Order %d\n",
  //               isosurfaceProjectionOrder);
  //  ELVIS_PRINTF("FindIsosurfaceInSegment: Epsilon %f\n", isosurfaceEpsilon);

  // TRICKY - We can't use dynamic memory allocation in OptiX, and we can't
  // allocate per-launch index memory from the CPU, so the temporary
  // workspace required by isosurface must be allocated statically.  Currently
  // the highest supported projection order is 20, which would required 21
  // coefficients.  We allocate space for 32 coefficients here to allow for
  // some growth in allowed order, and to improve memory alignment.
  if (isosurfaceProjectionOrder + 1 >= WORKSPACE_SIZE)
  {
    rtThrow(RT_EXCEPTION_USER + eRequestedIsosurfaceOrderTooLarge);
  }

  int elementId = seg.ElementId;
  int elementTypeId = seg.ElementTypeId;

  ElVisFloat a = seg.Start;
  ElVisFloat b = seg.End;
  ElVisFloat3 rayDirection = seg.RayDirection;

  ElVisFloat bestDepth = depth_buffer[launch_index];

  unsigned int numIsosurfaces = SurfaceIsovalues.size();

  // TODO - We don't need to reproject for each isovalue, we just need to adjust
  // the last coefficient.
  for (unsigned int isosurfaceId = 0; isosurfaceId < numIsosurfaces;
       ++isosurfaceId)
  {
    // Step 1 - project the function along the current segment onto a
    // polynomial with a user-defined order.  The user can increase the order
    // to increase the accuracy of the approximation (within reason, increasing
    // the order too high will also increase floating point errors).

    // The coefficients for the approximating polynomial using the legendre
    // basis.
    ElVisFloat legendrePolynomialCoefficients[WORKSPACE_SIZE];

    IsosurfaceFieldEvaluator f;
    f.initialize(origin, rayDirection, a, b, elementId, elementTypeId, FieldId);

    GenerateLeastSquaresPolynomialProjection(isosurfaceProjectionOrder,
                                             &Nodes[0], &Weights[0], f,
                                             legendrePolynomialCoefficients);
    //    ELVIS_PRINTF("FindIsosurfaceInSegment: First coefficient: %f\n",
    //                 legendrePolynomialCoefficients[0]);
    //    ELVIS_PRINTF(
    //      "FindIsosurfaceInSegment: Legendre coefficients: %8.15f, %8.15f,
    //      %8.15f, "
    //      "%8.15f, %8.15f, %8.15f, %8.15f, %8.15f, %8.15f, %8.15f\n",
    //      legendrePolynomialCoefficients[0],
    //      legendrePolynomialCoefficients[1],
    //      legendrePolynomialCoefficients[2],
    //      legendrePolynomialCoefficients[3],
    //      legendrePolynomialCoefficients[4],
    //      legendrePolynomialCoefficients[5],
    //      legendrePolynomialCoefficients[6],
    //      legendrePolynomialCoefficients[7],
    //      legendrePolynomialCoefficients[8],
    //      legendrePolynomialCoefficients[9]);

    // In some cases, the coefficients for the larger orders are very small,
    // which can indicate that we didn't need to project onto a polynomial
    // of such high order.  In this case, evaluating these terms can introduce
    // error into the final calculation, so we filter out all higher order
    // coefficients that don't fall within a user-specified epsilon.

    int reducedOrder = isosurfaceProjectionOrder;

    for (int i = isosurfaceProjectionOrder; i >= 1; --i)
    {
      if (Fabsf(legendrePolynomialCoefficients[i]) > isosurfaceEpsilon)
      {
        reducedOrder = i;
        break;
      }
    }

    // ELVIS_PRINTF("FindIsosurfaceInSegment: Reduced order %d\n",
    // reducedOrder);

    ElVisFloat monomialCoefficients[WORKSPACE_SIZE];
    ConvertToMonomial(reducedOrder, &MonomialConversionTable[0],
                      legendrePolynomialCoefficients, monomialCoefficients);

    //    ELVIS_PRINTF("Monomial %8.15f, %8.15f, %8.15f, %8.15f, %8.15f, %8.15f,
    //    "
    //                 "%8.15f, %8.15f, %8.15f\n",
    //                 monomialCoefficients[0], monomialCoefficients[1],
    //                 monomialCoefficients[2], monomialCoefficients[3],
    //                 monomialCoefficients[4], monomialCoefficients[5],
    //                 monomialCoefficients[6], monomialCoefficients[7],
    //                 monomialCoefficients[8]);

    monomialCoefficients[0] -= SurfaceIsovalues[isosurfaceId];

    ElVisFloat h_data[WORKSPACE_SIZE * WORKSPACE_SIZE];

    SquareMatrix h(h_data, reducedOrder);
    GenerateRowMajorHessenbergMatrix(monomialCoefficients, reducedOrder, h);

    //    ELVIS_PRINTF("Before balancing.\n");
    //    PrintMatrix(h);

    balance(h);

    //    ELVIS_PRINTF("After balancing.\n");
    //    PrintMatrix(h);

    ElVisFloat roots[WORKSPACE_SIZE];
    for (int i = 0; i <= reducedOrder; ++i)
    {
      // Initialize the roots to a value that will fall outside the range we are
      // looking for.
      // hqr can sometimes fail to set a root if one isn't found.
      roots[i] = MAKE_FLOAT(-2.0);
    }
    hqr(h, reducedOrder, roots);

    //    ELVIS_PRINTF("Roots %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
    //      roots[0],
    //      roots[1],
    //      roots[2],
    //      roots[3],
    //      roots[4],
    //      roots[5],
    //        roots[6],
    //        roots[7],
    //        roots[8],
    //        roots[9]);

    ElVisFloat foundRoot = ELVIS_FLOAT_MAX;
    for (int i = 0; i < reducedOrder; ++i)
    {
      ElVisFloat root = roots[i];
      if (root >= MAKE_FLOAT(-1.0) && root <= MAKE_FLOAT(1.0) &&
          root <= foundRoot)
      {
        ElVisFloat foundT =
          (root + MAKE_FLOAT(1.0)) / MAKE_FLOAT(2.0) * (f.B - f.A) + f.A;
        if (foundT < bestDepth)
        {
          foundRoot = root;
        }
      }
    }

    if (foundRoot != ELVIS_FLOAT_MAX)
    {
      ElVisFloat foundT =
        (foundRoot + MAKE_FLOAT(1.0)) / MAKE_FLOAT(2.0) * (f.B - f.A) + f.A;

      ElVisFloat3 foundIntersectionPoint = origin + foundT * rayDirection;

      intersection_buffer[launch_index] = foundIntersectionPoint;
      SampleBuffer[launch_index] = EvaluateFieldOptiX(
        elementId, elementTypeId, FieldId, foundIntersectionPoint);

      //      ELVIS_PRINTF("FindIsosurfaceInSegment: ########################
      //      Found "
      //                   "root %2.15f, in world %2.15f with value %f \n",
      //                   foundRoot, foundT, SampleBuffer[launch_index]);

      EvaluateNormalOptiX(elementId, elementTypeId, FieldId,
                          foundIntersectionPoint, normal_buffer[launch_index]);
      depth_buffer[launch_index] = foundT;
      bestDepth = foundT;
      result = true;
    }
//    else
//    {
//      // ELVIS_PRINTF("FindIsosurfaceInSegment: No root found\n");
//      for (int i = 0; i < reducedOrder; ++i)
//      {
//        ElVisFloat root = roots[i];
//        if (root >= MAKE_FLOAT(-1.0) && root <= MAKE_FLOAT(1.0))
//        {
//          ElVisFloat foundT =
//            (root + MAKE_FLOAT(1.0)) / MAKE_FLOAT(2.0) * (f.B - f.A) + f.A;
//          //          ELVIS_PRINTF("FindIsosurfaceInSegment: root[%d] = %2.15f,
//          //          foundT = "
//          //                       "%2.15f, bestDepth = %2.15f\n",
//          //                       i, root, foundT, bestDepth);
//        }
//      }
//    }
  }
  return result;
}

RT_PROGRAM void FindIsosurface()
{
  ElementTraversal(FindIsosurfaceInSegment);
}

#endif
