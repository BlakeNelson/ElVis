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

#ifndef ELVIS_CORE_CUDA_H
#define ELVIS_CORE_CUDA_H

#if defined(ELVIS_OPTIX_MODULE)
#define ELVIS_DEVICE __device__ __forceinline__
#elif defined(__CUDACC__)
#define ELVIS_DEVICE __device__
#else
#define ELVIS_DEVICE
#endif

enum ElVisError
{
  // Conversion succeeds.
  eNoError,

  // Conversion fails because the world point lies outside the element.
  ePointOutsideElement,

  eInvalidElementId,

  eInvalidElementType,

  eInvalidFaceId,

  eFieldNotDefinedOnFace,

  // Other failures.
  eConvergenceFailure,

  eRequestedIsosurfaceOrderTooLarge
};

#define checkElVisError(x)                                                     \
  (                                                                            \
    {                                                                          \
    ElVisError r = (x);                                                        \
    if (r != eNoError) rtThrow(RT_EXCEPTION_USER + r);                         \
    })

#endif // ELVIS_CORE_CUDA_H
