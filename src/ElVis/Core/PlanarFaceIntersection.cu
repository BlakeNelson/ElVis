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

#ifndef ELVIS_CORE_PLANAR_FACE_INTERSECTION_CU
#define ELVIS_CORE_PLANAR_FACE_INTERSECTION_CU

// A planar face is defined by 3 or 4 vertices.

// planar face needs a bounding box and intersection program.

// A local id to global id map

RT_PROGRAM void PlanarFaceIntersection(int primitiveId)
{
  // primitiveId is the local index for a planar face.
}

RT_PROGRAM void PlanarFaceBoundingBoxProgram(int primitiveId, float result[6])
{
  // primitiveId is the local index for a planar face.
  uint globalIdx = PlanarFaceToGlobalIdxMap[primitiveId];

  optix::Aabb* aabb = (optix::Aabb*)result;

  if( FaceEnabled[globalIdx] )
  {
      ElVisFloat3 p0 = FaceMinExtentBuffer[globalIdx];
      ElVisFloat3 p1 = FaceMaxExtentBuffer[globalIdx];

      aabb->m_min = make_float3(p0.x, p0.y, p0.z);
      aabb->m_max = make_float3(p1.x, p1.y, p1.z);
  }
  else
  {
      aabb->m_min = make_float3(100000.0f, 100000.0f, 100000.0f);
      aabb->m_max = make_float3(100000.1f, 100000.1f, 100000.1f);
  }
}

#endif