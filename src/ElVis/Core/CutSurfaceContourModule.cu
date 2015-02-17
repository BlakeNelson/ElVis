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

#ifndef ELVIS_CUT_SURFACE_CONTOUR_MODULE_CU
#define ELVIS_CUT_SURFACE_CONTOUR_MODULE_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>
#include <ElVis/Core/CutSurfacePayloads.cu>
#include <ElVis/Core/ConvertToColor.cu>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/Interval.hpp>
#include <ElVis/Core/IntervalPoint.cu>
#include <ElVis/Core/ElementId.h>

rtBuffer<ElVisFloat, 2> ContourSampleBuffer;
//rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtBuffer<ElVisFloat3, 2> ReferencePointAtIntersectionBuffer;
rtBuffer<unsigned int, 2> ElementIdAtIntersectionBuffer;
rtBuffer<unsigned int, 2> ElementTypeAtIntersectionBuffer;

rtBuffer<ElVisFloat, 1> Isovalues;

rtDeclareVariable(int, TreatElementBoundariesAsDiscontinuous, , );

RT_PROGRAM void ContourMiss()
{
}

__device__ __forceinline__ ElVisFloat3 CalculateRayDirection(const uint2& pixelIndex, const ElVisFloat2& offset)
{
    ElVisFloat2 screen = MakeFloat2(color_buffer.size());
    ElVisFloat2 pixelSize = MAKE_FLOAT(2.0)/screen;
    
    ElVisFloat2 pixelOffset = offset/MAKE_FLOAT(2.0);
    
    ElVisFloat2 d = MakeFloat2(launch_index) / screen * MAKE_FLOAT(2.0) - MAKE_FLOAT(1.0);
    d = d + pixelSize * pixelOffset;
    
    //float3 ray_origin = eye;
    ElVisFloat3 ray_direction = normalize(d.x*U + d.y*V + W);
    return ray_direction;
}

// raw_color_buffer 
RT_PROGRAM void SamplePixelCornersRayGeneratorForCategorization()
{
    ELVIS_PRINTF("SamplePixelCornersRayGenerator\n");
    // Note - there are occlusion issues here.
    ElVisFloat2 screen = MakeFloat2(color_buffer.size());
    ElVisFloat2 pixelSize = MAKE_FLOAT(2.0)/screen;
    
    ElVisFloat x = MAKE_FLOAT(-1.0);
    ElVisFloat y = MAKE_FLOAT(-1.0);
    ElVisFloat2 pixelOffset = MakeFloat2(x, y)*MAKE_FLOAT(.5);
    
    ElVisFloat2 d = MakeFloat2(launch_index) / screen * MAKE_FLOAT(2.0) - MAKE_FLOAT(1.0);
    d = d + pixelSize * pixelOffset;
    
    ElVisFloat3 ray_origin = eye;
    ElVisFloat3 ray_direction = CalculateRayDirection(launch_index, MakeFloat2(MAKE_FLOAT(-1.0), MAKE_FLOAT(-1.0)));//normalize(d.x*U + d.y*V + W);

    optix::Ray ray = optix::make_Ray(ConvertToFloat3(ray_origin), ConvertToFloat3(ray_direction), 0, 1e-3f, RT_DEFAULT_MAX);
    CutSurfaceScalarValuePayload payload;
    
    payload.Initialize();
    payload.isValid = false;
    payload.scalarValue = ELVIS_FLOAT_MAX;
    payload.Normal = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    payload.Color = MakeFloat3(MAKE_FLOAT(1.0), MAKE_FLOAT(1.0), MAKE_FLOAT(1.0));
    rtTrace(SurfaceGeometryGroup, ray, payload);
    
    ELVIS_PRINTF("SamplePixelCornersRayGeneratorForCategorization: Scalar Value (%f)\n", payload.scalarValue);
    ELVIS_PRINTF("SamplePixelCornersRayGeneratorForCategorization: Element Id (%d)\n", payload.elementId);
    ELVIS_PRINTF("SamplePixelCornersRayGeneratorForCategorization: Element Type (%d)\n", payload.elementType);
    ELVIS_PRINTF("SamplePixelCornersRayGeneratorForCategorization: Reference Point (%f, %f, %f)\n", payload.ReferenceIntersectionPoint.x,
                 payload.ReferenceIntersectionPoint.y, payload.ReferenceIntersectionPoint.z);
    ContourSampleBuffer[launch_index] = payload.scalarValue;
    ReferencePointAtIntersectionBuffer[launch_index] = payload.ReferenceIntersectionPoint;
    ElementIdAtIntersectionBuffer[launch_index] = payload.elementId;
    ElementTypeAtIntersectionBuffer[launch_index] = payload.elementType;

}

__device__ __forceinline__ ElVis::Interval<ElVisFloat> EvaluateBetweenReferencePoints(unsigned int elementId, unsigned int elementType, const ElVisFloat3& p0, const ElVisFloat3& p1)
{
     ElVis::Interval<ElVisFloat> r0(fminf(p0.x, p1.x), fmaxf(p0.x, p1.x));
     ElVis::Interval<ElVisFloat> s0(fminf(p0.y, p1.y), fmaxf(p0.y, p1.y));
     ElVis::Interval<ElVisFloat> t0(fminf(p0.z, p1.z), fmaxf(p0.z, p1.z));

     IntervalPoint ip(r0, s0, t0);
     ElVis::Interval<ElVisFloat> result;
     SampleScalarFieldAtReferencePointOptiX(elementId, elementType, FieldId, ip, ip,
                                            result);
     return result;
}

__device__ __forceinline__ ElVis::Interval<ElVisFloat> EvaluateEdge(unsigned int elementId, unsigned int elementType, uint2 i0, uint2 i1)
{
     ElVisFloat3 p0 = ReferencePointAtIntersectionBuffer[i0];
     ElVisFloat3 p1 = ReferencePointAtIntersectionBuffer[i1];

     ELVIS_PRINTF("EvaluateEdge: P0 = (%f, %f, %f), P1 = (%f, %f, %f).\n",
                  p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
     return EvaluateBetweenReferencePoints(elementId, elementType, p0, p1);
}

// p0 - The reference point at the beginning of the interval.
// p1 - The reference point at the end of the interval.
__device__ __forceinline__ ElVis::Interval<ElVisFloat> SubdivideInterval1(unsigned int elementId, unsigned int elementType,
                                  const ElVisFloat3& p0, const ElVisFloat3& p1, 
                                  const ElVisFloat2& cornerOffset0, const ElVisFloat2& cornerOffset1)
{
    // Input is two pixel corners.  We'll do up to two levels of subdivision to start with and see how that affects
    // the final image.
    // First level is a ray at 1/2 between pixel corners.
    // Seconds level is at 1/4 and 3/4.

    ElVisFloat2 offset;
    if( cornerOffset0.x == cornerOffset1.x )
    {
        // Vertical edge
        offset = MakeFloat2(cornerOffset0.x, MAKE_FLOAT(0.0));
    }
    else
    {
        // horizontal edge.
        offset = MakeFloat2(MAKE_FLOAT(0.0), cornerOffset0.y);
    }

    ElVisFloat3 ray_direction = CalculateRayDirection(launch_index, offset); 

    ElVisFloat3 ray_origin = eye;
    optix::Ray ray = optix::make_Ray(ConvertToFloat3(ray_origin), ConvertToFloat3(ray_direction), 2, 1e-3f, RT_DEFAULT_MAX);
    CutSurfaceScalarValuePayload payload;
    payload.Initialize();
    payload.isValid = false;
    payload.scalarValue = ELVIS_FLOAT_MAX;
    payload.Normal = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    payload.Color = MakeFloat3(MAKE_FLOAT(1.0), MAKE_FLOAT(1.0), MAKE_FLOAT(1.0));
    rtTrace(SurfaceGeometryGroup, ray, payload);
    
    // Now evaluate the interval math between p0-mid and mid-p1 to see if we can reject
    // this pixel.
    ElVisFloat3 mid = payload.ReferenceIntersectionPoint;
    ElVis::Interval<ElVisFloat> i0 = EvaluateBetweenReferencePoints(elementId, elementType, p0, mid);
    ElVis::Interval<ElVisFloat> i1 = EvaluateBetweenReferencePoints(elementId, elementType, mid, p1);

    return ElVis::Interval<ElVisFloat>(fminf(i0.GetLow(), i1.GetLow()), fmaxf(i0.GetHigh(), i1.GetHigh()));
}

__device__ __forceinline__ ElVis::Interval<ElVisFloat> SubdivideInterval2(unsigned int elementId, unsigned int elementType,
                                  const ElVisFloat3& p0, const ElVisFloat3& p1, 
                                  const ElVisFloat2& cornerOffset0, const ElVisFloat2& cornerOffset1)
{
    // Input is two pixel corners.  We'll do up to two levels of subdivision to start with and see how that affects
    // the final image.
    // First level is a ray at 1/2 between pixel corners.
    // Seconds level is at 1/4 and 3/4.

    ElVisFloat2 offset[3];
    if( cornerOffset0.x == cornerOffset1.x )
    {
        // Vertical edge
        offset[0] = MakeFloat2(cornerOffset0.x, MAKE_FLOAT(-.5));
        offset[1] = MakeFloat2(cornerOffset0.x, MAKE_FLOAT(0.0));
        offset[2] = MakeFloat2(cornerOffset0.x, MAKE_FLOAT(.5));
    }
    else
    {
        offset[0] = MakeFloat2(MAKE_FLOAT(-.5), cornerOffset0.y);
        offset[1] = MakeFloat2(MAKE_FLOAT(0.0), cornerOffset0.y);
        offset[2] = MakeFloat2(MAKE_FLOAT(.5), cornerOffset0.y);
    }

    ElVisFloat3 ray_direction[] = {
        CalculateRayDirection(launch_index, offset[0]),
        CalculateRayDirection(launch_index, offset[1]),
        CalculateRayDirection(launch_index, offset[2]) }; 

    CutSurfaceScalarValuePayload payload[3];
    ElVisFloat3 ray_origin = eye;

    for(unsigned int i = 0; i < 3; ++i)
    {
        optix::Ray ray = optix::make_Ray(ConvertToFloat3(ray_origin), ConvertToFloat3(ray_direction[i]), 2, 1e-3f, RT_DEFAULT_MAX);

        payload[i].Initialize();
        payload[i].isValid = false;
        payload[i].scalarValue = ELVIS_FLOAT_MAX;
        payload[i].Normal = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
        payload[i].Color = MakeFloat3(MAKE_FLOAT(1.0), MAKE_FLOAT(1.0), MAKE_FLOAT(1.0));
        rtTrace(SurfaceGeometryGroup, ray, payload[i]);
    }

    // Now evaluate the interval math between p0-mid and mid-p1 to see if we can reject
    // this pixel.
    ElVis::Interval<ElVisFloat> i0 = EvaluateBetweenReferencePoints(elementId, elementType, p0, payload[0].ReferenceIntersectionPoint);
    ElVis::Interval<ElVisFloat> i1 = EvaluateBetweenReferencePoints(elementId, elementType, payload[0].ReferenceIntersectionPoint, payload[1].ReferenceIntersectionPoint);
    ElVis::Interval<ElVisFloat> i2 = EvaluateBetweenReferencePoints(elementId, elementType, payload[1].ReferenceIntersectionPoint, payload[2].ReferenceIntersectionPoint);
    ElVis::Interval<ElVisFloat> i3 = EvaluateBetweenReferencePoints(elementId, elementType, payload[2].ReferenceIntersectionPoint, p1);

    i0.Combine(i1);
    i0.Combine(i2);
    i0.Combine(i3);
    return i0;
}

__device__ __forceinline__ ElVis::Interval<ElVisFloat> SubdivideInterval3(unsigned int elementId, unsigned int elementType,
                                  const ElVisFloat3& p0, const ElVisFloat3& p1, 
                                  const ElVisFloat2& cornerOffset0, const ElVisFloat2& cornerOffset1)
{
    // Input is two pixel corners.  We'll do up to two levels of subdivision to start with and see how that affects
    // the final image.
    // First level is a ray at 1/2 between pixel corners.
    // Seconds level is at 1/4 and 3/4.

    ElVisFloat2 offset[7];
    if( cornerOffset0.x == cornerOffset1.x )
    {
        // Vertical edge
        offset[0] = MakeFloat2(cornerOffset0.x, MAKE_FLOAT(-.75));
        offset[1] = MakeFloat2(cornerOffset0.x, MAKE_FLOAT(-.5));
        offset[2] = MakeFloat2(cornerOffset0.x, MAKE_FLOAT(-.25));
        offset[3] = MakeFloat2(cornerOffset0.x, MAKE_FLOAT(0.0));
        offset[4] = MakeFloat2(cornerOffset0.x, MAKE_FLOAT(.25));
        offset[5] = MakeFloat2(cornerOffset0.x, MAKE_FLOAT(.5));
        offset[6] = MakeFloat2(cornerOffset0.x, MAKE_FLOAT(.75));
    }
    else
    {
        offset[0] = MakeFloat2(MAKE_FLOAT(-.75), cornerOffset0.y);
        offset[1] = MakeFloat2(MAKE_FLOAT(-.5), cornerOffset0.y);
        offset[2] = MakeFloat2(MAKE_FLOAT(-.25), cornerOffset0.y);
        offset[3] = MakeFloat2(MAKE_FLOAT(0.0), cornerOffset0.y);
        offset[4] = MakeFloat2(MAKE_FLOAT(.25), cornerOffset0.y);
        offset[5] = MakeFloat2(MAKE_FLOAT(.5), cornerOffset0.y);
        offset[6] = MakeFloat2(MAKE_FLOAT(.75), cornerOffset0.y);
    }

    CutSurfaceScalarValuePayload payload[7];
    ElVisFloat3 ray_origin = eye;

    for(unsigned int i = 0; i < 7; ++i)
    {
        ElVisFloat3 ray_direction = CalculateRayDirection(launch_index, offset[i]);
        optix::Ray ray = optix::make_Ray(ConvertToFloat3(ray_origin), ConvertToFloat3(ray_direction), 2, 1e-3f, RT_DEFAULT_MAX);

        payload[i].Initialize();
        payload[i].isValid = false;
        payload[i].scalarValue = ELVIS_FLOAT_MAX;
        payload[i].Normal = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
        payload[i].Color = MakeFloat3(MAKE_FLOAT(1.0), MAKE_FLOAT(1.0), MAKE_FLOAT(1.0));
        rtTrace(SurfaceGeometryGroup, ray, payload[i]);
    }

    // Now evaluate the interval math between p0-mid and mid-p1 to see if we can reject
    // this pixel.
    ElVis::Interval<ElVisFloat> result = EvaluateBetweenReferencePoints(elementId, elementType, p0, payload[0].ReferenceIntersectionPoint);
    for(unsigned int i = 0; i < 6; ++i)
    {
        ElVis::Interval<ElVisFloat> i1 = EvaluateBetweenReferencePoints(elementId, elementType, payload[i].ReferenceIntersectionPoint, payload[i+1].ReferenceIntersectionPoint);
        result.Combine(i1);
    }
    ElVis::Interval<ElVisFloat> i3 = EvaluateBetweenReferencePoints(elementId, elementType, payload[6].ReferenceIntersectionPoint, p1);
    result.Combine(i3);
    return result;
}

RT_PROGRAM void CategorizeMeshPixels()
{
    // Corner testing.
    // c0 = lower left corner
    // c1 = lower right corner
    // c2 = upper left corner
    // c3 = upper right corner
    uint2 c0_index = make_uint2(launch_index.x, launch_index.y);
    uint2 c1_index = make_uint2(launch_index.x, launch_index.y);;
    c1_index.x += 1;
    uint2 c2_index = make_uint2(launch_index.x, launch_index.y);;
    c2_index.y += 1;
    uint2 c3_index = make_uint2(launch_index.x, launch_index.y);;
    c3_index.x += 1;
    c3_index.y += 1;

    ElVis::ElementId id0;
    ElVis::ElementId id1;
    ElVis::ElementId id2;
    ElVis::ElementId id3;

    id0.Id = ElementIdAtIntersectionBuffer[c0_index];
    id1.Id = ElementIdAtIntersectionBuffer[c1_index];
    id2.Id = ElementIdAtIntersectionBuffer[c2_index];
    id3.Id = ElementIdAtIntersectionBuffer[c3_index];

    id0.Type = ElementTypeAtIntersectionBuffer[c0_index];
    id1.Type = ElementTypeAtIntersectionBuffer[c1_index];
    id2.Type = ElementTypeAtIntersectionBuffer[c2_index];
    id3.Type = ElementTypeAtIntersectionBuffer[c3_index];

    if( id0.Id == -1 ||
        id1.Id == -1 ||
        id2.Id == -1 ||
        id3.Id == -1 )
    {
        return;
    }

    bool pixelIsElementBoundary =
     ( id0 != id1 ||
        id1 != id2 ||
        id2 != id3 );

    if( pixelIsElementBoundary )
    {
        raw_color_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(0.5), MAKE_FLOAT(0.5), MAKE_FLOAT(0.5));
        color_buffer[launch_index] = ConvertToColor(raw_color_buffer[launch_index]);
        normal_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
        SampleBuffer[launch_index] = ELVIS_FLOAT_MAX;
    }
}

rtDeclareVariable(int, MatchVisual3Contours, , );

RT_PROGRAM void CategorizeContourPixels()
{
    ELVIS_PRINTF("CategorizeContourPixels\n");
    // Corner testing.
    // c0 = lower left corner
    // c1 = lower right corner
    // c2 = upper left corner
    // c3 = upper right corner
    uint2 c0_index = make_uint2(launch_index.x, launch_index.y);
    uint2 c1_index = make_uint2(launch_index.x, launch_index.y);;
    c1_index.x += 1;
    uint2 c2_index = make_uint2(launch_index.x, launch_index.y);;
    c2_index.y += 1;
    uint2 c3_index = make_uint2(launch_index.x, launch_index.y);;
    c3_index.x += 1;
    c3_index.y += 1;
    
    ElVisFloat c0 = ContourSampleBuffer[c0_index];
    ElVisFloat c1 = ContourSampleBuffer[c1_index];
    ElVisFloat c2 = ContourSampleBuffer[c2_index];
    ElVisFloat c3 = ContourSampleBuffer[c3_index];
    
    // The 5000 are to get around a bug for the demos but needs to be fixed.
    bool allSamplesValid = true;
    allSamplesValid = (c0 != ELVIS_FLOAT_MAX) &&
        (c1 != ELVIS_FLOAT_MAX) &&
        (c2 != ELVIS_FLOAT_MAX) &&
        (c3 != ELVIS_FLOAT_MAX) &&
        c0 < 5000 && c1 < 5000 && c2 < 5000 && c3 < 5000;


    ElVisFloat3 visual3BackgroundColor = MakeFloat3(MAKE_FLOAT(1.0), MAKE_FLOAT(1.0), MAKE_FLOAT(1.0));

    ELVIS_PRINTF("CategorizeContourPixels: All Samples Valid %d\n", allSamplesValid);
    if( !allSamplesValid ) 
    {
        if( MatchVisual3Contours )
        {
            raw_color_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));;
            color_buffer[launch_index] = ConvertToColor(raw_color_buffer[launch_index]);
            SampleBuffer[launch_index] = ELVIS_FLOAT_MAX;
            normal_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
        }
        return;
    }

    ElVis::ElementId id0;
    ElVis::ElementId id1;
    ElVis::ElementId id2;
    ElVis::ElementId id3;


    id0.Id = ElementIdAtIntersectionBuffer[c0_index];
    id1.Id = ElementIdAtIntersectionBuffer[c1_index];
    id2.Id = ElementIdAtIntersectionBuffer[c2_index];
    id3.Id = ElementIdAtIntersectionBuffer[c3_index];

    id0.Type = ElementTypeAtIntersectionBuffer[c0_index];
    id1.Type = ElementTypeAtIntersectionBuffer[c1_index];
    id2.Type = ElementTypeAtIntersectionBuffer[c2_index];
    id3.Type = ElementTypeAtIntersectionBuffer[c3_index];


    bool pixelIsElementBoundary =
     ( id0 != id1 ||
        id1 != id2 ||
        id2 != id3 );

    if( TreatElementBoundariesAsDiscontinuous && pixelIsElementBoundary )
    {
        if( MatchVisual3Contours )
        {
            raw_color_buffer[launch_index] = visual3BackgroundColor;
            color_buffer[launch_index] = ConvertToColor(raw_color_buffer[launch_index]);
            SampleBuffer[launch_index] = ELVIS_FLOAT_MAX;
            normal_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
        }
        return;
    }

    ElVisFloat3 elementBoundaryColor = MakeFloat3(MAKE_FLOAT(.5), MAKE_FLOAT(.5), MAKE_FLOAT(.5));
    ElVisFloat3 contourColor = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    ElVisFloat3 ambiguousColor = MakeFloat3(MAKE_FLOAT(.25), MAKE_FLOAT(.5), MAKE_FLOAT(.5));
    ElVisFloat3 secondLevelAmbiguous = MakeFloat3(MAKE_FLOAT(1.0), MAKE_FLOAT(.5), MAKE_FLOAT(.5));


    bool oneIsovalueIsValid = false;
    if( MatchVisual3Contours )
    {
        bool oneIsovalueIsValid = false;
        int numIsovalues = Isovalues.size();
        ELVIS_PRINTF("CategorizeContourPixels: Num isovalues: %d\n", numIsovalues);
        for(int isoValueIndex = 0; isoValueIndex < Isovalues.size(); ++isoValueIndex)
        {
            ElVisFloat isovalue = Isovalues[isoValueIndex];
            ELVIS_PRINTF("CategorizeContourPixels: testing isovalue: %f\n", isovalue);
            bool lowerThanOneValue = (isovalue <= c0) || (isovalue <= c1) || (isovalue <= c2) || (isovalue <= c3);
            bool higherThanOneValue = (isovalue >= c0) || (isovalue >= c1) || (isovalue >= c2) || (isovalue >= c3);

            if( lowerThanOneValue && higherThanOneValue )
            {
                oneIsovalueIsValid = true;
                break;

            }
        }

        if( !oneIsovalueIsValid )
        {
            raw_color_buffer[launch_index] = visual3BackgroundColor;
            color_buffer[launch_index] = ConvertToColor(raw_color_buffer[launch_index]);
            SampleBuffer[launch_index] = ELVIS_FLOAT_MAX;
            normal_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
        }
    }
    else
    {
        int numIsovalues = Isovalues.size();
        ELVIS_PRINTF("CategorizeContourPixels: Num isovalues: %d\n", numIsovalues);
        for(int isoValueIndex = 0; isoValueIndex < Isovalues.size(); ++isoValueIndex)
        {
            ElVisFloat isovalue = Isovalues[isoValueIndex];
            ELVIS_PRINTF("CategorizeContourPixels: testing isovalue: %f\n", isovalue);
            bool lowerThanOneValue = (isovalue <= c0) || (isovalue <= c1) || (isovalue <= c2) || (isovalue <= c3);
            bool higherThanOneValue = (isovalue >= c0) || (isovalue >= c1) || (isovalue >= c2) || (isovalue >= c3);

            if( lowerThanOneValue && higherThanOneValue )
            {
                ELVIS_PRINTF("(%d, %d), Isovalue %f and corners %f, %f, %f, %f\n", launch_index.x, launch_index.y, isovalue, c0, c1, c2, c3);
                // BLACK - contour
                raw_color_buffer[launch_index] = contourColor;
                color_buffer[launch_index] = ConvertToColor(raw_color_buffer[launch_index]);
                normal_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));

                // Clear out the sample buffer to prevent the color mapper from kicking in.
                SampleBuffer[launch_index] = ELVIS_FLOAT_MAX;
                oneIsovalueIsValid = true;
            }
        }
    }

    if( !oneIsovalueIsValid )
    {
        ELVIS_PRINTF("CategorizeContourPixels: No isovalues valid.\n");
        // If the element types are all different, then I can't use interval arithmetic.
        if( ElementIdAtIntersectionBuffer[c0_index] != ElementIdAtIntersectionBuffer[c1_index] ||
            ElementIdAtIntersectionBuffer[c0_index] != ElementIdAtIntersectionBuffer[c2_index] ||
            ElementIdAtIntersectionBuffer[c0_index] != ElementIdAtIntersectionBuffer[c3_index] ||
            ElementTypeAtIntersectionBuffer[c0_index] != ElementTypeAtIntersectionBuffer[c1_index] || 
            ElementTypeAtIntersectionBuffer[c0_index] != ElementTypeAtIntersectionBuffer[c2_index] ||
            ElementTypeAtIntersectionBuffer[c0_index] != ElementTypeAtIntersectionBuffer[c3_index])
        {            
            // Mark the element boundary.
            raw_color_buffer[launch_index] = elementBoundaryColor;
            color_buffer[launch_index] = ConvertToColor(raw_color_buffer[launch_index]);
            normal_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
            return;
        }

        // No Subdivisions
        unsigned int id = ElementIdAtIntersectionBuffer[c0_index];
        unsigned int type = ElementTypeAtIntersectionBuffer[c0_index];

        ElVis::Interval<ElVisFloat> edge0 = EvaluateEdge(id, type, c0_index, c1_index);
        ElVis::Interval<ElVisFloat> edge1 = EvaluateEdge(id, type, c0_index, c2_index);
        ElVis::Interval<ElVisFloat> edge2 = EvaluateEdge(id, type, c1_index, c3_index);
        ElVis::Interval<ElVisFloat> edge3 = EvaluateEdge(id, type, c2_index, c3_index);

        bool mayContainAnIsovalue = false;
        for(int isoValueIndex = 0; isoValueIndex < Isovalues.size(); ++isoValueIndex)
        {
            float isovalue = Isovalues[isoValueIndex];
            ELVIS_PRINTF("CategorizeContourPixels: Testing ambiguous %f: (%f, %f), (%f, %f), (%f, %f), (%f, %f).\n",
                         isovalue, edge0.GetLow(), edge0.GetHigh(),
                         edge1.GetLow(), edge1.GetHigh(),
                         edge2.GetLow(), edge2.GetHigh(),
                         edge3.GetLow(), edge3.GetHigh());

            if(edge0.Contains(isovalue) || edge1.Contains(isovalue) ||
                edge2.Contains(isovalue) || edge3.Contains(isovalue) )
            {
                ELVIS_PRINTF("CategorizeContourPixels: May contain isovalue %f: (%f, %f), (%f, %f), (%f, %f), (%f, %f).\n",
                             isovalue, edge0.GetLow(), edge0.GetHigh(),
                             edge1.GetLow(), edge1.GetHigh(),
                             edge2.GetLow(), edge2.GetHigh(),
                             edge3.GetLow(), edge3.GetHigh());
                mayContainAnIsovalue = true;
            }
        }

        if( !mayContainAnIsovalue )
        {
            // Definitely not: WHITE
            raw_color_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(1.0), MAKE_FLOAT(1.0), MAKE_FLOAT(1.0));
            color_buffer[launch_index] = ConvertToColor(raw_color_buffer[launch_index]);
            normal_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
            return;
        }

        const int numSubdivisions = 3;
        if( numSubdivisions == 1 )
        {
            edge0 = SubdivideInterval1(id, type,
                    ReferencePointAtIntersectionBuffer[c0_index],
                    ReferencePointAtIntersectionBuffer[c2_index],
                    make_float2(-1.0, -1.0), make_float2(-1.0, 1.0));
            edge1 = SubdivideInterval1(id, type,
                ReferencePointAtIntersectionBuffer[c1_index],
                ReferencePointAtIntersectionBuffer[c3_index],
                make_float2(1.0, -1.0), make_float2(1.0, 1.0));

            edge2 = SubdivideInterval1(id, type,
                ReferencePointAtIntersectionBuffer[c0_index],
                ReferencePointAtIntersectionBuffer[c1_index],
                make_float2(-1.0, -1.0), make_float2(1.0, -1.0));

            edge3 = SubdivideInterval1(id, type,
                ReferencePointAtIntersectionBuffer[c2_index],
                ReferencePointAtIntersectionBuffer[c3_index],
                make_float2(-1.0, 1.0), make_float2(1.0, 1.0));
        }

        if( numSubdivisions == 2 )
        {
            edge0 = SubdivideInterval2(id, type,
                    ReferencePointAtIntersectionBuffer[c0_index],
                    ReferencePointAtIntersectionBuffer[c2_index],
                    make_float2(-1.0, -1.0), make_float2(-1.0, 1.0));
            edge1 = SubdivideInterval2(id, type,
                ReferencePointAtIntersectionBuffer[c1_index],
                ReferencePointAtIntersectionBuffer[c3_index],
                make_float2(1.0, -1.0), make_float2(1.0, 1.0));

            edge2 = SubdivideInterval2(id, type,
                ReferencePointAtIntersectionBuffer[c0_index],
                ReferencePointAtIntersectionBuffer[c1_index],
                make_float2(-1.0, -1.0), make_float2(1.0, -1.0));

            edge3 = SubdivideInterval2(id, type,
                ReferencePointAtIntersectionBuffer[c2_index],
                ReferencePointAtIntersectionBuffer[c3_index],
                make_float2(-1.0, 1.0), make_float2(1.0, 1.0));
        }

        if( numSubdivisions == 3 )
        {
            edge0 = SubdivideInterval3(id, type,
                    ReferencePointAtIntersectionBuffer[c0_index],
                    ReferencePointAtIntersectionBuffer[c2_index],
                    make_float2(-1.0, -1.0), make_float2(-1.0, 1.0));
            edge1 = SubdivideInterval3(id, type,
                ReferencePointAtIntersectionBuffer[c1_index],
                ReferencePointAtIntersectionBuffer[c3_index],
                make_float2(1.0, -1.0), make_float2(1.0, 1.0));

            edge2 = SubdivideInterval3(id, type,
                ReferencePointAtIntersectionBuffer[c0_index],
                ReferencePointAtIntersectionBuffer[c1_index],
                make_float2(-1.0, -1.0), make_float2(1.0, -1.0));

            edge3 = SubdivideInterval3(id, type,
                ReferencePointAtIntersectionBuffer[c2_index],
                ReferencePointAtIntersectionBuffer[c3_index],
                make_float2(-1.0, 1.0), make_float2(1.0, 1.0));
        }

        mayContainAnIsovalue = false;
        for(int isoValueIndex = 0; isoValueIndex < Isovalues.size(); ++isoValueIndex)
        {
            float isovalue = Isovalues[isoValueIndex];
            if(edge0.Contains(isovalue) || edge1.Contains(isovalue) ||
                edge2.Contains(isovalue) || edge3.Contains(isovalue) )
            {
                mayContainAnIsovalue = true;
            }
        }

        if( mayContainAnIsovalue )
        {
            raw_color_buffer[launch_index] = ambiguousColor;
            color_buffer[launch_index] = ConvertToColor(raw_color_buffer[launch_index]);
            normal_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
        }
        else
        {
            // Definitely not: WHITE
            raw_color_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(1.0), MAKE_FLOAT(1.0), MAKE_FLOAT(1.0));
            color_buffer[launch_index] = ConvertToColor(raw_color_buffer[launch_index]);
            normal_buffer[launch_index] = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
        }
    }
}

#endif
