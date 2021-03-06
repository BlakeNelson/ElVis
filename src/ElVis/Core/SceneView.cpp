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

#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/timer.hpp>

#include <ElVis/Core/Color.h>
#include <ElVis/Core/Light.h>
#include <ElVis/Core/OpenGL.h>
#include <ElVis/Core/Plane.h>
#include <ElVis/Core/PrimaryRayModule.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/RenderModuleFactory.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/SampleVolumeSamplerObject.h>
#include <ElVis/Core/Scene.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/Stat.h>
#include <ElVis/Core/sutil.h>
#include <ElVis/Core/Timer.h>
#include <ElVis/Core/Util.hpp>
#include <ElVis/Core/VolumeRenderingModule.h>
#include <ElVis/Core/RenderModuleFactory.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/IsosurfaceModule.h>
#include <ElVis/Core/IsosurfaceModule.pb.h>
#include <ElVis/Core/PrimaryRayModule.h>
#include <ElVis/Core/PrimaryRayModule.pb.h>
#include <ElVis/Core/LightingModule.h>
#include <ElVis/Core/LightingModule.pb.h>
#include <ElVis/Core/ColorMapperModule.h>
#include <ElVis/Core/ColorMapperModule.pb.h>
#include <ElVis/Core/VolumeRenderingModule.h>
#include <ElVis/Core/VolumeRenderingModule.pb.h>

#include <stdio.h>


#define png_infopp_NULL (png_infopp) NULL
#ifndef int_p_NULL
#define int_p_NULL (int*) NULL
#endif

// Serialization constants
namespace
{
  const std::string VIEW_SETTINGS_KEY_NAME("ViewSettings");
  const std::string RENDER_MODULES_KEY_NAME("RenderModules");
  const std::string RENDER_MODULE_KEY_NAME("RenderModule");
  const std::string RENDER_MODULE_NAMES_KEY_NAME("RenderModuleNames");
  const std::string OPTIX_STACK_SIZE_KEY_NAME("StackSize");
}

// Custom serialization of the render module list.
namespace boost
{
  namespace serialization
  {

    template <class Archive>
    void save(Archive& ar,
              const std::list<boost::shared_ptr<ElVis::RenderModule>>& modules,
              const unsigned int version)
    {

      // Write all of the module names, then serialize the modules.  This is
      // a workaround for the difficulty we fond in getting the boost
      // serialization library to correctly handle derived classes.
      std::vector<std::string> moduleNames;
      for (const auto& pModule : modules)
      {
        moduleNames.push_back(pModule->GetName());
      }
      ar& boost::serialization::make_nvp(
        RENDER_MODULE_NAMES_KEY_NAME.c_str(), moduleNames);

      // TRICKY - for this to work, we need to write each module out in the
      // same order as the names are listed above.
      for (const auto& pModule : modules)
      {
        // TRICKY - serialize by reference to prevent the serilization library
        // from trying to serialize the derived manually.  We handle derived
        // classes manually.
        const auto& moduleReference = *pModule;
        ar& boost::serialization::make_nvp(
          RENDER_MODULE_KEY_NAME.c_str(), moduleReference);
      }
    }

    template <class Archive>
    void load(Archive& ar,
              std::list<boost::shared_ptr<ElVis::RenderModule>>& modules,
              const unsigned int version)
    {
      std::vector<std::string> moduleNames;
      ar& boost::serialization::make_nvp(
        RENDER_MODULE_NAMES_KEY_NAME.c_str(), moduleNames);

      for (const auto& moduleName : moduleNames)
      {
        auto found = std::find_if(
          modules.begin(), modules.end(),
          [moduleName](boost::shared_ptr<ElVis::RenderModule> pModule)
          {
            return pModule->GetName() == moduleName;
          });

        if (found == modules.end()) continue;
        auto pModule = *found;
        auto& moduleReference = *pModule;
        ar& boost::serialization::make_nvp(
          RENDER_MODULE_KEY_NAME.c_str(), moduleReference);
      }
    }
  }
}

BOOST_SERIALIZATION_SPLIT_FREE(
  std::list<boost::shared_ptr<ElVis::RenderModule>>);

namespace ElVis
{
  void SetRasterPositionToLowerLeftCorner()
  {
    GLint newPos[] = {0, 0};
    glWindowPos2iv(newPos);
  }

  const std::string SceneView::ColorBufferName("color_buffer");
  const std::string SceneView::DepthBufferName("depth_buffer");
  const std::string SceneView::NormalBufferName("normal_buffer");
  const std::string SceneView::RawColorBufferName("raw_color_buffer");
  const std::string SceneView::IntersectionBufferName("intersection_buffer");

  SceneView::SceneView()
    : m_scene(),
      m_width(8),
      m_height(8),
      m_viewSettings(new Camera()),
      m_colorBuffer("color_buffer"),
      m_rawColorBuffer(RawColorBufferName),
      m_depthBuffer(DepthBufferName),
      m_normalBuffer(NormalBufferName),
      m_intersectionBuffer(IntersectionBufferName),
      m_sampleBuffer("SampleBuffer"),
      m_elementIdBuffer("ElementIdBuffer"),
      m_elementTypeBuffer("ElementTypeBuffer"),
      m_depthBits(24),
      m_rayGenerationPrograms(),
      m_allRenderModules(),
      m_scalarFieldIndex(0),
      m_passedInitialOptixSetup(false),
      m_faceIntersectionToleranceDirty(true),
      m_faceIntersectionTolerance(1e-5),
      m_headlightColor(82.0 / 255.0, 82.0 / 255.0, 82.0 / 255.0),
      m_headlightColorIsDirty(true),
      m_enableOptiXExceptions(true),
      m_exceptionProgram(),
      // m_backgroundColor(0.0, 0.0, 0.0)
      m_backgroundColor(1.0, 1.0, 1.0),
      m_backgroundColorIsDirty(true),
      m_projectionType(ePerspective),
      m_optixStackSize(18000)
  {
    m_projectionType.OnDirtyFlagChanged.connect(boost::bind(
      &SceneView::HandleSynchedObjectChanged<SceneViewProjection>, this, _1));
    m_viewSettings->OnCameraChanged.connect(
      boost::bind(&SceneView::SetupCamera, this));

    m_enableOptiXExceptions = true;
  }

  SceneView::~SceneView() {}

  void SceneView::SetDepthBufferBits(int newValue)
  {
    m_depthBits = newValue;

    if (m_passedInitialOptixSetup)
    {
      if (!GetScene()) return;

      optixu::Context m_context = GetScene()->GetContext();
      if (!m_context) return;

      m_context["DepthBits"]->setInt(m_depthBits);
    }
  }

  void SceneView::SetScalarFieldIndex(int index)
  {
    m_scalarFieldIndex = index;
    OnSceneViewChanged(*this);

    if (m_passedInitialOptixSetup)
    {
      if (!GetScene()) return;

      optixu::Context m_context = GetScene()->GetContext();
      if (!m_context) return;

      m_context["FieldId"]->setInt(m_scalarFieldIndex);
    }
  }

  void SceneView::Resize(unsigned int width, unsigned int height)
  {
    if (width == m_width && height == m_height)
    {
      return;
    }
    m_width = width;
    m_height = height;
    WindowSizeHasChanged();
  }

  void SceneView::WindowSizeHasChanged()
  {
    m_viewSettings->SetAspectRatio(m_width, m_height);
    if (m_depthBuffer.Initialized())
    {
      m_colorBuffer.SetDimensions(GetWidth(), GetHeight());

      m_depthBuffer.SetDimensions(GetWidth(), GetHeight());
      m_normalBuffer.SetDimensions(GetWidth(), GetHeight());
      m_rawColorBuffer.SetDimensions(GetWidth(), GetHeight());
      m_intersectionBuffer.SetDimensions(GetWidth(), GetHeight());
      m_sampleBuffer.SetDimensions(GetWidth(), GetHeight());
      m_elementIdBuffer.SetDimensions(GetWidth(), GetHeight());
      m_elementTypeBuffer.SetDimensions(GetWidth(), GetHeight());
      BOOST_FOREACH (boost::shared_ptr<RenderModule> module, m_allRenderModules)
      {
        module->Resize(GetWidth(), GetHeight());
      }

      SetupCamera();
    }
    OnWindowSizeChanged(m_width, m_height);
  }

  void SceneView::SetCamera(const Camera& view)
  {
    if (*m_viewSettings == view) return;
    *m_viewSettings = view;
    ViewingParametersHaveChanged();
  }

  void SceneView::ViewingParametersHaveChanged() { SetupCamera(); }

  boost::shared_ptr<PrimaryRayModule> SceneView::GetPrimaryRayModule() const
  {
    BOOST_FOREACH (boost::shared_ptr<RenderModule> module, m_allRenderModules)
    {
      boost::shared_ptr<PrimaryRayModule> result =
        boost::dynamic_pointer_cast<PrimaryRayModule>(module);
      if (result) return result;
    }

    return boost::shared_ptr<PrimaryRayModule>();
  }

  void SceneView::SetupCamera()
  {
    ElVisFloat3 cam_eye = MakeFloat3(GetEye());

    ElVisFloat3 u = MakeFloat3(m_viewSettings->GetU());
    ElVisFloat3 v = MakeFloat3(m_viewSettings->GetV());
    ElVisFloat3 w = MakeFloat3(m_viewSettings->GetW());

    if (GetScene())
    {
      optixu::Context p = GetScene()->GetContext();
      if (p.get())
      {
        SetFloat(p["eye"], cam_eye);
        SetFloat(p["U"], u);
        SetFloat(p["V"], v);
        SetFloat(p["W"], w);
        p["near"]->setFloat(m_viewSettings->GetNear());
        p["far"]->setFloat(m_viewSettings->GetFar());
      }
    }

    OnSceneViewChanged(*this);
  }

  void SceneView::WriteDepthBuffer(const std::string& filePrefix)
  {
    //        unsigned int numEntries = GetWidth()*GetHeight();
    //        uchar3* imageData = new uchar3[numEntries];
    //        float* data = new float[numEntries];

    //        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    //        //SetRasterPositionToLowerLeftCorner();
    //        glReadPixels(0, 0, GetWidth(), GetHeight(), GL_DEPTH_COMPONENT,
    //        GL_FLOAT, data);

    //        float* max = std::max_element(data, data+numEntries);
    //        std::cout << "Max from depth buffer = " << *max << std::endl;
    //        for(unsigned int i = 0; i < numEntries; ++i)
    //        {
    //            imageData[i].x = data[i]*255.0/(*max);
    //            imageData[i].y = data[i]*255.0/(*max);
    //            imageData[i].z = data[i]*255.0/(*max);
    //        }

    //        {

    //            boost::gil::rgb8_image_t forPng(GetWidth(), GetHeight());
    //            boost::gil::copy_pixels(
    //            boost::gil::interleaved_view(GetWidth(), GetHeight(),
    //            (boost::gil::rgb8_pixel_t*)imageData, 3*GetWidth()),
    //            boost::gil::view(forPng));
    //            boost::gil::png_write_view(filePrefix + ".png",
    //            boost::gil::const_view(forPng));
    //        }

    //        delete [] data;

    //        if( m_depthBuffer.Initialized() )
    //        {
    //            data = static_cast<float*>(m_depthBuffer.MapOptiXPointer());
    //            max = std::max_element(data, data+numEntries);
    //            std::cout << "Max from optix depth buffer = " << *max <<
    //            std::endl;
    //            for(unsigned int i = 0; i < numEntries; ++i)
    //            {
    //                imageData[i].x = data[i]*255.0/(*max);
    //                imageData[i].y = data[i]*255.0/(*max);
    //                imageData[i].z = data[i]*255.0/(*max);
    //            }
    //            m_depthBuffer.UnmapOptiXPointer();

    //            {

    //                boost::gil::rgb8_image_t forPng(GetWidth(), GetHeight());
    //                boost::gil::copy_pixels(
    //                boost::gil::interleaved_view(GetWidth(), GetHeight(),
    //                (boost::gil::rgb8_pixel_t*)imageData, 3*GetWidth()),
    //                boost::gil::view(forPng));
    //                boost::gil::png_write_view(filePrefix + "_optix.png",
    //                boost::gil::const_view(forPng));
    //            }
    //        }

    //        delete [] imageData;
  }

  void SceneView::WriteColorBufferToFile(const std::string& fileName)
  {

    if (!m_colorBuffer.Initialized()) return;

    auto colorBuffer = m_colorBuffer.Map();

    if (!colorBuffer) return;

    boost::gil::rgba8_image_t forPng(GetWidth(), GetHeight());
    boost::gil::copy_pixels(
      boost::gil::interleaved_view(
        GetWidth(), GetHeight(), (boost::gil::rgba8_pixel_t*)colorBuffer.get(),
        4 * GetWidth()),
      boost::gil::view(forPng));
    auto flippedView =
      boost::gil::flipped_up_down_view(boost::gil::const_view(forPng));
    boost::gil::png_write_view(
      fileName + ".png", flippedView);

    std::cout << "Done writing images." << std::endl;
  }

  Timer SceneView::Draw()
  {
    Timer result;

    if (!GetScene()) return result;
    if (!GetScene()->GetModel()) return result;

    try
    {
      result.Start();

      PrepareForDisplay();

      bool continueWithRender = false;
      BOOST_FOREACH (boost::shared_ptr<RenderModule> module, m_allRenderModules)
      {
        if (module->GetEnabled() && module->GetRenderRequired())
        {
          continueWithRender = true;
        }
      }
      if (!continueWithRender) return result;

      //            ClearDepthBuffer();
      ClearColorBuffer();
      BOOST_FOREACH (boost::shared_ptr<RenderModule> module, m_allRenderModules)
      {
        if (module->GetEnabled())
        {
          module->Setup(this);
        }
      }
      GetScene()->SynchronizeWithOptiXIfNeeded();
      SynchronizeWithGPUIfNeeded(GetScene()->GetContext());

      ElVis::Timer t;
      t.Start();
      GetScene()->GetContext()->compile();
      t.Stop();
      // std::cout << "Time to Compile: " << t.TimePerTest(1) << std::endl;

      BOOST_FOREACH (boost::shared_ptr<RenderModule> module, m_allRenderModules)
      {
        if (module->GetEnabled())
        {
          // std::string moduleName = typeid(*module).name();
          // printf("Starting Module %s.\n", moduleName.c_str());

          Timer t;
          t.Start();
          module->Render(this);
          t.Stop();
          double elapsed = t.TimePerTest(1);
          m_timings[module] = elapsed;

          // printf("Module %s time = %e.\n", moduleName.c_str(), elapsed);
        }
      }
      result.Stop();
      return result;
    }
    catch (optixu::Exception& e)
    {
      std::cout << "Exception encountered in SceneView::Draw." << std::endl;
      std::cerr << e.getErrorString() << std::endl;
      std::cout << e.getErrorString().c_str() << std::endl;
    }
    catch (std::exception& e)
    {
      std::cout << "Exception encountered in SceneView::Draw." << std::endl;
      std::cout << e.what() << std::endl;
    }
    catch (...)
    {
      std::cout << "Exception encountered in SceneView::Draw" << std::endl;
    }
    return result;
  }

  void SceneView::AddRenderModule(boost::shared_ptr<RenderModule> module)
  {
    m_allRenderModules.push_back(module);
    module->OnModuleChanged.connect(
      boost::bind(&SceneView::HandleRenderModuleChanged, this, _1));
    module->OnRenderFlagsChanged.connect(
      boost::bind(&SceneView::HandleRenderModuleChanged, this, _1));
    // If anything changes in the scene or the view, then this is an indication
    // that the modules
    // need to re-render, but don't necessarily need to redo their setup.

    OnWindowSizeChanged.connect(
      boost::bind(&RenderModule::SetRenderRequired, module.get()));

    OnSceneViewChanged.connect(
      boost::bind(&RenderModule::SetRenderRequired, module));
  }

  void SceneView::HandleRenderModuleChanged(const RenderModule&)
  {
    OnNeedsRedraw(*this);
    // OnSceneViewChanged(*this);
  }

  void SceneView::ClearDepthBuffer()
  {
    if (m_depthBuffer.Initialized())
    {
      auto data = m_depthBuffer.Map();
      for (unsigned int i = 0; i < GetWidth() * GetHeight(); ++i)
      {
        data[i] = 1.0f;
      }
    }
  }

  void SceneView::ClearColorBuffer()
  {
    if (m_colorBuffer.Initialized())
    {
      auto data = m_colorBuffer.Map();
      for (unsigned int i = 0; i < GetWidth() * GetHeight(); ++i)
      {
        data[i].x = 255;
        data[i].y = 255;
        data[i].z = 255;
        data[i].w = 0;
      }
    }
  }

  void SceneView::DisplayBuffersToScreen()
  {
    if (!m_colorBuffer.Initialized() || !m_depthBuffer.Initialized())
    {
      return;
    }

    try
    {
      //           // First, write the new depth buffer.
      //           // Note that this must happen before any other OpenGL related
      //           // activities since it will erase the entire depth buffer.
      //           // There is a way to do this with the stencil buffer so that
      //           other OpenGL
      //           // stuff can come first, but for now we'll assume this plugin
      //           is always
      //           // first.

      // Debugging code.
      //            float depthBias;
      //            float depthScale;
      //            glGetFloatv(GL_DEPTH_SCALE, &depthScale);
      //            glGetFloatv(GL_DEPTH_BIAS, &depthBias);
      //            std::cout << "Depth Scale = " << depthScale << ", bias = "
      //            << depthBias << std::endl;
      //            float depthRange[2];
      //            glGetFloatv(GL_DEPTH_RANGE, depthRange);
      //            std::cout << "Depth Range (" << depthRange[0] << ", " <<
      //            depthRange[1] << ")" << std::endl;

      // Disable the depth test before writing so that all updates
      // will make it.
      GLboolean depthTestEnabled = true;
      glGetBooleanv(GL_DEPTH_TEST, &depthTestEnabled);

      //////////////////////////////////
      // Depth Buffer Writing
      //////////////////////////////////
      //            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      //            SetRasterPositionToLowerLeftCorner();

      //            // Disable the color buffer to prevent upddates there.
      //            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

      //            if( depthTestEnabled )
      //            {
      //               glDisable(GL_DEPTH_TEST);
      //            }

      //            void* depthData = m_depthBuffer.map();
      //            glDrawPixels(GetWidth(), GetHeight(), GL_DEPTH_COMPONENT,
      //            GL_FLOAT, depthData);
      //            m_depthBuffer.unmap();

      //////////////////////////////////////////////
      // Color buffer writing
      ////////////////////////////////////////////////
      glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

      // Now that the depth is written, write the color buffer.
      glDepthMask(GL_FALSE);
      SetRasterPositionToLowerLeftCorner();

      glDrawBuffer(GL_BACK);
      auto colorData = m_colorBuffer.Map();
      glDrawPixels(GetWidth(), GetHeight(), GL_RGBA, GL_UNSIGNED_BYTE,
                   (void*)colorData.get());

      // GLenum error = glGetError();

      glDepthMask(GL_TRUE);
      if (depthTestEnabled)
      {
        glEnable(GL_DEPTH_TEST);
      }

      ////////////////////////
      // Trial code to see how drawing individual points works.
      ////////////////////////
      //            glEnable(GL_COLOR_MATERIAL);
      //            ElVisFloat3* intersectionData =
      //            static_cast<ElVisFloat3*>(m_intersectionBuffer.map());
      //            uchar4* color = static_cast<uchar4*>(m_colorBuffer.map());
      //            ElVisFloat* scalars =
      //            static_cast<ElVisFloat*>(m_sampleBuffer.map());

      //            glBegin(GL_POINTS);
      //            for(int i = 0; i < GetWidth()*GetHeight(); ++i)
      //            {
      //                std::cout << scalars[i] << std::endl;
      //                if( scalars[i] < ELVIS_FLOAT_COMPARE )
      //                {
      //                    std::cout << intersectionData[i].x << " " <<
      //                    intersectionData[i].y << " " <<
      //                    intersectionData[i].z << std::endl;

      //                    glColor3f(static_cast<ElVisFloat>(color[i].x)/255.0f,
      //                              static_cast<ElVisFloat>(color[i].y)/255.0f,
      //                              static_cast<ElVisFloat>(color[i].z)/255.0f);
      //                    glVertex3f(intersectionData[i].x,
      //                    intersectionData[i].y, intersectionData[i].z);
      //                }
      //            }
      //            glEnd();
      //            m_intersectionBuffer.unmap();
      //            m_colorBuffer.unmap();
      //            m_sampleBuffer.unmap();
    }
    catch (optixu::Exception& e)
    {
      std::cerr << e.getErrorString() << std::endl;
    }
    catch (...)
    {
      std::cout << "Caught exception in display buffers." << std::endl;
    }
  }

  WorldPoint SceneView::GetIntersectionPoint(unsigned int pixel_x,
                                             unsigned int pixel_y)
  {
    ElVisFloat3 element = m_intersectionBuffer(pixel_x, m_height - pixel_y - 1);
    return WorldPoint(element.x, element.y, element.z);
  }

  WorldPoint SceneView::GetNormal(unsigned int pixel_x, unsigned int pixel_y)
  {
    ElVisFloat3 element = m_normalBuffer(pixel_x, m_height - pixel_y - 1);
    return WorldPoint(element.x, element.y, element.z);
  }

  int SceneView::GetElementId(unsigned int pixel_x, unsigned int pixel_y)
  {
    return m_elementIdBuffer(pixel_x, m_height - pixel_y - 1);
  }

  int SceneView::GetElementTypeId(unsigned int pixel_x, unsigned int pixel_y)
  {
    return m_elementTypeBuffer(pixel_x, m_height - pixel_y - 1);
  }

  ElVisFloat SceneView::GetScalarSample(unsigned int pixel_x,
                                        unsigned int pixel_y)
  {
    return m_sampleBuffer(pixel_x, m_height - pixel_y - 1);
  }

  void SceneView::SetOptixStackSize(int size)
  {
    if (size <= 0) return;
    m_optixStackSize = size;

    auto context = GetContext();
    if (context)
    {
      context->setStackSize(m_optixStackSize);
      OnSceneViewChanged(*this);
    }
  }


  void SceneView::PrepareForDisplay()
  {
    if (m_colorBuffer.Initialized())
    {
      // We have already setup the view.
      return;
    }

    try
    {
      optixu::Context m_context = GetScene()->GetContext();
      std::cout << "Setting stack size: " << m_optixStackSize << std::endl;
      m_context->setStackSize(m_optixStackSize);

      // Ray Type 0 - Primary rays that intersect actual geometry.  Closest
      // hit programs determine exactly how the geometry is handled.
      // Ray Type 1 - Rays that find the current element and evaluate the scalar
      // value at a point.
      // TODO - Query the modules to see if they require a specific ray type
      // similar to what we do with the ray generation programs.
      m_context->setRayTypeCount(3);

      // unsigned int memAlloc = 0;

      // Setup various buffers the modules can use.
      // TODO - making the color and depth buffers INPUT/OUTPUT slows thing down
      // according to the documentation (the buffers are stored in main memory).
      // How does the SDK display to OpenGL without doing this?
      m_colorBuffer.SetContext(m_context);
      m_colorBuffer.SetDimensions(GetWidth(), GetHeight());
      m_rawColorBuffer.SetContext(m_context);
      m_rawColorBuffer.SetDimensions(GetWidth(), GetHeight());
      // unsigned int colorBuf = GetWidth() * GetHeight() * 4;

      m_depthBuffer.SetContext(m_context);
      m_depthBuffer.SetDimensions(GetWidth(), GetHeight());
      // unsigned int depthBuffSize = GetWidth() * GetHeight() * sizeof(float);

      m_sampleBuffer.SetContext(m_context);
      m_sampleBuffer.SetDimensions(GetWidth(), GetHeight());
      // unsigned int sampleBufferSize = GetWidth() * GetHeight() *
      // sizeof(float);

      m_normalBuffer.SetContext(m_context);
      m_normalBuffer.SetDimensions(GetWidth(), GetHeight());
      // unsigned int normalBufferSize = GetWidth() * GetHeight() *
      // sizeof(float)*3;

      m_intersectionBuffer.SetContext(m_context);
      m_intersectionBuffer.SetDimensions(GetWidth(), GetHeight());
      // unsigned int intersectionSize = GetWidth() * GetHeight() *
      // sizeof(float)*3;

      m_elementIdBuffer.SetContext(m_context);
      m_elementIdBuffer.SetDimensions(GetWidth(), GetHeight());

      m_elementTypeBuffer.SetContext(m_context);
      m_elementTypeBuffer.SetDimensions(GetWidth(), GetHeight());

      m_exceptionProgram =
        PtxManager::LoadProgram(m_context, GetPTXPrefix(), "ExceptionProgram");
      m_context->setPrintLaunchIndex(-1, -1, -1);
      SetupCamera();
      m_context["DepthBits"]->setInt(m_depthBits);
      m_context["FieldId"]->setInt(m_scalarFieldIndex);

      ElVisFloat3 bgColor;
      bgColor.x = m_backgroundColor.Red();
      bgColor.y = m_backgroundColor.Green();
      bgColor.z = m_backgroundColor.Blue();

      SetFloat(m_context["BGColor"], bgColor);

      // This was an attempt to fix some issues we were seeing with isosurfaces.
      // The timeout is executed on the host at a user-defined intervale, which
      // call allow the host to display progress information.  The theory was
      // that the OptiX kernel was timing out, but setting this call did not
      // seem to help.  However, it would be good to integrate this into ElVis for
      // longer running scenes and user interactivity.
      // m_context->setTimeoutCallback([]() { return 0; }, .1);

      m_passedInitialOptixSetup = true;
    }
    catch (optixu::Exception& e)
    {
      std::cerr << e.getErrorString() << std::endl;
      throw;
    }
  }

  RayGeneratorProgram SceneView::AddRayGenerationProgram(
    const std::string& programName)
  {
    auto found = m_rayGenerationPrograms.find(programName);
    if (found != m_rayGenerationPrograms.end())
    {
      return (*found).second;
    }

    RayGeneratorProgram result;

    optixu::Context m_context = GetScene()->GetContext();
    result.Program =
      PtxManager::LoadProgram(m_context, GetPTXPrefix(), programName.c_str());
    result.Index = static_cast<unsigned int>(m_rayGenerationPrograms.size());

    m_rayGenerationPrograms[programName] = result;

    m_context->setEntryPointCount(
      static_cast<unsigned int>(m_rayGenerationPrograms.size()));
    m_context->setRayGenerationProgram(result.Index, result.Program);
    return result;
  }

  ElVisFloat SceneView::GetFaceIntersectionTolerance() const
  {
    return m_faceIntersectionTolerance;
  }

  void SceneView::SetFaceIntersectionTolerance(ElVisFloat value)
  {
    if (value != m_faceIntersectionTolerance)
    {
      m_faceIntersectionTolerance = value;
      m_faceIntersectionToleranceDirty = true;
      OnSceneViewChanged(*this);
    }
  }

  void SceneView::SynchronizeWithGPUIfNeeded(optixu::Context context)
  {
    if (m_faceIntersectionToleranceDirty)
    {
      SetFloat(context["FaceTolerance"], m_faceIntersectionTolerance);
      m_faceIntersectionToleranceDirty = false;
    }

    if (m_headlightColorIsDirty)
    {
      ElVisFloat3 c;
      c.x = m_headlightColor.Red();
      c.y = m_headlightColor.Green();
      c.z = m_headlightColor.Blue();

      SetFloat(context["HeadlightColor"], c);
      m_headlightColorIsDirty = false;
    }

    if (m_backgroundColorIsDirty)
    {
      ElVisFloat3 bgColor;
      bgColor.x = m_backgroundColor.Red();
      bgColor.y = m_backgroundColor.Green();
      bgColor.z = m_backgroundColor.Blue();

      SetFloat(context["BGColor"], bgColor);
      m_backgroundColorIsDirty = false;
    }

    if (m_projectionType.IsDirty())
    {
      ElVis::SceneViewProjection data = *m_projectionType;
      context["ProjectionType"]->setUserData(
        sizeof(ElVis::SceneViewProjection), &data);
      m_projectionType.MarkClean();
    }

    if (context->getExceptionEnabled(RT_EXCEPTION_ALL) !=
        m_enableOptiXExceptions)
    {
      std::cout << "Setting exception flag to "
                << (m_enableOptiXExceptions ? "true" : "false") << std::endl;
      context->setExceptionEnabled(RT_EXCEPTION_ALL, m_enableOptiXExceptions);
    }

    for (unsigned int i = 0; i < context->getEntryPointCount(); ++i)
    {
      context->setExceptionProgram(i, m_exceptionProgram);
    }
    context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
  }

  const Color& SceneView::GetHeadlightColor() const { return m_headlightColor; }

  void SceneView::SetHeadlightColor(const Color& newValue)
  {
    if (m_headlightColor != newValue)
    {
      m_headlightColor = newValue;
      m_headlightColorIsDirty = true;
      OnSceneViewChanged(*this);
    }
  }

  void SceneView::SetBackgroundColor(const Color& newValue)
  {
    if (m_backgroundColor != newValue)
    {
      m_backgroundColor = newValue;
      m_backgroundColorIsDirty = true;
      OnSceneViewChanged(*this);
    }
  }

  Stat SceneView::CalculateScalarSampleStats()
  {
    return Stat();
    //        if( !m_sampleBuffer.Initialized() ) return Stat();

    //        ElVisFloat* samples =
    //        static_cast<ElVisFloat*>(m_sampleBuffer.MapOptiXPointer());
    //        int size = GetWidth()*GetHeight();
    //        Stat result(samples, ELVIS_FLOAT_MAX, size);
    //        m_sampleBuffer.UnmapOptiXPointer();
    //        return result;
  }

  void SceneView::SetProjectionType(SceneViewProjection type)
  {
    m_projectionType = type;
  }

  SceneViewProjection SceneView::GetProjectionType() const
  {
    return *m_projectionType;
  }

  void SceneView::UpdateCamera(const ElVis::Serialization::Camera& data)
  {
    m_viewSettings->Deserialize(data);
  }

  std::unique_ptr<ElVis::Serialization::SceneView> SceneView::Serialize() const
  {
    std::unique_ptr<ElVis::Serialization::SceneView> pResult(new ElVis::Serialization::SceneView());
    pResult->set_stack_size(m_optixStackSize);
    pResult->set_allocated_view_settings(m_viewSettings->Serialize().release());

    for(const auto& pRenderModule : m_allRenderModules)
    {
      auto pNewSerializedModule = pResult->add_render_modules();
      *pNewSerializedModule = *pRenderModule->Serialize();
    }
    return pResult;
  }

  namespace
  {
    template<typename T, typename SerializationType>
    void DeserializeToExistingRenderModule(
              std::list<boost::shared_ptr<ElVis::RenderModule>>& modules,
        const ElVis::Serialization::RenderModule& data)
    {
        auto found = std::find_if(
          modules.begin(), modules.end(),
          [](boost::shared_ptr<ElVis::RenderModule> pModule) -> bool
          {
            return boost::dynamic_pointer_cast<T>(pModule).get() != nullptr;
          });

        if (found == modules.end()) throw std::runtime_error("Invalid deserialized module.");
        auto pModule = *found;
        pModule->Deserialize(data);

        SerializationType moduleData;
        data.concrete_module().UnpackTo(&moduleData);
        auto pCastModule = boost::dynamic_pointer_cast<T>(pModule);
        pCastModule->Deserialize(moduleData);
    }
  } // end anonymous namespace

  void SceneView::Deserialize(const ElVis::Serialization::SceneView& input)
  {
    SetOptixStackSize(input.stack_size());
    m_viewSettings->Deserialize(input.view_settings());

    for(int i = 0; i < input.render_modules_size(); ++i)
    {
      const auto& renderModuleData = input.render_modules(i);

      if( renderModuleData.concrete_module().Is<ElVis::Serialization::IsosurfaceModule>() )
      {
        DeserializeToExistingRenderModule<IsosurfaceModule, ElVis::Serialization::IsosurfaceModule>(this->m_allRenderModules, renderModuleData);
      }

      if( renderModuleData.concrete_module().Is<ElVis::Serialization::PrimaryRayModule>() )
      {
        DeserializeToExistingRenderModule<PrimaryRayModule, ElVis::Serialization::PrimaryRayModule>(this->m_allRenderModules, renderModuleData);
      }

      if( renderModuleData.concrete_module().Is<ElVis::Serialization::ColorMapperModule>() )
      {
        DeserializeToExistingRenderModule<ColorMapperModule, ElVis::Serialization::ColorMapperModule>(this->m_allRenderModules, renderModuleData);
      }

      if( renderModuleData.concrete_module().Is<ElVis::Serialization::LightingModule>() )
      {
        DeserializeToExistingRenderModule<LightingModule, ElVis::Serialization::LightingModule>(this->m_allRenderModules, renderModuleData);
      }

      if( renderModuleData.concrete_module().Is<ElVis::Serialization::VolumeRenderingModule>() )
      {
        DeserializeToExistingRenderModule<VolumeRenderingModule, ElVis::Serialization::VolumeRenderingModule>(this->m_allRenderModules, renderModuleData);
      }
    }
    OnSceneViewChanged(*this);
  }
}
