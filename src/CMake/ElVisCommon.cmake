#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2009 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software. 
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

function(postBuildCopy theTarget copyList toPath)
  if(IS_ABSOLUTE ${toPath}) # absolute toPath
    set(dest ${toPath})
  else() # toPath is relative to target location
    set(dest $<TARGET_FILE_DIR:${theTarget}>/${toPath})
  endif()
  if(NOT EXISTS ${dest})
    add_custom_command(TARGET ${theTarget} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${dest})
  endif()
  foreach(_item ${copyList})
    # Handle target separately.
    if(TARGET ${_item})
      set(_item $<TARGET_FILE:${_item}>)
    endif()
    if(${_item} STREQUAL optimized)
      if(CMAKE_CONFIGURATION_TYPES)
        set(CONDITION1 IF $(Configuration)==Release)
        set(CONDITION2 IF $(Configuration)==RelWithDebInfo)
      endif()
    elseif(${_item} STREQUAL debug)
      if(CMAKE_CONFIGURATION_TYPES)
        set(CONDITION1 IF $(Configuration)==Debug)
      endif()
    else()
      if(IS_DIRECTORY ${_item})
        get_filename_component(dir ${_item} NAME)
        if(NOT EXISTS ${dest}/${dir})
          add_custom_command(TARGET ${theTarget} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${_item} ${dest}/${dir})
        endif()
      else()
        set(COPY_CMD ${CMAKE_COMMAND} -E copy_if_different ${_item} ${dest})
        if(CONDITION2)
          list(APPEND CONDITION1 "(")
          set(ELSECONDITION ")" ELSE "(" ${CONDITION2} "(" ${COPY_CMD} "))")
          set(CONDITION2)
        else()
          set(ELSECONDITION)
        endif()
        add_custom_command(TARGET ${theTarget} POST_BUILD
          COMMAND ${CONDITION1} ${COPY_CMD} ${ELSECONDITION})
      endif()
      set(CONDITION1)
    endif()
  endforeach()
endfunction()
