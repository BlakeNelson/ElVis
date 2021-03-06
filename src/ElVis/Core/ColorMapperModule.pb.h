// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: ColorMapperModule.proto

#ifndef PROTOBUF_ColorMapperModule_2eproto__INCLUDED
#define PROTOBUF_ColorMapperModule_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3000000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
#include "ColorMap.pb.h"
// @@protoc_insertion_point(includes)

namespace ElVis {
namespace Serialization {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_ColorMapperModule_2eproto();
void protobuf_AssignDesc_ColorMapperModule_2eproto();
void protobuf_ShutdownFile_ColorMapperModule_2eproto();

class ColorMapperModule;

// ===================================================================

class ColorMapperModule : public ::google::protobuf::Message {
 public:
  ColorMapperModule();
  virtual ~ColorMapperModule();

  ColorMapperModule(const ColorMapperModule& from);

  inline ColorMapperModule& operator=(const ColorMapperModule& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const ColorMapperModule& default_instance();

  void Swap(ColorMapperModule* other);

  // implements Message ----------------------------------------------

  inline ColorMapperModule* New() const { return New(NULL); }

  ColorMapperModule* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const ColorMapperModule& from);
  void MergeFrom(const ColorMapperModule& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(ColorMapperModule* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional .ElVis.Serialization.ColorMap color_map = 1;
  bool has_color_map() const;
  void clear_color_map();
  static const int kColorMapFieldNumber = 1;
  const ::ElVis::Serialization::ColorMap& color_map() const;
  ::ElVis::Serialization::ColorMap* mutable_color_map();
  ::ElVis::Serialization::ColorMap* release_color_map();
  void set_allocated_color_map(::ElVis::Serialization::ColorMap* color_map);

  // optional uint32 size = 2;
  void clear_size();
  static const int kSizeFieldNumber = 2;
  ::google::protobuf::uint32 size() const;
  void set_size(::google::protobuf::uint32 value);

  // @@protoc_insertion_point(class_scope:ElVis.Serialization.ColorMapperModule)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  bool _is_default_instance_;
  ::ElVis::Serialization::ColorMap* color_map_;
  ::google::protobuf::uint32 size_;
  mutable int _cached_size_;
  friend void  protobuf_AddDesc_ColorMapperModule_2eproto();
  friend void protobuf_AssignDesc_ColorMapperModule_2eproto();
  friend void protobuf_ShutdownFile_ColorMapperModule_2eproto();

  void InitAsDefaultInstance();
  static ColorMapperModule* default_instance_;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// ColorMapperModule

// optional .ElVis.Serialization.ColorMap color_map = 1;
inline bool ColorMapperModule::has_color_map() const {
  return !_is_default_instance_ && color_map_ != NULL;
}
inline void ColorMapperModule::clear_color_map() {
  if (GetArenaNoVirtual() == NULL && color_map_ != NULL) delete color_map_;
  color_map_ = NULL;
}
inline const ::ElVis::Serialization::ColorMap& ColorMapperModule::color_map() const {
  // @@protoc_insertion_point(field_get:ElVis.Serialization.ColorMapperModule.color_map)
  return color_map_ != NULL ? *color_map_ : *default_instance_->color_map_;
}
inline ::ElVis::Serialization::ColorMap* ColorMapperModule::mutable_color_map() {
  
  if (color_map_ == NULL) {
    color_map_ = new ::ElVis::Serialization::ColorMap;
  }
  // @@protoc_insertion_point(field_mutable:ElVis.Serialization.ColorMapperModule.color_map)
  return color_map_;
}
inline ::ElVis::Serialization::ColorMap* ColorMapperModule::release_color_map() {
  
  ::ElVis::Serialization::ColorMap* temp = color_map_;
  color_map_ = NULL;
  return temp;
}
inline void ColorMapperModule::set_allocated_color_map(::ElVis::Serialization::ColorMap* color_map) {
  delete color_map_;
  color_map_ = color_map;
  if (color_map) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:ElVis.Serialization.ColorMapperModule.color_map)
}

// optional uint32 size = 2;
inline void ColorMapperModule::clear_size() {
  size_ = 0u;
}
inline ::google::protobuf::uint32 ColorMapperModule::size() const {
  // @@protoc_insertion_point(field_get:ElVis.Serialization.ColorMapperModule.size)
  return size_;
}
inline void ColorMapperModule::set_size(::google::protobuf::uint32 value) {
  
  size_ = value;
  // @@protoc_insertion_point(field_set:ElVis.Serialization.ColorMapperModule.size)
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace Serialization
}  // namespace ElVis

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_ColorMapperModule_2eproto__INCLUDED
