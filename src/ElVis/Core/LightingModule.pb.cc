// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: LightingModule.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "LightingModule.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace ElVis {
namespace Serialization {

namespace {

const ::google::protobuf::Descriptor* LightingModule_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  LightingModule_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_LightingModule_2eproto() {
  protobuf_AddDesc_LightingModule_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "LightingModule.proto");
  GOOGLE_CHECK(file != NULL);
  LightingModule_descriptor_ = file->message_type(0);
  static const int LightingModule_offsets_[1] = {
  };
  LightingModule_reflection_ =
    ::google::protobuf::internal::GeneratedMessageReflection::NewGeneratedMessageReflection(
      LightingModule_descriptor_,
      LightingModule::default_instance_,
      LightingModule_offsets_,
      -1,
      -1,
      -1,
      sizeof(LightingModule),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(LightingModule, _internal_metadata_),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(LightingModule, _is_default_instance_));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_LightingModule_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
      LightingModule_descriptor_, &LightingModule::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_LightingModule_2eproto() {
  delete LightingModule::default_instance_;
  delete LightingModule_reflection_;
}

void protobuf_AddDesc_LightingModule_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\024LightingModule.proto\022\023ElVis.Serializat"
    "ion\"\020\n\016LightingModuleb\006proto3", 69);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "LightingModule.proto", &protobuf_RegisterTypes);
  LightingModule::default_instance_ = new LightingModule();
  LightingModule::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_LightingModule_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_LightingModule_2eproto {
  StaticDescriptorInitializer_LightingModule_2eproto() {
    protobuf_AddDesc_LightingModule_2eproto();
  }
} static_descriptor_initializer_LightingModule_2eproto_;

namespace {

static void MergeFromFail(int line) GOOGLE_ATTRIBUTE_COLD;
static void MergeFromFail(int line) {
  GOOGLE_CHECK(false) << __FILE__ << ":" << line;
}

}  // namespace


// ===================================================================

#ifndef _MSC_VER
#endif  // !_MSC_VER

LightingModule::LightingModule()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:ElVis.Serialization.LightingModule)
}

void LightingModule::InitAsDefaultInstance() {
  _is_default_instance_ = true;
}

LightingModule::LightingModule(const LightingModule& from)
  : ::google::protobuf::Message(),
    _internal_metadata_(NULL) {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:ElVis.Serialization.LightingModule)
}

void LightingModule::SharedCtor() {
    _is_default_instance_ = false;
  _cached_size_ = 0;
}

LightingModule::~LightingModule() {
  // @@protoc_insertion_point(destructor:ElVis.Serialization.LightingModule)
  SharedDtor();
}

void LightingModule::SharedDtor() {
  if (this != default_instance_) {
  }
}

void LightingModule::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* LightingModule::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return LightingModule_descriptor_;
}

const LightingModule& LightingModule::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_LightingModule_2eproto();
  return *default_instance_;
}

LightingModule* LightingModule::default_instance_ = NULL;

LightingModule* LightingModule::New(::google::protobuf::Arena* arena) const {
  LightingModule* n = new LightingModule;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void LightingModule::Clear() {
}

bool LightingModule::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:ElVis.Serialization.LightingModule)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
  handle_unusual:
    if (tag == 0 ||
        ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
        ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
      goto success;
    }
    DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));
  }
success:
  // @@protoc_insertion_point(parse_success:ElVis.Serialization.LightingModule)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:ElVis.Serialization.LightingModule)
  return false;
#undef DO_
}

void LightingModule::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:ElVis.Serialization.LightingModule)
  // @@protoc_insertion_point(serialize_end:ElVis.Serialization.LightingModule)
}

::google::protobuf::uint8* LightingModule::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:ElVis.Serialization.LightingModule)
  // @@protoc_insertion_point(serialize_to_array_end:ElVis.Serialization.LightingModule)
  return target;
}

int LightingModule::ByteSize() const {
  int total_size = 0;

  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void LightingModule::MergeFrom(const ::google::protobuf::Message& from) {
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  const LightingModule* source = 
      ::google::protobuf::internal::DynamicCastToGenerated<const LightingModule>(
          &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void LightingModule::MergeFrom(const LightingModule& from) {
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
}

void LightingModule::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void LightingModule::CopyFrom(const LightingModule& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool LightingModule::IsInitialized() const {

  return true;
}

void LightingModule::Swap(LightingModule* other) {
  if (other == this) return;
  InternalSwap(other);
}
void LightingModule::InternalSwap(LightingModule* other) {
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata LightingModule::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = LightingModule_descriptor_;
  metadata.reflection = LightingModule_reflection_;
  return metadata;
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// LightingModule

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace Serialization
}  // namespace ElVis

// @@protoc_insertion_point(global_scope)
