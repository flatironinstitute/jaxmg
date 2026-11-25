// Copyright 2026 JAXMg contributors.
// Licensed under the Apache License, Version 2.0.

// Check the public C extension boundary independently of JAX and a GPU runtime.
#include <cassert>
#include <limits>
#include <string>

#include "xla_ffi/runtime.h"

struct XLA_FFI_Error {
  std::string message;
};

namespace {
int requests = 0, lookups = 0;
bool fail_lookup = false;
XLA_FFI_Collectives_Extension extension{};

template <typename Args>
void CheckGroup(const Args* args) {
  assert(args->group_mode == XLA_FFI_GROUP_FLATTENED_ID);
  assert(args->communication_id == 1);
  assert(args->num_groups == 1 && args->groups[0].size == 6);
  for (int64_t rank = 0; rank < 6; ++rank)
    assert(args->groups[0].ids[rank] == rank);
}
XLA_FFI_Error* Request(const XLA_FFI_Collectives_Extension*,
                       XLA_FFI_Communicator_Request_Args* args) {
  CheckGroup(args);
  ++requests;
  return nullptr;
}
XLA_FFI_Error* Get(const XLA_FFI_Collectives_Extension*,
                   XLA_FFI_Communicator_Get_Args* args) {
  CheckGroup(args);
  ++lookups;
  if (fail_lookup) return new XLA_FFI_Error{"communicator unavailable"};
  args->communicator = reinterpret_cast<XLA_FFI_Communicator*>(123);
  return nullptr;
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TestPrepare, jaxmg::XlaCusolverMpPrepare,
    xla::ffi::Ffi::BindPrepare()
        .Attrs()
        .Ctx<xla::ffi::Extension<jaxmg::Collectives>>());

// runtime.cc only needs these CUDA/NCCL entry points for status conversion.
extern "C" const char* cudaGetErrorString(cudaError_t) {
  return "CUDA failure";
}
extern "C" const char* ncclGetErrorString(ncclResult_t) {
  return "NCCL failure";
}
extern "C" ncclResult_t ncclCommUserRank(ncclComm_t, int* rank) {
  *rank = 2;
  return ncclSuccess;
}
extern "C" ncclResult_t ncclCommCount(ncclComm_t, int* count) {
  *count = 6;
  return ncclSuccess;
}
extern "C" ncclResult_t ncclAllReduce(const void*, void*, size_t,
                                      ncclDataType_t, ncclRedOp_t, ncclComm_t,
                                      cudaStream_t) {
  return ncclSuccess;
}

int main() {
  namespace ffi = xla::ffi;
  XLA_FFI_Api api{};
  api.XLA_FFI_Error_Create = [](XLA_FFI_Error_Create_Args* args) {
    return new XLA_FFI_Error{args->message};
  };
  api.XLA_FFI_Error_GetDetails = [](XLA_FFI_Error_GetDetails_Args* args) {
    args->message = args->error->message.c_str();
    args->errc = XLA_FFI_Error_Code_INTERNAL;
  };
  api.XLA_FFI_Error_Destroy = [](XLA_FFI_Error_Destroy_Args* args) {
    delete args->error;
  };
  api.XLA_FFI_InvokeContext_FindExtension =
      [](XLA_FFI_InvokeContext_FindExtension_Args* args) -> XLA_FFI_Error* {
    assert(args->extension_type == XLA_FFI_Extension_Collectives);
    args->extension = &extension.extension_base;
    return nullptr;
  };
  extension.extension_base = ffi::MakeExtensionHeader<jaxmg::Collectives>();
  extension.request_communicator = Request;
  extension.get_communicator = Get;
  int64_t n = 128, rows = 2, cols = 3;
  XLA_FFI_Scalar values[] = {{XLA_FFI_DataType_S64, &n},
                             {XLA_FFI_DataType_S64, &cols},
                             {XLA_FFI_DataType_S64, &rows}};
  XLA_FFI_ByteSpan names[] = {
      {"n", 1}, {"process_cols", 12}, {"process_rows", 12}};
  XLA_FFI_ByteSpan* name_ptrs[] = {&names[0], &names[1], &names[2]};
  XLA_FFI_AttrType types[] = {XLA_FFI_AttrType_SCALAR, XLA_FFI_AttrType_SCALAR,
                              XLA_FFI_AttrType_SCALAR};
  void* attrs[] = {&values[0], &values[1], &values[2]};
  XLA_FFI_CallFrame frame{};
  frame.struct_size = XLA_FFI_CallFrame_STRUCT_SIZE;
  frame.api = &api;
  frame.stage = XLA_FFI_ExecutionStage_PREPARE;
  frame.attrs = {
      XLA_FFI_Attrs_STRUCT_SIZE, nullptr, 3, types, name_ptrs, attrs};
  for (const int minor : {1, 2, 3}) {
    extension.extension_base.id.minor_version = minor;
    ffi::DiagnosticEngine diagnostic;
    auto context = ffi::CtxDecoding<ffi::Extension<jaxmg::Collectives>>::Decode(
        &api, nullptr, diagnostic);
    assert(context.has_value());
    // Preparation accepts additional solver attributes without binding them.
    assert(TestPrepare(&frame) == nullptr);
    const auto comm = jaxmg::BorrowNcclComm("test", *context, 6);
    assert(comm.ok() && *comm == reinterpret_cast<ncclComm_t>(123));
    assert(jaxmg::ValidateBorrowedNcclComm("test", *comm, 2, 6).ok());
    assert(!jaxmg::ValidateBorrowedNcclComm("test", *comm, 1, 6).ok());
  }
  assert(requests == 3 && lookups == 3);
  assert(!jaxmg::Collectives::Support(1, 1));
  assert(!jaxmg::Collectives::Support(0, 0));
  auto context = jaxmg::Collectives::Create(&api, &extension);
  rows = 0;
  assert(!jaxmg::XlaCusolverMpPrepare(ffi::Dictionary(&frame.attrs), context)
              .ok());
  rows = std::numeric_limits<int64_t>::max();
  assert(!jaxmg::XlaCusolverMpPrepare(ffi::Dictionary(&frame.attrs), context)
              .ok());
  fail_lookup = true;
  const auto failed = jaxmg::BorrowNcclComm("test", context, 6);
  assert(!failed.ok() &&
         failed.status().message() == "communicator unavailable");
}
