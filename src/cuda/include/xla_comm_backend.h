// Copyright 2026 JAXMg contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Compatibility umbrella header for the XLA communicator cuSOLVERMp backend.
//
// New code should include the narrower headers directly:
//
//   * include/xla_comm_common.h
//   * memory_redist/memory_redist.h
//   * cusolvermp_routines/cusolvermp_routines.h
//
// Existing translation units still include this umbrella while the backend is
// being reorganized. Keeping it as a thin include-only layer avoids a noisy
// include-path-only change and makes the intended internal boundaries explicit.

#ifndef JAXMG_XLA_COMM_BACKEND_H_
#define JAXMG_XLA_COMM_BACKEND_H_

#include "../cusolvermp_routines/cusolvermp_routines.h"
#include "../memory_redist/memory_redist.h"
#include "xla_comm_common.h"

#endif  // JAXMG_XLA_COMM_BACKEND_H_
