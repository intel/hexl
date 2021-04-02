// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdbool.h>

#include "cpuinfo_x86.h"  // NOLINT(build/include_subdir)

namespace intel {
namespace hexl {

static const cpu_features::X86Features features =
    cpu_features::GetX86Info().features;
static const bool has_avx512ifma = features.avx512ifma;
static const bool has_avx512dq =
    features.avx512f && features.avx512dq && features.avx512vl;

}  // namespace hexl
}  // namespace intel
