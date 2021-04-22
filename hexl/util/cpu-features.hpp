// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdbool.h>

#include <cstdlib>

#include "cpuinfo_x86.h"  // NOLINT(build/include_subdir)

namespace intel {
namespace hexl {

// Use to disable avx512 dispatching at runtime
static const bool disable_avx512dq =
    (std::getenv("HEXL_DISABLE_AVX512DQ") != nullptr);
static const bool disable_avx512ifma =
    disable_avx512dq || (std::getenv("HEXL_DISABLE_AVX512IFMA") != nullptr);
static const bool disable_avx512vbmi2 =
    disable_avx512dq || (std::getenv("HEXL_DISABLE_AVX512VBMI2") != nullptr);

static const cpu_features::X86Features features =
    cpu_features::GetX86Info().features;

static const bool has_avx512dq = features.avx512f && features.avx512dq &&
                                 features.avx512vl && !disable_avx512dq;

static const bool has_avx512ifma = features.avx512ifma && !disable_avx512ifma;

static const bool has_avx512vbmi2 =
    features.avx512vbmi2 && !disable_avx512vbmi2;

}  // namespace hexl
}  // namespace intel
