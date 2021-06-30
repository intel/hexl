// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/util/defines.hpp"

#ifdef HEXL_USE_MSVC
#include "hexl/util/msvc.hpp"
#elif defined HEXL_USE_GNU
#include "hexl/util/gcc.hpp"
#elif defined HEXL_USE_CLANG
#include "hexl/util/clang.hpp"
#endif
