// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef HEXL_USE_MSVC
#include "util/msvc.hpp"
#elif defined HEXL_USE_GNU
#include "util/gcc.hpp"
#elif defined HEXL_USE_CLANG
#include "util/clang.hpp"
#endif
