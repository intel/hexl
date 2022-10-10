// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#pragma once

namespace intel {
namespace hexl {

/// @brief Number of threads for thread pool
extern uint64_t HEXL_NUM_THREADS;
/// @brief Number of recursive levels as parallel calls
extern uint64_t HEXL_NTT_PARALLEL_DEPTH;
/// @brief Wait time (ms) before thread enters sleep mode
extern uint64_t HEXL_THREAD_WAIT_TIME;

}  // namespace hexl
}  // namespace intel
