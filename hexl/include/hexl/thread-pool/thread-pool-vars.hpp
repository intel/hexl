// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace intel {
namespace hexl {

// Number of threads for thread pool
extern int HEXL_NUM_THREADS;
// Number of recursive levels as parallel calls
extern int HEXL_NTT_PARALLEL_DEPTH;
// Wait time (ms) before thread enters sleep mode
extern uint64_t HEXL_THREAD_WAIT_TIME;

}  // namespace hexl
}  // namespace intel
