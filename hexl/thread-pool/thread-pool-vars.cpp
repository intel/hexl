// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "thread-pool/thread-pool-vars-util.hpp"

namespace intel {
namespace hexl {

uint16_t HEXL_THREAD_WAIT_TIME = 20;  // ~ 100x wake up time on ICX machine

#ifdef HEXL_MULTI_THREADING
uint HEXL_NUM_THREADS = setup_num_threads("HEXL_NUM_THREADS");
uint HEXL_NTT_PARALLEL_DEPTH = setup_ntt_calls("HEXL_NTT_PARALLEL_DEPTH");
#else
uint HEXL_NUM_THREADS = 1;
uint HEXL_NTT_PARALLEL_DEPTH = 0;
#endif

}  // namespace hexl
}  // namespace intel
