// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "thread-pool/thread-pool-vars-util.hpp"

namespace intel {
namespace hexl {
#ifdef HEXL_MULTI_THREADING
int HEXL_NUM_THREADS = setup_num_threads("HEXL_NUM_THREADS");
int HEXL_NTT_PARALLEL_DEPTH = setup_ntt_calls("HEXL_NTT_PARALLEL_DEPTH");
#else
int HEXL_NUM_THREADS = 1;
int HEXL_NTT_PARALLEL_DEPTH = 0;
#endif
}  // namespace hexl
}  // namespace intel
