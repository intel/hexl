// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/experimental/seal/locks.hpp"
#include "ntt/ntt-internal.hpp"

namespace intel {
namespace hexl {

struct HashPair {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    auto hash1 = std::hash<T1>{}(p.first);
    auto hash2 = std::hash<T2>{}(p.second);
    return hash_combine(hash1, hash2);
  }

  // Golden Ratio Hashing with seeds
  static std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }
};

NTT& GetNTT(size_t N, uint64_t modulus) {
  static std::unordered_map<std::pair<uint64_t, uint64_t>, NTT, HashPair>
      ntt_cache;
  static RWLock ntt_cache_locker;

  std::pair<uint64_t, uint64_t> key{N, modulus};

  // Enable shared access to NTT already present
  {
    ReadLock reader_lock(ntt_cache_locker.AcquireRead());
    auto ntt_it = ntt_cache.find(key);
    if (ntt_it != ntt_cache.end()) {
      return ntt_it->second;
    }
  }

  // Deal with NTT not yet present
  WriteLock write_lock(ntt_cache_locker.AcquireWrite());

  // Check ntt_cache for value (may be added by another thread)
  auto ntt_it = ntt_cache.find(key);
  if (ntt_it == ntt_cache.end()) {
    NTT ntt(N, modulus);
    ntt_it = ntt_cache.emplace(std::move(key), std::move(ntt)).first;
  }
  return ntt_it->second;
}

}  // namespace hexl
}  // namespace intel
