// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <shared_mutex>

namespace intel {
namespace hexl {

using Lock = std::shared_mutex;
using WriteLock = std::unique_lock<Lock>;
using ReadLock = std::shared_lock<Lock>;

class RWLock {
 public:
  RWLock() = default;
  inline ReadLock AcquireRead() { return ReadLock(rw_mutex); }
  inline WriteLock AcquireWrite() { return WriteLock(rw_mutex); }
  inline ReadLock TryAcquireRead() noexcept {
    return ReadLock(rw_mutex, std::try_to_lock);
  }
  inline WriteLock TryAcquireWrite() noexcept {
    return WriteLock(rw_mutex, std::try_to_lock);
  }

 private:
  RWLock(const RWLock& copy) = delete;
  RWLock& operator=(const RWLock& assign) = delete;
  Lock rw_mutex{};
};

}  // namespace hexl
}  // namespace intel
