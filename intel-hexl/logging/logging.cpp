// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "logging/logging.hpp"

#ifdef HEXL_DEBUG

#include <gflags/gflags.h>

INITIALIZE_EASYLOGGINGPP;

DEFINE_int32(v, 0,
             "enable verbose (DEBUG) logging. Increasing verbosity from 1 to 5 "
             "(maximum debugging)");

el::Configurations LogConfigurationFromFlags() {
  el::Configurations conf;
  conf.setToDefault();
  conf.set(el::Level::Global, el::ConfigurationType::ToFile, "false");

  if (FLAGS_v) {
    el::Loggers::setVerboseLevel(FLAGS_v);
  } else {
    conf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
  }
  return conf;
}

#endif
