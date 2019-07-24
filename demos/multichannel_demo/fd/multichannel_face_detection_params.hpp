// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/// @brief message for probability threshold argument
static const char thresh_output_message[] = "Probability threshold for detections";

/// @brief message for image width
static const char image_width[] = "Show image width";

/// @brief message for image height
static const char image_height[] = "Show image height";

/// @brief message for throughput enabling
static const char enable_throughput[] = "Enable throughput";

/// \brief Flag to output raw scoring results<br>
/// It is an optional parameter. Ignored for human-pose-estimation
DEFINE_double(t, 0.5, thresh_output_message);

/// @brief Define parameter for input image width for text detection model <br>
/// It is a optional parameter
DEFINE_uint32(width, 1280, image_width);

/// @brief Define parameter for input image height for text detection model <br>
/// It is a optional parameter
DEFINE_uint32(height, 720, image_height);

/// @brief Define parameter for input image height for text detection model <br>
/// It is a optional parameter
DEFINE_bool(throughput, false, enable_throughput);
