#!/usr/bin/env python3

# Copyright (c) 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import subprocess  # nosec B404  # disable import-subprocess check
import sys

from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    subprocess.run([sys.executable, '--',
        str(args.input_dir / 'keras-YOLOv3-model-set/tools/model_converter/convert.py'),
        str(args.input_dir / 'keras-YOLOv3-model-set/cfg/yolov4.cfg'),
        str(args.input_dir / 'yolov4.weights'),
        str(args.output_dir / 'yolo-v4.savedmodel'),
        '--yolo4_reorder',
    ], check=True)


if __name__ == '__main__':
    main()
