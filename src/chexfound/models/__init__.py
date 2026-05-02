# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from .vision_transformer import DinoVisionTransformer, vit_small, vit_base, vit_large, vit_giant2
from .model_factory import build_model_from_cfg
