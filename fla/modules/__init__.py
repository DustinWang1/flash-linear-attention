# -*- coding: utf-8 -*-

from fla.modules.convolution import ImplicitLongConvolution, LongConvolution, ShortConvolution
from fla.modules.fused_bitlinear import BitLinear, FusedBitLinear
from fla.modules.fused_cross_entropy import FusedCrossEntropyLoss
from fla.modules.fused_kl_div import FusedKLDivLoss
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.modules.fused_norm_gate import (
    FusedLayerNormGated,
    FusedLayerNormSwishGate,
    FusedLayerNormSwishGateLinear,
    FusedRMSNormGated,
    FusedRMSNormSwishGate,
    FusedRMSNormSwishGateLinear
)
from fla.modules.fused_septlinear import SeptLinear
from fla.modules.layernorm import GroupNorm, GroupNormLinear, LayerNorm, LayerNormLinear, RMSNorm, RMSNormLinear
from fla.modules.mlp import GatedMLP
from fla.modules.rotary import RotaryEmbedding

__all__ = [
    'ImplicitLongConvolution', 'LongConvolution', 'ShortConvolution', 'SeptLinear',
    'BitLinear', 'FusedBitLinear',
    'FusedCrossEntropyLoss', 'FusedLinearCrossEntropyLoss', 'FusedKLDivLoss',
    'GroupNorm', 'GroupNormLinear', 'LayerNorm', 'LayerNormLinear', 'RMSNorm', 'RMSNormLinear',
    'FusedLayerNormGated', 'FusedLayerNormSwishGate', 'FusedLayerNormSwishGateLinear',
    'FusedRMSNormGated', 'FusedRMSNormSwishGate', 'FusedRMSNormSwishGateLinear',
    'GatedMLP',
    'RotaryEmbedding'
]
