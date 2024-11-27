# -*- coding: utf-8 -*-

from fla.layers import (ABCAttention, Attention, BasedLinearAttention,
                        BitAttention, DeltaNet, GatedLinearAttention,
                        GatedSlotAttention, HGRN2Attention, HGRNAttention,
                        LinearAttention, MultiScaleRetention,
                        ReBasedLinearAttention, SeptAttention)
from fla.models import (ABCForCausalLM, ABCModel, DeltaNetForCausalLM,
                        DeltaNetModel, GLAForCausalLM, GLAModel,
                        GSAForCausalLM, GSAModel, HGRN2ForCausalLM, HGRN2Model,
                        HGRNForCausalLM, LinearAttentionForCausalLM,
                        LinearAttentionModel, RetNetForCausalLM, RetNetModel,
                        RWKV6ForCausalLM, RWKV6Model, TransformerForCausalLM,
                        TransformerModel, BitNetForCausalLM, BitNetModel, 
                        SeptNetForCausalLM, SeptNetModel)
                        


__all__ = [
    'ABCAttention',
    'Attention',
    'BasedLinearAttention',
    'BitAttention',
    'DeltaNet',
    'HGRNAttention',
    'HGRN2Attention',
    'GatedLinearAttention',
    'GatedSlotAttention',
    'LinearAttention',
    'MultiScaleRetention',
    'ReBasedLinearAttention',
    'SeptAttention',
    'ABCForCausalLM',
    'ABCModel',
    'BitNetForCausalLM',
    'BitNetModel',
    'DeltaNetForCausalLM',
    'DeltaNetModel',
    'HGRNForCausalLM',
    'HGRNModel',
    'HGRN2ForCausalLM',
    'HGRN2Model',
    'GLAForCausalLM',
    'GLAModel',
    'GSAForCausalLM',
    'GSAModel',
    'LinearAttentionForCausalLM',
    'LinearAttentionModel',
    'RetNetForCausalLM',
    'RetNetModel',
    'RWKV6ForCausalLM',
    'RWKV6Model',
    'TransformerForCausalLM',
    'TransformerModel',
    'SeptNetForCausalLM',
    'SeptNetModel',
    'chunk_gla',
    'chunk_retention',
    'fused_chunk_based',
    'fused_chunk_gla',
    'fused_chunk_retention'
]

__version__ = '0.1'
