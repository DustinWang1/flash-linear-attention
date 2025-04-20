# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.septnet.configuration_septnet import SeptNetConfig
from fla.models.septnet.modeling_septnet import SeptNetForCausalLM, SeptNetModel

AutoConfig.register(SeptNetConfig.model_type, SeptNetConfig)
AutoModel.register(SeptNetConfig, SeptNetModel)
AutoModelForCausalLM.register(SeptNetConfig, SeptNetForCausalLM)


__all__ = ['SeptNetConfig', 'SeptNetForCausalLM', 'SeptNetModel']
