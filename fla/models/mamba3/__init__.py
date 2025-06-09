# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mamba3.configuration_mamba3 import Mamba3Config
from fla.models.mamba3.modeling_mamba3 import Mamba3ForCausalLM, Mamba3Model

AutoConfig.register(Mamba3Config.model_type, Mamba3Config)
AutoModel.register(Mamba3Config, Mamba3Model)
AutoModelForCausalLM.register(Mamba3Config, Mamba3ForCausalLM)

__all__ = ['Mamba3Config', 'Mamba3ForCausalLM', 'Mamba3Model']
