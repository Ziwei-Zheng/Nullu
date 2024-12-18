from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from qwen_vl_chat.modeling_qwen import make_context
from model.base import LargeMultimodalModel, create_hook


class Qwen_VL_Chat(LargeMultimodalModel):

    def __init__(self, args):
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)
    
    @torch.no_grad()
    def _basic_forward(self, image_path, prompt, answer=None, return_dict=False):
        query = self.tokenizer.from_list_format([
            {'image': image_path}, # Either a local path or an url
            {'text': prompt},
        ])
        max_window_size = self.model.generation_config.max_window_size
        raw_text, context_tokens = make_context(
            self.tokenizer,
            query,
            history=None,
            system="You are a helpful assistant.",
            max_window_size=max_window_size,
            chat_format=self.model.generation_config.chat_format,
        )
        input_ids = torch.tensor([context_tokens]).to(self.model.device)
    
        outputs = self.model(
            input_ids,
            return_dict=return_dict,
            output_attentions=return_dict,
            output_hidden_states=return_dict
        )
        return outputs
    
    @torch.no_grad()
    def chat(self, image_path, prompt):
        query = self.tokenizer.from_list_format([
            {'image': image_path}, # Either a local path or an url
            {'text': prompt},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response
    
    def register_hooks(self):
        self.model.attn_heads, self.model.attn_residual, self.model.mlp_residual, self.model.vit_satt = [], [], [], []
        attn_head_hook = create_hook(self.model.attn_heads, loc='input')
        attn_residual_hook = create_hook(self.model.attn_residual)
        mlp_residual_hook = create_hook(self.model.mlp_residual)
        vit_forward_hook = create_hook(self.model.vit_satt, loc='input')
        self.hooks = []
        for layer in self.model.transformer.h:
            self.hooks.append(layer.attn.c_proj.register_forward_hook(attn_head_hook))
            self.hooks.append(layer.attn.register_forward_hook(attn_residual_hook))
            self.hooks.append(layer.mlp.register_forward_hook(mlp_residual_hook))
        for layer in self.model.transformer.visual.transformer.resblocks:
            self.hooks.append(layer.attn.out_proj.register_forward_hook(vit_forward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_activations(self, image_path, prompt, answer=None):
        self.register_hooks()
        outputs = self._basic_forward(image_path, prompt, answer, return_dict=True)
        attn_heads = torch.cat(self.model.attn_heads).reshape(32, -1, 32, 128)   # [32, seq_len, 4096] -> [32, seq_len, 32, 128]
        attn_residual = torch.cat(self.model.attn_residual)   # [32, seq_len, 4096]
        mlp_residual = torch.stack(self.model.mlp_residual)   # [32, seq_len, 4096]
        hidden_states = torch.stack(outputs.hidden_states)[1:, 0]   # [32, seq_len, 4096]
        vit_attn_heads = torch.cat(self.model.vit_satt).reshape(48, -1, 16, 104)   # [48, 1024, 1, 1664] -> [48, 1024, 16, 104]
        self.remove_hooks()
        return hidden_states, mlp_residual, attn_residual, attn_heads #, vit_attn_heads
