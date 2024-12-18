import sys
import os
import json
###################################################
####### Set the path to the repository here #######
sys.path.append("../llava/")
###################################################
import cv2

import torch
from torch import nn
import numpy as np
from io import BytesIO
from transformers import TextStreamer
from transformers.generation import BeamSearchDecoderOnlyOutput

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from model.base import LargeMultimodalModel


def add_diffusion_noise(image_tensor, noise_step): # from VCD
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd


def create_hook(feat_list, loc='output'):
    if loc == 'output':
        def hook(module, input, output):
            feat_list.append(output[0])
    else:
        def hook(module, input, output):
            feat_list.append(input[0])
    return hook


class LLaVA(LargeMultimodalModel):
    def __init__(self, args):
        super(LLaVA, self).__init__()
        load_8bit = False
        load_4bit = False
        
        # Load Model
        disable_torch_init()

        model_name = get_model_name_from_path(args.model_path)
        if "finetune-lora" in args.model_path:
            model_base = "liuhaotian/llava-v1.5-7b"
        elif "lora" in args.model_path:
            model_base = "lmsys/vicuna-7b-v1.5"
        else:
            model_base = None
        self.args = args
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(args.model_path, model_base, model_name, load_8bit, load_4bit)
        # print(model_name)
        self.conv_mode = "llava_v1"

        # self.num_img_tokens = (self.model.config.vision_config.image_size // self.model.config.vision_config.patch_size)**2
        self.num_lm_layers = self.model.config.num_hidden_layers
        self.num_lm_hidden_size = self.model.config.hidden_size
        self.lm_head = self.model.lm_head
    
    def refresh_chat(self):
        self.conv = conv_templates[self.conv_mode].copy()
        self.roles = self.conv.roles
    
    @torch.no_grad()
    def _basic_forward(self, noise_flag, image_path, prompt, answer=None, return_dict=False):
        self.refresh_chat()
        image = cv2.imread(image_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.unsqueeze(0).half().to(self.device)
        if noise_flag:
            image_tensor = add_diffusion_noise(image_tensor, 500)

        # image_tensor = None ############
        # inp = prompt ###############

        # message
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], answer)

        conv_prompt = self.conv.get_prompt()
        input_ids = tokenizer_image_token(conv_prompt, self.tokenizer, 
                                          IMAGE_TOKEN_INDEX, 
                                          return_tensors='pt').unsqueeze(0).cuda()

        outputs = self.model(
            input_ids,
            images=image_tensor, 
            return_dict=return_dict,
            output_attentions=return_dict,
            output_hidden_states=return_dict)
        
        # if return_dict:
        #     outputs = outputs[0]

        return outputs
    
    @torch.no_grad()
    def chat(self, image_path, prompt, answer=None, return_dict=False):
        self.refresh_chat()
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.unsqueeze(0).half().to(self.device)

        # message
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)

        conv_prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(conv_prompt, self.tokenizer, 
                                          IMAGE_TOKEN_INDEX, 
                                          return_tensors='pt').unsqueeze(0).cuda()
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = ["###"]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # with torch.inference_mode():
        outputs = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if self.args.temperature > 0 else False,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            num_beams=self.args.num_beams,
            max_new_tokens=self.args.max_length,
            # streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            return_dict_in_generate=return_dict,
            output_attentions=return_dict,
            output_hidden_states=return_dict,
            output_scores=return_dict,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def register_hooks(self):
        self.model.attn_heads, self.model.attn_residual, self.model.mlp_residual, self.model.vit_satt = [], [], [], []
        attn_head_hook = create_hook(self.model.attn_heads, loc='input')
        attn_residual_hook = create_hook(self.model.attn_residual)
        mlp_residual_hook = create_hook(self.model.mlp_residual)
        # vit_forward_hook = create_hook(self.model.vit_satt, loc='input')
        self.hooks = []
        for layer in self.model.base_model.layers:
            self.hooks.append(layer.self_attn.o_proj.register_forward_hook(attn_head_hook))
            self.hooks.append(layer.self_attn.register_forward_hook(attn_residual_hook))
            self.hooks.append(layer.mlp.register_forward_hook(mlp_residual_hook))
        # for layer in self.model.base_model.vision_tower.vision_tower.vision_model.encoder.layers:
        #     self.hooks.append(layer.self_attn.out_proj.register_forward_hook(vit_forward_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_activations(self, image_path, prompt, answer=None, noise_flag=False):##########
        self.register_hooks()
        outputs = self._basic_forward(noise_flag, image_path, prompt, answer, return_dict=True)###########
        attn_heads = torch.cat(self.model.attn_heads).reshape(32, -1, 32, 128)   # [32, seq_len, 4096] -> [32, seq_len, 32, 128]
        attn_residual = torch.cat(self.model.attn_residual)   # [32, seq_len, 4096]
        mlp_residual = torch.stack(self.model.mlp_residual)   # [32, seq_len, 4096]
        hidden_states = torch.stack(outputs.hidden_states)[1:, 0]   # [32, seq_len, 4096]
        # vit_attn_heads = torch.cat(self.model.vit_satt).reshape(24, -1, 16, 64)   # [24, img_len, 1024] -> [24, seq_len, 16, 64]
        self.remove_hooks()
        return hidden_states, mlp_residual, attn_residual, attn_heads #, vit_attn_heads
