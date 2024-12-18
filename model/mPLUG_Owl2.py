import sys

import torch
from torch import nn
from PIL import Image
import cv2
from transformers import TextStreamer

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from model.base import LargeMultimodalModel, create_hook


class mPLUG_Owl2(LargeMultimodalModel):
    def __init__(self, args):
        super(mPLUG_Owl2, self).__init__()
        model_path = args.model_path   # 'MAGAer13/mplug-owl2-llama2-7b'
        model_name = get_model_name_from_path(model_path)
        
        self.args = args
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")

    @torch.no_grad()
    def chat(self, image_path, prompt, return_dict=False):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))
        
        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        conv = conv_templates["mplug_owl2"].copy()
        roles = conv.roles

        inp = DEFAULT_IMAGE_TOKEN + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        outputs = self.model.generate(
            input_ids,
            images=image_tensor,
            # streamer=streamer,
            use_cache=True,
            max_new_tokens = self.args.max_length,
            stopping_criteria=[stopping_criteria],
            
            do_sample=True if self.args.temperature > 0 else False,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            num_beams=self.args.num_beams,

            return_dict_in_generate=return_dict,
            output_attentions=return_dict,
            output_hidden_states=return_dict,
            output_scores=return_dict
        )
        outputs = self.tokenizer.decode(outputs[0, input_ids.shape[1]:]).strip() ####
        return outputs
    
    @torch.no_grad()
    def _basic_forward(self, image_path, prompt, answer=None, return_dict=False):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))
        
        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        conv = conv_templates["mplug_owl2"].copy()

        inp = DEFAULT_IMAGE_TOKEN + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], answer)

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

        outputs = self.model(
            input_ids,
            images=image_tensor,
            return_dict=return_dict,
            output_attentions=return_dict,
            output_hidden_states=return_dict)
        return outputs
    
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
        # for layer in self.model.base_model.vision_model.encoder.layers:
        #     self.hooks.append(layer.self_attn.dense.register_forward_hook(vit_forward_hook)) ############
    
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
        # vit_attn_heads = torch.cat(self.model.vit_satt).reshape(24, -1, 16, 64)   # [24, img_len, 1024] -> [24, seq_len, 16, 64]
        self.remove_hooks()
        return hidden_states, mlp_residual, attn_residual, attn_heads #, vit_attn_heads
