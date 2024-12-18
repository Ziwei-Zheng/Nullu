import argparse
import sys
###################################################
####### Set the path to the repository here #######
class Args(argparse.Namespace):
    cfg_path = './minigpt4/minigpt4_llama2_eval.yaml'
    options = []
###################################################

import torch
from torch import nn
from PIL import Image
import cv2
from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from model.base import LargeMultimodalModel, create_hook
    
    
class MiniGPT4(LargeMultimodalModel):
    def __init__(self, args):
        super(MiniGPT4, self).__init__()
        
        minigpt_args = Args()
        cfg = Config(minigpt_args)
        cfg.model_cfg.llama_model = args.model_path

        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to('cuda')
        print(f"Load llama-2-7b-chat-hf from: {cfg.model_cfg.llama_model}")

        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                     'pretrain_llama2': CONV_VISION_LLama2}

        self.CONV_VISION = conv_dict[model_config.model_type]
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.tokenizer = self.model.llama_tokenizer
        self.num_lm_layers = self.model.llama_model.config.num_hidden_layers
        self.num_lm_hidden_size = self.model.llama_model.config.hidden_size
        self.lm_head = self.model.llama_model.lm_head

        self.args = args
        
        print('Initialization Finished')
    
    def refresh_chat(self):
        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(self.device) for ids in stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        self.Chat = Chat(self.model, self.vis_processor, 
                    device=self.device, 
                    stopping_criteria=stopping_criteria)
        self.Chat_state = self.CONV_VISION.copy()
    
    @torch.no_grad()
    def chat(self, image_path, prompt):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        self.refresh_chat()
        
        img_list = []
        llm_message = self.Chat.upload_img(image, self.Chat_state, img_list)
        self.Chat.encode_img(img_list)
        
        self.Chat.ask(prompt, self.Chat_state)
        llm_message = self.Chat.answer(conv=self.Chat_state,
                                       img_list=img_list,
                                       do_sample=True if self.args.temperature>0 else False,
                                       temperature=self.args.temperature,
                                       top_p = self.args.top_p,
                                    #    top_k = self.args.top_k,
                                       num_beams=self.args.num_beams,
                                       max_new_tokens=self.args.max_length,
                                       max_length=2000)[0]
        # print(self.Chat_state)
        return llm_message
    
    @torch.no_grad()
    def _basic_forward(self, image_path, prompt, answer=None, return_dict=False):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        self.refresh_chat()

        img_list = []
        llm_message = self.Chat.upload_img(image, self.Chat_state, img_list)
        self.Chat.encode_img(img_list)
        
        self.Chat.ask(prompt, self.Chat_state)
        self.Chat_state.append_message(self.Chat_state.roles[1], answer)
        prompt = self.Chat_state.get_prompt()
        embs = self.model.get_context_emb(prompt, img_list)

        # llama forward
        with self.model.maybe_autocast():
            outputs = self.model.llama_model(
                inputs_embeds=embs,
                return_dict=True,
                output_attentions=True,
                output_hidden_states=True,
            )

        return outputs
    
    def register_hooks(self):
        self.model.attn_heads, self.model.attn_residual, self.model.mlp_residual, self.model.vit_satt = [], [], [], []
        attn_head_hook = create_hook(self.model.attn_heads, loc='input')
        attn_residual_hook = create_hook(self.model.attn_residual)
        mlp_residual_hook = create_hook(self.model.mlp_residual)
        # vit_forward_hook = create_hook(self.model.vit_satt, loc='input')
        self.hooks = []
        for layer in self.model.llama_model.model.layers:
            self.hooks.append(layer.self_attn.o_proj.register_forward_hook(attn_head_hook))
            self.hooks.append(layer.self_attn.register_forward_hook(attn_residual_hook))
            self.hooks.append(layer.mlp.register_forward_hook(mlp_residual_hook))
        # for layer in self.model.visual_encoder.blocks:
        #     self.hooks.append(layer.attn.proj.register_forward_hook(vit_forward_hook))
    
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
        # vit_attn_heads = torch.cat(self.model.vit_satt).reshape(39, -1, 16, 88)[:, 1:]   # remove cls_token
        self.remove_hooks()
        return hidden_states, mlp_residual, attn_residual, attn_heads #, vit_attn_heads