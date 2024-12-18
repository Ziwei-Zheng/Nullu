import os
import sys
import inspect

import json
import torch
import logging
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import cv2

logging.getLogger().setLevel(logging.INFO)

class HalluEdit():
    def __init__(self, model, ebd='mean', centering=False, top_k_ranks=2, edit_layer_range=None, random_dps=True, alpha=1):

        self.model = model
        self.model.model.eval()
        self.tokenizer = model.tokenizer

        self.alpha = alpha

        model_config = getattr(model, 'model', None) and getattr(model.model, 'config', None)

        if model_config: # model.model.config.model_type
            model_type = getattr(model_config, 'model_type', None)
            self.D = model.model.config.hidden_size
            self.num_layers = model.model.config.num_hidden_layers
            self.E = model.model.lm_head
            self.lm_sep_idx = 2
            # print(f'self.model_name is {model_type}')

        else: # model.args.model_name
            self.D = model.num_lm_hidden_size
            self.num_layers = model.num_lm_layers
            self.E = model.lm_head
            if model.args.model_name == ('MiniGPT4' or 'LLaVA-7B-HF'):
                self.lm_sep_idx = 3
            else:
                self.lm_sep_idx = 2
            
        print(f'args.model_name is {model.args.model_name}')

        self.ebd = ebd
        self.random_dps = random_dps
        self.centering = centering
        self.top_k_ranks = top_k_ranks
        if edit_layer_range is None:
            self.edit_layer_range = np.arange(self.num_layers)
        else:
            self.edit_layer_range = edit_layer_range

        self.f = open(f'logit_lens_test_{model.args.model_name}.txt', 'w')


    @staticmethod
    def project_into_vocabluary(vector, E, tokenizer, top_k=20, bottom_k=-1):
        """
        Project a vector into the vocabulary space and return the top_k tokens.
        :param vector: D dimensional vector
        :param E: Language model embedding matrix (V, D)
        :param tokenizer: Model tokenizer
        :param top_k: How many top tokens to return
        :param bottom_k: How many bottom tokens to return. If -1, return top_k tokens
        :return:
        """
        vector = vector.to(torch.float32).to('cuda')
        E = E.to(torch.float32).to('cuda')
        # vocab_ranking = torch.matmul(E, vector)     # (V,)
        vocab_ranking = E(vector)     # (V,)
        sorted_token_ids = np.argsort(vocab_ranking.detach().cpu().numpy())[::-1]  # Descending order
        if bottom_k == -1:
            sorted_tokens = [tokenizer.decode(x).strip() for x in sorted_token_ids[:top_k]]
            logging.debug([(sorted_token_ids[i], sorted_tokens[i], vocab_ranking[sorted_token_ids[i]].item()) for i in range(top_k)])
        else :
            sorted_tokens = [tokenizer.decode(x).strip() for x in sorted_token_ids[-bottom_k:][::-1]]  # Least score to most score
        return sorted_tokens


    def _get_hidden_sentence_embeddings(self, data):

        hidden_sent_embs = []
        for ins in tqdm(data):
            image = cv2.imread(ins['img_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(ins['img_path'])
            prompt = ins['question']
            answer = ins['answer']

            if self.ebd == 'mean':
                outputs, _ = self.model._basic_forward(image, prompt, answer, return_dict=True)
                hidden_states = torch.stack(outputs.hidden_states)[1:, 0]   # [32, seq_len, 4096]
                hidden_sent_embs.append(hidden_states.mean(1).cpu())   # sentence mean, [32, 4096]
            elif self.ebd == 'last':
                outputs, _ = self.model._basic_forward(image, prompt, answer, return_dict=True)
                hidden_states = torch.stack(outputs.hidden_states)[1:, 0]   # [32, seq_len, 4096]
                hidden_sent_embs.append(hidden_states[:, -1].cpu())   # last token, [32, 4096]
            elif self.ebd == 'mlp_residual':
                _, mlp_residual, _, _, _, _ = self.model.get_activations(image, prompt, answer)   # [32, seq_len, 4096]
                hidden_sent_embs.append(mlp_residual[:, -1].cpu())   # last token, [32, 4096]
            else:
                raise NotImplementedError
        
        hidden_sent_embs = torch.stack(hidden_sent_embs).permute(1, 0, 2)   # [32, N, 4096]
        return hidden_sent_embs


    def _get_difference_matrix(self, pos_data, neg_data):
        non_preferred_sent_embs = self._get_hidden_sentence_embeddings(pos_data) if isinstance(pos_data, list) else pos_data.permute(1, 0, 2)  # (L, N, D)
        preferred_sent_embs = self._get_hidden_sentence_embeddings(neg_data) if isinstance(neg_data, list) else neg_data.permute(1, 0, 2)  # (L, N, D)

        difference_matrix = (preferred_sent_embs - non_preferred_sent_embs) / 2  # (L, N, D)

        logging.info('Difference matrix calculated.')
        del non_preferred_sent_embs

        if self.centering:
            logging.info('Centering: Removing first singular vector from preference matrix.')

            for layer_num in range(difference_matrix.shape[0]):
                d = difference_matrix[layer_num].to(torch.float32)
                pref = deepcopy(preferred_sent_embs[layer_num].to(torch.float32))

                u, s, vt = torch.linalg.svd(pref, full_matrices=False)  # (N, D) -> (N, N), (N,), (N, D)
                projection_vector = vt[0].unsqueeze(dim=-1)  # (D, 1)

                sorted_tokens = self.project_into_vocabluary(projection_vector.squeeze(), self.E.cpu(), self.tokenizer, top_k=10)
                self.f.write(f'Layer {layer_num} - mu: {" | ".join([x for x in sorted_tokens])}\n')

                P = projection_vector @ projection_vector.T  # (D, D)
                I = torch.eye(projection_vector.shape[0]).to(pref.device)  # (D, D)

                d = d @ (I - self.alpha * P)  # (N, D) @ (D, D) -> (N, D) alpha
                difference_matrix[layer_num] = d.to(difference_matrix[layer_num].dtype) # d

        return difference_matrix


    def get_ats(self, pos_data, neg_data):

        difference_matrix = self._get_difference_matrix(pos_data, neg_data)  # (L, N, D)
        ats = {}

        for key in self.model.model.state_dict():
            if self.model.args.model_name == 'MiniGPT4':
               # 'llama_model.model.layers.2.mlp.down_proj.weight_format'
                if (
                    'weight' in key 
                    and 'mlp' in key 
                    and '_format' not in key 
                    and not 'visual_encoder' in key 
                    and not 'gate_proj' in key 
                    and not 'up_proj' in key
                ):
                    layer_num = int(key.split('.')[self.lm_sep_idx])  
                    ats[key] = difference_matrix[layer_num]
            elif self.model.args.model_name == 'Qwen_VL_Chat':
                if (
                    'mlp.c_proj.weight' in key 
                    and not 'visual' in key 
                ):
                    layer_num = int(key.split('.')[self.lm_sep_idx])  
                    ats[key] = difference_matrix[layer_num]
            elif self.model.args.model_name == 'mPLUG_Owl2':
                if (
                    'weight' in key 
                    and 'mlp' in key 
                    and not 'vision' in key 
                    and not 'gate_proj' in key 
                    and not 'up_proj' in key
                    and not 'visual' in key # owl2
                ):
                    layer_num = int(key.split('.')[self.lm_sep_idx])  
                    ats[key] = difference_matrix[layer_num]
            else:
                if (
                    'weight' in key 
                    and 'mlp' in key 
                    and not 'vision_tower' in key 
                    and not 'gate_proj' in key 
                    and not 'up_proj' in key
                ):
                    layer_num = int(key.split('.')[self.lm_sep_idx])  # Format: 'language_model.model.layers.0.mlp.gate_proj.weight'
                    ats[key] = difference_matrix[layer_num]
        return ats
    
    
    def svd_on_ats(self, ats):
        '''
        Key(D, 4D) -> U(D, D) S(D) V^T(D, 4D)
        Value(4D, D) -> U(4D, D) S(4D) V^T(D, D)
        x_l (N, D) -> U(N, N); S(N,); V^T(N, D)

        Note: v @ v.T is not numerically I, but plotting it as a heatmap shows that it is close to I.
        '''
        svd = {}
        for key in ats:
            logging.debug(f'Calculating SVD for: {key}')
            M = ats[key].to(torch.float32)  # SVD function only works with float32
            u, s, vt = torch.linalg.svd(M.cuda(), full_matrices=False)  # Skinny SVD, vt is V^T
            svd[key] = {'u': u.cpu(), 's': s.cpu(), 'v': vt.T.cpu()}
        logging.info('SVD of ATS calculated.')
        return svd


    def find_p_hallu(self, svd, rank_range=20):
        hallu_subspace = {}

        # singular_list = []
        for key in svd.keys():
            layer_num = int(key.split('.')[self.lm_sep_idx])  # Format: 'language_model.model.layers.0.mlp.up_proj.weight'
            if layer_num not in self.edit_layer_range:
                logging.info(f'Skipping layer {layer_num}')
                continue
            self.f.write(f'Calculating hallu subspace for: {key}\n')
            logging.info(f'Calculating hallu subspace for: {key}')

            singular_vectors = svd[key]['v']  # (D, N): N cols of (D,) vectors
            # singular_list.append(singular_vectors) 
            hallu_rank_list = np.arange(self.top_k_ranks)  # [0, 1] by default

            # Sum outer products of shortlisted ranks
            p_hallu = torch.zeros(self.D, self.D)
            for r in hallu_rank_list:
                singular_vector = singular_vectors[:, r].unsqueeze(dim=1)  # (D, 1)
                p_hallu += singular_vector @ singular_vector.T  # (D, 1) @ (1, D) -> (D, D)

                sorted_tokens = self.project_into_vocabluary(singular_vector.squeeze(), self.E.cpu(), self.tokenizer, top_k=10)
                self.f.write(f'Layer {layer_num} - rank{r}: {" | ".join([x for x in sorted_tokens])}\n')

            hallu_subspace[key] = p_hallu
        # singular_tensor = torch.stack([sv.clone().detach() for sv in singular_list]) 
        # torch.save(singular_tensor, 'singular_lure_layers16-32.pkl') 
        logging.info('Hallu subspace calculated.')
        return hallu_subspace


    def edit_model(self, hallu_subspace, edit_keys=True, edit_values=True, layer_range=None):
        assert edit_keys or edit_values, 'At least one of edit_keys or edit_values should be True'
        logging.info(f'Editing keys: {edit_keys}, Editing values: {edit_values}.')

        if layer_range is None:
            layer_range = np.arange(self.num_layers)
        logging.info(f'Editing layers: {layer_range}')

        edited_state_dict = self.model.model.state_dict()
        for key in edited_state_dict:
            if key in hallu_subspace:
                layer_num = int(key.split('.')[self.lm_sep_idx])
                if layer_num in layer_range:
                    logging.info(f'Editing: {key}')
                    logging.info(f'Module {key}: P_hallu mean: {hallu_subspace[key].mean()}.')

                    P_filter = torch.eye(self.D) - hallu_subspace[key]
                    if self.model.args.model_name == 'MiniGPT4':
                        P_filter = P_filter.to(edited_state_dict[key].device).to(self.model.model.llama_model.dtype)
                    else:
                        P_filter = P_filter.to(edited_state_dict[key].device).to(self.model.model.dtype)

                    weight = edited_state_dict[key]
                    weight = weight.T

                    if edit_keys and 'up_proj' in key:
                        modified_weight = P_filter @ weight  # (D, D) @ (D, 4D) -> (D, 4D)
                    elif edit_values and 'down_proj' in key:
                        modified_weight = weight @ P_filter  # (4D, D) @ (D, D) -> (4D, D)
                    elif 'c_proj' in key: # Qwen_VL_Chat
                        print('c_proj')
                        modified_weight = weight @ P_filter
                    else:
                        print('no modified_weight')
                        continue
                    if torch.allclose(weight, modified_weight) and ('gate_proj' not in key):
                        logging.warning(f'Module {key} not edited after projection.')
                        print(f'Module {key} not edited after projection.')

                    # if self.model_category in ['llama', 'mistral', 'opt', 'gptj']:
                    modified_weight = modified_weight.T

                    edited_state_dict[key] = modified_weight.to('cuda').contiguous()  # contiguous for saving to disk

        self.model.model.load_state_dict(edited_state_dict, assign=True)
        logging.info('Edited model created.')
        return self.model.model


    def setup_for_edits(self, pos_data, neg_data):
        ats = self.get_ats(pos_data, neg_data)
        svd = self.svd_on_ats(ats)
        del ats
        self.hallu_subspace = self.find_p_hallu(svd)
        del svd
        torch.cuda.empty_cache()


    def apply_edit_end_to_end(self, pos_data, neg_data, edit_keys=True, edit_values=True, layer_range=None):
        # Measure speed and memory use
        # import time
        # import psutil
        # import pynvml
        # start_time = time.time()
        # before_memory = psutil.virtual_memory().used
        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # before_gpu_memory_used = info.used

        # Find P_hallu
        self.setup_for_edits(pos_data, neg_data)

        # Apply edit
        edited_model = self.edit_model(self.hallu_subspace, edit_keys, edit_values, layer_range)
        torch.cuda.empty_cache()

        # end_time = time.time()
        # time.sleep(1)
        # after_memory = psutil.virtual_memory().used
        # info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # after_gpu_memory_used = info.used
        # print(f"Elapsed time: {end_time - start_time} seconds")
        # print(f"System Memory Used: {(after_memory - before_memory) / (1024 * 1024)} MB")
        # print(f"GPU Memory Used: {(after_gpu_memory_used - before_gpu_memory_used) / (1024 ** 2)} MB")

        return edited_model
    
