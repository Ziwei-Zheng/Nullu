import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print(sys.path)
import argparse
import random
import pickle
import time

import cv2
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from model import build_model
from dataset import build_dataset


def get_model_activtion(args, data, model, saved_file):

    all_results = []
    
    for sample in tqdm(data, desc="Processing data samples"):
        image_path = sample["image_path"]
        image_id = os.path.basename(image_path)
        prompt = sample["question"]
        answer = sample["answer"]
        label = sample["label"]

        hidden_states, mlp_residual, attn_residual, attn_heads = model.get_activations(image_path, prompt, answer)

        out = {
            "image": image_id,
            "model_name": args.model_name,
            "question": prompt,
            "answer": answer,
            "label": label,
            "attn_residual": attn_residual[:, -1].cpu(),
            "hidden_states": hidden_states[:, -1].cpu(),
            "mlp_residual": mlp_residual[:, -1].cpu(),
            "attn_heads": attn_heads[:, -1].cpu(),
            "hidden_states_mean": hidden_states.mean(1).cpu(),
        }
        all_results.append(out)
    
    with open(saved_file, 'wb') as file:
        pickle.dump(all_results, file)

    print(f'Saved activations to {saved_file}')


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def main(args):

    setup_seeds(args.seed)

    model = build_model(args)

    pos_data, neg_data = build_dataset(args.dataset, args.split, args.sampling, args.num_samples)
    data = pos_data + neg_data
        
    output_dir = f"./output/{args.model_name}/"
    os.makedirs(output_dir, exist_ok=True)

    sampling_tag = f"_{args.sampling}{args.num_samples}" if args.num_samples else ""
    save_tag = f"_{args.save}" if args.save else ""
    saved_file = os.path.join(
        output_dir,
        f"{args.dataset}_{args.split}{sampling_tag}{save_tag}_{args.seed}_activations.pkl"
    )
    get_model_activtion(args, data, model, saved_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')
    parser.add_argument("--model_name", choices=['LLaVA-7B', 'MiniGPT4', 'mPLUG_Owl2'], default="MiniGPT4") 
    parser.add_argument("--model_path", default="/workspace/data1/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590")
    parser.add_argument("--dataset", default="lure")
    parser.add_argument("--split", default="train")

    parser.add_argument("--num_samples", type=int, default=None) 
    parser.add_argument("--sampling", choices=['first', 'random'], default='first') 

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="")   
    main(parser.parse_args())