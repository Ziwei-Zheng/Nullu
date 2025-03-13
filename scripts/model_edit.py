import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import argparse
import cv2
import json
import numpy as np
import random
import pickle
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from model import build_model
from utils.halluedit import HalluEdit


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# pos: halluciation, neg: non_halluciation
def load_embedding_data(pkl_path, loc):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")
    
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    pos_data, neg_data = [], []
    for entry in data:
        if entry['label'] == 0:
            pos_data.append(entry[loc])
        else:
            neg_data.append(entry[loc])

    if not pos_data:
        raise ValueError("No positive data found.")
    if not neg_data:
        raise ValueError("No negative data found.")

    pos_data = torch.stack(pos_data).float()
    neg_data = torch.stack(neg_data).float()

    if pos_data.size(0) != neg_data.size(0):
        raise ValueError("Positive and negative data sizes do not match.")

    return pos_data, neg_data


def save_model_and_config(tokenizer, edited_model, save_path, model_name, config_paths):

    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    
    if model_name == 'MiniGPT4':
        edited_model.llama_model.save_pretrained(save_path)
    else:
        edited_model.save_pretrained(save_path)

    for config_name, config_path in config_paths.items():
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(save_path, config_name))
            print(f"Copied {config_path} to {save_path}")

    print(f'Saved edited model to {save_path}')


def main(args):

    setup_seeds()

    model = build_model(args)
    
    if args.emb_path is not None:
        loc = {
            'mean': 'hidden_states_mean',
            'last': 'hidden_states',
            'mlp_residual': 'mlp_residual'
        }.get(args.ebd)

        pos_data, neg_data = load_embedding_data(args.emb_path, loc=loc)
        print(f'Loading offline embeddings from {args.emb_path}')

    output_dir = os.path.join("./output", args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply editing
    if args.lowest_layer == -1 or args.highest_layer == -1:
        layer_range = None
    else:
        layer_range = np.arange(args.lowest_layer, args.highest_layer)
    
    editor = HalluEdit(model=model, ebd=args.ebd, centering=args.centering, alpha=args.alpha,
                       top_k_ranks=args.top_k_ranks, edit_layer_range=layer_range, random_dps=True)

    edited_model = editor.apply_edit_end_to_end(pos_data, neg_data, 
                                                edit_keys=args.edit_keys, edit_values=args.edit_values, layer_range=layer_range)
    
    # Save edited model
    save_dir = args.save_model_dir
    os.makedirs(save_dir, exist_ok=True)

    save_tag = f"-{args.save}" if args.save is not None else ""

    save_name = f"{args.model_name}-top{args.top_k_ranks}-{args.lowest_layer}-{args.highest_layer}{save_tag}"
    save_path = os.path.join(args.save_model_dir, save_name)
    
    config_paths = {
        'preprocessor_config.json': os.path.join(args.model_path, 'preprocessor_config.json'),
        'configuration.json': os.path.join(args.model_path, 'configuration.json')
    }

    save_model_and_config(editor.tokenizer, edited_model, save_path, args.model_name, config_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')
    # parser.add_argument("--model_name", choices=['LLaVA-7B', 'MiniGPT4', 'mPLUG_Owl2'], default="mPLUG_Owl2") 
    # parser.add_argument("--model_path", default="/workspace/data1/huggingface/hub/models--MAGAer13--mplug-owl2-llama2-7b/snapshots/200342bbdd0eef019b02b4d7c9b17df235bba4ad")
    parser.add_argument("--model_name", choices=['LLaVA-7B', 'MiniGPT4', 'mPLUG_Owl2'], default="MiniGPT4") 
    parser.add_argument("--model_path", default="/workspace/data1/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590")
    parser.add_argument(
        "--emb_path", type=str, 
        default="./output/MiniGPT4/lure_train_first20_1_42_activations.pkl"
    ) 

    parser.add_argument("--centering", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--ebd", choices=['mean', 'last', 'mlp_residual'], default='mean')
    parser.add_argument("--edit_keys", action="store_true", default=False)
    parser.add_argument("--edit_values", action="store_true", default=True)

    parser.add_argument("--top_k_ranks", type=int, default=8) #
    parser.add_argument("--lowest_layer", type=int, default=16) # 31-32,16-32,16-24,24-32
    parser.add_argument("--highest_layer", type=int, default=32) #

    parser.add_argument("--save_model_dir", type=str, default="./output/edited_model")
    parser.add_argument("--save", type=str, default="test")
    
    main(parser.parse_args())
