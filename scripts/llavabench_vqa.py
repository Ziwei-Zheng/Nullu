import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import random
import pickle
import math

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.backends.cudnn as cudnn

from model import build_model

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def eval_model(args):

    setup_seeds()

    model = build_model(args)

    answers_file = "./eval/llavabench_answer/" + args.model_path.split('/')[-1] + args.save + ".jsonl"

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        ans_id = shortuuid.uuid()

        image_path = os.path.join(args.image_folder, image_file)

        outputs = model.chat(image_path, qs)
        ans_file.write(json.dumps(
            {
                "image_id": image_file,
                "question_id": idx,
                "prompt": cur_prompt,
                "response": outputs,
                "answer_id": ans_id,
                "model_id": args.model_name,
                "metadata": {}
            }
        ) + "\n")
        ans_file.flush()
    ans_file.close()
    print(f"Save answer file to {answers_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="")

    parser.add_argument("--model_name", choices=['LLaVA-7B', 'MiniGPT4', 'mPLUG_Owl2'], default="LLaVA-7B") 
    parser.add_argument("--model_path", default="/workspace/llava-v1.5-7b")
  
    parser.add_argument("--image-folder", type=str, default="") # liuhaotian/llava-bench-in-the-wild/images
    parser.add_argument("--question_file", type=str, default="") # liuhaotian/llava-bench-in-the-wild/questions.jsonl

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--max-length", type=int, default=1024)

    args = parser.parse_args()

    eval_model(args)
    
    

    