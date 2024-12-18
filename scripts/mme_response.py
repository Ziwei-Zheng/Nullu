import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from model import build_model
from PIL import Image


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def process_files(args, model):

    save_tag = f"_{args.save}" if args.save else ""
    model_name = os.path.basename(args.model_path) 
    output_dir = os.path.join("./eval/mme", f"{model_name}-seed{args.seed}{save_tag}")
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(args.reference_dir):
        print(f"\nProcessing file: {file_name}")
        if file_name.endswith(".txt"):
            task_name = file_name.split(".")[0]
            task_dir = os.path.join(args.base_dir, task_name)
            if not os.path.exists(task_dir):
                print(f"Task data directory {task_dir} does not exist, skipping this task.")
                continue

            input_file_path = os.path.join(args.reference_dir, file_name)
            output_file_path = os.path.join(output_dir, f"{task_name}.txt")

            with open(input_file_path, "r") as input_file:
                lines = input_file.readlines()

            # 10%
            # grouped_lines = [lines[i:i + 2] for i in range(0, len(lines), 2)]
            # sampled_groups = random.sample(grouped_lines, len(grouped_lines)//10)
            # input_file = [line for group in sampled_groups for line in group]

            # all
            input_file = lines

            with open(output_file_path, "w") as output_file:
                for i, line in tqdm(enumerate(input_file), mininterval=0.1, desc="Processing"):
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        print(f"Incorrect format in file {file_name} at line {i}, skipping this line.")
                        continue
                    
                    image_name, question, ground_truth_answer = parts
                    
                    if os.path.exists(os.path.join(task_dir, "images", image_name)):
                        image_path = os.path.join(task_dir, "images", image_name)
                    else:
                        image_path = os.path.join(task_dir, image_name)
                    
                    if not os.path.exists(image_path):
                        print(f"Image path {image_path} does not exist, skipping this line.")
                        continue

                    try:
                        from PIL import Image
                        with Image.open(image_path) as img:
                            img = img.convert("RGB")
                    except OSError as e:
                        print(f"Failed to load image at {image_path}, error: {e}, skipping this line.")
                        continue

                    response = model.chat(image_path, question)

                    response = response.replace("\n", " ").strip()
                    output_file.write(f"{image_name}\t{question}\t{ground_truth_answer}\t{response}\n")


def main(args):

    setup_seeds(args.seed)
    model = build_model(args)
    process_files(args, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')
    parser.add_argument("--reference_dir", default="/data/MME_Benchmark_release_version/eval_tool/Your_Results")
    parser.add_argument("--base_dir", default="/workspace/MME")
    parser.add_argument("--model_name", default="LLaVA-7B") 
    parser.add_argument("--model_path", default="/workspace/llava-v1.5-7b")

    # parser.add_argument("--model_name", default="Qwen_VL_Chat") 
    # parser.add_argument("--model_path", default="/workspace/data1/huggingface/hub/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8")
    parser.add_argument("--seed", type=int, default="42")
    parser.add_argument("--save", default="")

    parser.add_argument("--temperature", type=float, default=0) # 0 1 1
    parser.add_argument("--top_p", type=float, default=None) # 0 0.9 1
    parser.add_argument("--top_k", type=float, default=None) # 0 0.9 1
    parser.add_argument("--num_beams", type=int, default=1) # 1 3 1
    parser.add_argument("--token_id", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=64) # 64

    main(parser.parse_args())