import argparse
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import base64
from openai import OpenAI
import time
from tqdm import tqdm
import logging

# Set up logging configuration
logging.basicConfig(filename="eval/llavabench/log.txt", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Define a LogFile class to redirect print statements to a log file
class LogFile:
    def __init__(self, file_name):
        self.file = open(file_name, 'a')
    
    def write(self, message):
        self.file.write(message)
        self.file.flush()
    
    def close(self):
        self.file.close()

# Define the prompt template for evaluation (Assistant 1 vs Assistant 2)
vcd_prompt = (
    "[Assistant 1] \n"
    "{Response1} \n"
    "[End of Assistant 1] \n\n"
    "[Assistant 2] \n"
    "{Response2} \n"
    "[End of Assistant 2]"
)

# Function to encode the image as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to get the evaluation result from OpenAI API with image and prompt
def get_res_with_image(image_path, prompt, api_key, base_url):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    base64_image = encode_image(image_path)
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    completion = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": """
                You are an AI designed to evaluate and score the performance of two AI assistants in describing a given image. 
                Your primary focus is on the accuracy and detailedness of their descriptions. 
                You will assess the accuracy by checking for hallucinations - any part of the description that is inconsistent with the image content. 
                For detailedness, you will consider how rich the response is in necessary details, excluding any hallucinated parts.
                You will provide scores on a scale from 1 to 10 for each assistant separately, based on these criteria.
                After scoring, you will offer an explanation for your evaluation, ensuring it is free from bias and not influenced by the order of presentation of the responses.
                
                Input format:
                [Assistant 1] {Response 1} [End of Assistant 1]
                [Assistant 2] {Response 2} [End of Assistant 2]
                
                Output format:
                Accuracy: 
                Scores of the two answers: 
                Reason:

                Detailedness: 
                Scores of the two answers: 
                Reason:
                """},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"                    
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]},
        ]
    )
    return completion.choices[0].message.content

# Function to read JSONL file and parse each line
def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def parse_args():
    parser = argparse.ArgumentParser()
    
    # File path arguments
    parser.add_argument('--raw_answer_path', type=str, default='eval/llavabench/MiniGPT4.jsonl', help="Path to the raw answer JSONL file.")
    parser.add_argument('--edit_answer_path', type=str, default='eval/llavabench/MiniGPT4-lure--top16-mean-False.jsonl', help="Path to the edited answer JSONL file.")
    parser.add_argument('--images_dir', type=str, default='/workspace/data1/llavabench/images', help="Directory containing image files.")
    
    # OpenAI-related parameters
    parser.add_argument('--api_key', type=str, default="", help="OpenAI API Key.")
    parser.add_argument('--base_url', type=str, default="", help="Base URL for OpenAI API.")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Extracting arguments
    raw_answer_path = args.raw_answer_path
    edit_answer_path = args.edit_answer_path
    images_dir = args.images_dir
    log_file = os.path.join(f"eval/llavabench/results", os.path.basename(edit_answer_path).replace(".jsonl",".log"))
    api_key = args.api_key
    base_url = args.base_url
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Read raw and edited answers from the respective JSONL files
    raws = read_jsonl(raw_answer_path)
    edits = read_jsonl(edit_answer_path)

    # Open log file to capture print statements
    logfile = LogFile(log_file)
    sys.stdout = logfile

    # Process each pair of raw and edited answers
    for raw, edit in tqdm(zip(raws, edits), total=len(raws)):
        image_path = os.path.join(images_dir, raw['image_id'])
        prompt = vcd_prompt.format(Response1=raw['response'], Response2=edit['response'])
        
        # Get the result from the OpenAI API
        response = get_res_with_image(image_path, prompt, api_key, base_url)

        print(prompt)
        print(response)
        
        # Log the processed response
        logging.info("Processed response:\n%s", json.dumps(response, indent=4))

    # Restore the original stdout and close the output file
    sys.stdout = sys.__stdout__
    logfile.close()
