import os
import argparse
import json

from tqdm import tqdm
import torch

from eval.utils import chair
from utils.func import read_jsonl
testfiles = [
    './data/pope/coco_val/coco_val_pope_random.json',
    './data/pope/coco_val/coco_val_pope_popular.json',
    './data/pope/coco_val/coco_val_pope_adversarial.json',
    # './data/pope/coco_pope_random.json',
    # './data/pope/coco_pope_popular.json',
    # './data/pope/coco_pope_adversarial.json'
]


def calculate_metrics(pred_list, label_list, beta=1):
    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):

        if pred == 1 and label == 1:
            TP += 1
        elif pred == 1 and label == 0:
            FP += 1
        elif pred == 0 and label == 0:
            TN += 1
        elif pred == 0 and label == 1:
            FN += 1

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f_beta = (1+beta**2) *precision*recall / (beta**2 * precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    metrics = {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        f'F{beta} score': f_beta,
        # 'Yes ratio': yes_ratio
    }
    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F{} score: {}'.format(beta, f_beta))
    # print('Yes ratio: {}'.format(yes_ratio))

    return metrics


def recorder_opope(node_words, pred_list, obj_words):
    for obj_word in obj_words:
        pred_list.append(1 if obj_word in node_words else 0)
    return pred_list


def prepare_save_paths(response_file, testfile, strategy_suffix):
    dir_path = os.path.join(os.path.dirname(response_file), 'results')
    os.makedirs(dir_path, exist_ok=True)

    chat_file = os.path.basename(response_file)
    save_file = os.path.join(dir_path, chat_file)

    chat_save_file = save_file.replace('_chat.jsonl', f'_{strategy_suffix}_chat.jsonl')
    result_save_file = save_file.replace('_chat.jsonl', f'_{strategy_suffix}_result.json')

    return chat_save_file, result_save_file


def process_group(responses_dict, group):

    obj_words = [ins['text'].split()[3] for ins in group]  # 'Is there a/an {} in the image?'
    image_name = group[0]['image']
    response = responses_dict.get(image_name)

    if response is None:
        print(f"Warning: No caption available for image '{image_name}'")
    
    tool = chair.CHAIR()
    _, node_words, _, _ = tool.caption_to_words(response)
    node_words = set(node_words)

    pred_list = recorder_opope(node_words, [], obj_words)
    label_list = [1 if ins['label'] == 'yes' else 0 for ins in group]
    print(f'response:{response}\n{node_words}\npred_list: {obj_words}{pred_list[-6:]}')

    for ins in group:
        ins['response'] = response

    return pred_list, label_list, group

def opope_calculation(response_file):

    responses = read_jsonl(response_file)
    responses_dict = {response['image_id']: response['caption'] for response in responses} ########
    image_id_list = list(responses_dict.keys())

    for testfile in testfiles:
        with open(testfile, 'r') as f:
            inputs = [json.loads(line) for line in f]

        groups = [inputs[i:i+6] for i in range(0, len(inputs), 6)]
        groups_dict = {group[0]['image']:group for group in groups}

        answers, all_pred_list, all_label_list = [], [], []
        for image_id in tqdm(image_id_list):
            group = groups_dict.get(image_id)
            if not group:
                print(f"Warning: No question groups available for response '{image_id}'")

            pred_list, label_list, enriched_group = process_group(responses_dict, group)
            if enriched_group:
                answers.extend(enriched_group)
                all_pred_list.extend(pred_list)
                all_label_list.extend(label_list)

        strategy_suffix = os.path.basename(testfile).split('_')[-1].replace('.json', '')
        metrics = calculate_metrics(all_pred_list, all_label_list, beta=0.2)

        chat_save_file, result_save_file = prepare_save_paths(response_file, testfile, strategy_suffix)

        with open(chat_save_file, 'w') as f:
            for answer in answers:
                f.write(json.dumps(answer) + '\n')

        with open(result_save_file, 'w') as f:
            json.dump(metrics, f, indent=4)

    print(f'Results saved to: {os.path.dirname(response_file)}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file", default="/workspace/Attack-edit/eval/opope/LLaVA-7B-lure--top8-mean-alpha1-False/LLaVA-7B_beam=1_temp=0.0_114514_114514_greedy__chat.jsonl")  
    args = parser.parse_args()
    opope_calculation(args.response_file)