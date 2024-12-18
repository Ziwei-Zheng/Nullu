import os
import json
import argparse

def pope_calculation(response_dir):
    files = [
        os.path.join(response_dir, f)
        for f in os.listdir(response_dir)
        if f.endswith("_chat.jsonl")
    ]
    for file in files:
        pred_list = [json.loads(q)['answer'] for q in open(file, 'r')]
        label_list = [json.loads(q)['label'] for q in open(file, 'r')]

        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, label in zip(pred_list, label_list):
            pred_binary = 1 if pred == 'yes' else 0
            label_binary = 1 if label == 'yes' else 0
            
            if pred_binary == 1 and label_binary == 1:
                TP += 1
            elif pred_binary == 1 and label_binary == 0:
                FP += 1
            elif pred_binary == 0 and label_binary == 0:
                TN += 1
            elif pred_binary == 0 and label_binary == 1:
                FN += 1

        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2*precision*recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        yes_ratio = pred_list.count('yes') / len(pred_list)

        metrics = {
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 score': f1,
            'Yes ratio': yes_ratio
        }
        print('TP\tFP\tTN\tFN\t')
        print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))
        print('Accuracy: {}'.format(acc))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F1 score: {}'.format(f1))
        print('Yes ratio: {}'.format(yes_ratio))

        result_save_file = file.replace('_chat.jsonl', f'_result.json')
        with open(result_save_file, 'w') as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_dir", default="/workspace/Attack-edit/eval/pope/LLaVA-7B-lure-val-top1-mean-False")  
    args = parser.parse_args()
    pope_calculation(args.response_dir)