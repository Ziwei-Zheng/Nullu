import os
import json
import random

from tqdm import tqdm

from dataset.base import BaseDataset
from utils.func import read_jsonl


class POPEDataset(BaseDataset):
    def __init__(self, split="val", data_root="/data/coco/", sampling="random", num_samples=500):
        super(POPEDataset, self).__init__()
        self.testfiles = [
            './data/pope/coco_val/coco_val_pope_random.json',
            './data/pope/coco_val/coco_val_pope_popular.json',
            './data/pope/coco_val/coco_val_pope_adversarial.json',
            # './data/pope/coco_pope_random.json',
            # './data/pope/coco_pope_popular.json',
            # './data/pope/coco_pope_adversarial.json'
        ]
        self.img_root = os.path.join(data_root, f"{split}2014")
        self.sampling = sampling
        self.num_samples = num_samples


    def get_data(self):

        val_data = {}

        for testfile in self.testfiles:

            with open(testfile, 'r') as f:
                inputs = [json.loads(line) for line in f]

            groups = [inputs[i:i+6] for i in range(0, len(inputs), 6)] # 6 question for each image
            if self.num_samples:
                if self.num_samples > len(groups):
                    print(f"num_samples {self.num_samples} exceeds the number of images ({len(groups)})")
                    sampled_groups = groups
                elif self.sampling == "first":
                    sampled_groups = groups[:self.num_samples]
                elif self.sampling == "random":
                    sampled_groups = random.sample(groups, self.num_samples) # 500*6
                else:
                    raise ValueError(f"Unsupported sampling strategy: {self.sampling}")
            else:
                sampled_groups = groups

            strategy = os.path.basename(testfile).split('_')[-1].replace('.json', '')
            val_data[strategy] = [
                {
                    "image_path": os.path.join(self.img_root, ins['image']),
                    "question": ins['text'],
                    "label": ins['label']
                }
                for group in sampled_groups
                for ins in group
            ]

        return val_data
            