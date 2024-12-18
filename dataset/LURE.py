import os
import random

from dataset.base import BaseDataset
from utils.func import read_jsonl

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap



class LUREDataset(BaseDataset):
    def __init__(self, split="train", data_root="/data/coco2014/", sampling="random", num_samples=None):
        super(LUREDataset, self).__init__()
        self.data_root = data_root
        self.ann_path = "./data/LURE/hallucination5k_train.jsonl"
        self.img_root = os.path.join(data_root, f"{split}2014")
        self.sampling = sampling
        self.num_samples = num_samples


    def get_data(self):

        annotations = read_jsonl(self.ann_path)
        
        if self.num_samples:
            if self.num_samples > len(annotations):
                print(f"num_samples {self.num_samples} exceeds the number of annotations ({len(annotations)})")
            elif self.sampling == "first":
                annotations = annotations[:self.num_samples]
            elif self.sampling == "random":
                annotations = random.sample(annotations, self.num_samples)
            else:
                raise ValueError(f"Unsupported sampling strategy: {self.sampling}")

        def build_entry(ann, answer, label):
            return {
                "image_path": os.path.join(self.img_root, ann['image']),
                "question": "Please describe this image in detail.",
                "answer": answer,
                "label": label
            }

        pos_data = [build_entry(ann, ann['h_value'], 0) for ann in annotations]
        neg_data = [build_entry(ann, ann['value'], 1) for ann in annotations]

        return pos_data, neg_data