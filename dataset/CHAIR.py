import os
import random

from dataset.base import BaseDataset

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap



class CHAIRDataset(BaseDataset):
    def __init__(self, split="val", data_root="/data/coco2014/", sampling="random", num_samples=500):
        super(CHAIRDataset, self).__init__()
        self.ann_path = os.path.join(data_root, f"annotations/instances_{split}2014.json")
        self.caption_path = os.path.join(data_root, f"annotations/captions_{split}2014.json")
        self.img_root = os.path.join(data_root, f"{split}2014")
        self.sampling = sampling
        self.num_samples = num_samples


    def get_data(self):

        coco = COCO(self.caption_path)

        img_ids = coco.getImgIds()
        if self.num_samples:
            if self.num_samples > len(img_ids):
                print(f"num_samples {self.num_samples} exceeds the number of img_ids ({len(img_ids)})")
                sampled_img_ids = img_ids
            elif self.sampling == "first":
                sampled_img_ids = img_ids[:self.num_samples]
            elif self.sampling == "random":
                sampled_img_ids = random.sample(img_ids, self.num_samples)
            else:
                raise ValueError(f"Unsupported sampling strategy: {self.sampling}")
        else:
            sampled_img_ids = img_ids

        val_data = []
        for cur_img_id in sampled_img_ids:
            cur_img = coco.loadImgs(cur_img_id)[0]
            cur_img_path = cur_img["file_name"]
            val_data.append(
                {
                    "image_id": cur_img_id,
                    "image_path": os.path.join(self.img_root, cur_img_path),
                    "question": 'Please describe this image in detail.',
                }
            )

        return val_data