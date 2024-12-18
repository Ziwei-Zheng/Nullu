import os
# os.environ["_JAVA_OPTIONS"] = "-Xmx4g" 
import json
from tqdm import tqdm
import numpy as np

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict
# from utils import chair
from eval.utils import chair
from utils.func import read_jsonl
import argparse
data_path = '/data/coco2014/'
caption_file_path = data_path + "annotations/captions_val2014.json"


def chair_calculation(response_file):

    coco = COCO(caption_file_path)
    loaded_json = read_jsonl(response_file)

    # eliminate the items in loaded_json with the same key:
    for i in range(len(loaded_json)):
        for j in range(i + 1, len(loaded_json)):
            if loaded_json[i]["image_id"] == loaded_json[j]["image_id"]:
                loaded_json.pop(j)
                break

    print("loaded_json:", len(loaded_json))

    # construct output file as input to CHAIR evaluation
    # output format follows https://github.com/ruotianluo/self-critical.pytorch
    formulated_output_dict = {}
    # overall result
    all_overall_scores = defaultdict(list)
    # imgToEval per image result
    img_to_eval_dict = {}
    # to save memory, load 100 captions at a time
    for start_idx in tqdm(range(0, len(loaded_json), 50), desc="Generating CHAIR Input"):
        # define the current iteration end index
        end_idx = min(start_idx + 50, len(loaded_json))
        coco_res = coco.loadRes(
            loaded_json[start_idx:end_idx],
        )
        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.params["image_id"] = coco_res.getImgIds()
        coco_eval.evaluate()

        # keep track of the overall scores
        for metric, score in coco_eval.eval.items():
            all_overall_scores[metric].append(score)

        # imgToEval per image result
        for i, cur_img_id in enumerate(coco_res.getImgIds()):
            cur_eval_dict = coco_eval.evalImgs[i]
            # add caption to the eval dict
            cur_eval_dict["caption"] = coco_res.imgToAnns[cur_img_id][0]["caption"]
            img_to_eval_dict[cur_img_id] = cur_eval_dict

    # overall result
    overall_dict = {}
    for metric, score in all_overall_scores.items():
        overall_dict[metric] = np.mean(score)
    formulated_output_dict["overall"] = overall_dict
    formulated_output_dict["imgToEval"] = img_to_eval_dict

    # # sanity check the results
    # num_samples = 500
    # if len(img_to_eval_dict) != num_samples:
    #     raise Exception(
    #         f"Resulting output_dict has number of images {len(img_to_eval_dict)} different from num_samples {num_samples}"
    #     )

    print(f"\nGenerated {len(img_to_eval_dict)} samples results in CHAIR format.")

    # save the formulated output dict
    formulated_output_path = response_file.replace('.jsonl', '_chair_input.json')

    with open(formulated_output_path, "w") as f:
        json.dump(formulated_output_dict, f)



    chair_input_path = formulated_output_path

    # annotation path should be under data dir
    annotation_dir = f"{data_path}/annotations"
    # load the generated captions
    _, imids, _ = chair.load_generated_captions(chair_input_path)
    # initialize CHAIR with generated captions and annotations
    evaluator = chair.CHAIR(imids, annotation_dir)
    evaluator.get_annotations()

    # compute chair metrics
    cap_dict = evaluator.compute_chair(chair_input_path)
    # save to json pretty print
    chair_json_path = response_file.replace('.jsonl', '_chair_output.json')

    with open(chair_json_path, "w") as f:
        json.dump(cap_dict, f, indent=4)


    halc_caption_result = cap_dict["sentences"]
    halc_result = {}
    for i in halc_caption_result:
        halc_result[i["image_id"]] = {"caption": i["caption"], 
                                    # "cider": max(np.log10(i["metrics"]["CIDEr"])+20, 0),
                                    # "meteor": i["metrics"]["METEOR"],
                                    "chairs": i["metrics"]["CHAIRs"],
                                    "chairi": i["metrics"]["CHAIRi"],
                                    "bleu": (i["metrics"]["Bleu_1"] + i["metrics"]["Bleu_2"] + i["metrics"]["Bleu_3"] + i["metrics"]["Bleu_4"])/4,
                                    "objects_num": len(i["mscoco_generated_words"]),
                                    "words_num": len(i["words"]),
                                    "hallucinate_num": len(i["hallucination_idxs"])}

    cider_sum = 0
    chairs_sum = 0
    object_sum = 0
    # meteor_sum = 0
    bleu_sum = 0
    words_sum = 0
    hallucinate_sum = 0

    for i in halc_result:
        # meteor_sum += halc_result[i]["meteor"]
        bleu_sum += halc_result[i]["bleu"]
        # cider_sum += halc_result[i]["cider"]
        chairs_sum += halc_result[i]["chairs"]
        object_sum += halc_result[i]["objects_num"]
        words_sum += halc_result[i]["words_num"]
        hallucinate_sum += halc_result[i]["hallucinate_num"]

    # meteor_sum = meteor_sum / len(halc_result)
    # log_cider_sum = cider_sum / len(halc_result)
    chairs_sum = chairs_sum / len(halc_result)
    chairi_sum = hallucinate_sum / object_sum
    bleu_sum = bleu_sum / len(halc_result)
    # print("meteor: ", meteor_sum)
    # print("log_cider: ", log_cider_sum)
    print("chairs: ", chairs_sum)
    print("chairi: ", chairi_sum)
    print("bleu: ", bleu_sum)
    print("hallucinate_sum: ", hallucinate_sum)

    result = {
        "respone_file":response_file,
        # "meteor": meteor_sum,
        # "log_cider":log_cider_sum,
        "chairs": chairs_sum,
        "chairi": chairi_sum,
        "bleu": bleu_sum,
        "hallucinate_sum": hallucinate_sum
    }
    halc_path = chair_json_path.replace('_chair_output.json', '_chair_result.json')
    with open(halc_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f'saved results to {chair_json_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file", default="./eval/chair/LLaVA-7B-language_bias--top4-mean-alpha1-False_max128_noise500/raw_val_all_oe_t=0.0_b=1_42_num64_114514_chat.jsonl")
    args = parser.parse_args()
    chair_calculation(args.response_file)
