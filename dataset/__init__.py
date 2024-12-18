dataset_roots = {
    "lure": "/data/coco2014/",
    "chair": "/data/coco2014/",
    "pope": "/data/coco2014/",
    "opope": "/data/coco2014/"
}


def build_dataset(dataset_name, split, sampling, num_samples):
    if dataset_name == "lure":
        from .LURE import LUREDataset
        dataset = LUREDataset(split, dataset_roots[dataset_name], sampling, num_samples)
    elif dataset_name == "chair":
        from .CHAIR import CHAIRDataset
        dataset = CHAIRDataset(split, dataset_roots[dataset_name], sampling, num_samples)
    elif dataset_name == "pope":
        from .POPE import POPEDataset
        dataset = POPEDataset(split, dataset_roots[dataset_name], sampling, num_samples)
    elif dataset_name == "opope":
        from .OPOPE import POPEDataset
        dataset = POPEDataset(split, dataset_roots[dataset_name], sampling, num_samples)
    else:
        from .base import BaseDataset
        dataset = BaseDataset()
        
    return dataset.get_data()
