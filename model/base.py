class LargeMultimodalModel:
    def __init__(self):
        self.device = "cuda"
        pass
    
    def forward(self, image, prompt):
        return ""

def create_hook(feat_list, loc='output'):
    if loc == 'output':
        def hook(module, input, output):
            feat_list.append(output[0])
    else:
        def hook(module, input, output):
            feat_list.append(input[0])
    return hook