def build_model(args):
    if args.model_name == "LLaVA-7B":
        from .LLaVA import LLaVA
        model = LLaVA(args)
    elif args.model_name == "MiniGPT4":
        from .MiniGPT4 import MiniGPT4
        model = MiniGPT4(args)
    elif args.model_name == "mPLUG_Owl2":
        from .mPLUG_Owl2 import mPLUG_Owl2
        model = mPLUG_Owl2(args)
    elif args.model_name == "Qwen_VL_Chat":
        from .Qwen_VL_Chat import Qwen_VL_Chat
        model = Qwen_VL_Chat(args)
    else:
        model = None
        
    return model
