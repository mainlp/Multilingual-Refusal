from pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str, lang: str= 'en') -> ModelBase:

    if 'qwen2' in model_path.lower():
        from pipeline.model_utils.qwen2_model import Qwen2Model
        return Qwen2Model(model_path, lang)
    elif 'qwen' in model_path.lower():
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path, lang)

    if 'llama-3' in model_path.lower():
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path, lang)
    elif 'llama' in model_path.lower():
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path, lang)
    elif 'gemma-2-' in model_path.lower():
        from pipeline.model_utils.gemma2_model import Gemma2Model
        return Gemma2Model(model_path, lang)
    elif 'gemma' in model_path.lower():
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path, lang) 
    elif 'yi-1.5' in model_path.lower():
        from pipeline.model_utils.yi1_5_model import Yi1_5Model
        return Yi1_5Model(model_path, lang)
    elif 'yi' in model_path.lower():
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path, lang)

    else:
        raise ValueError(f"Unknown model family: {model_path}")
