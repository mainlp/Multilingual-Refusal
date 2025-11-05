# from pipeline import run_pipeline
from pipeline.run_pipeline import LMEvalHarness
from pipeline.model_utils.model_factory import construct_model_base
import os
import mmengine



def initiate_model():
    return None
    
    
    

def test_harness_eval():
    
    cfg = mmengine.Config.fromfile('configs/cfg.yaml')
    model_alias = os.path.basename(cfg.model_path)
    cfg.model_alias = model_alias
    cfg.artifact_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", cfg.model_alias, 'over_refusal_vs_harmles_vs_harmful_add_coe_0.5')
    
    model_base = construct_model_base(cfg.model_path)       
    
    cfg.eval_harness['limit'] = 1
    eval_harness_evaluator = LMEvalHarness(cfg.eval_harness)
    
    lm_eval_results = eval_harness_evaluator.evaluate(
        model=model_base
    )
    
    for task_name in cfg.eval_harness['tasks']:
        task_results = lm_eval_results[task_name]




def collect__results():
    
    pass