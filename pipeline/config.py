
import os

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    model_alias: str
    model_path: str
    n_train: int = 128
    n_test: int = 128
    n_val: int = 32
    filter_train: bool = True
    filter_val: bool = False
    evaluation_datasets: Tuple[str] = ("jailbreakbench",) #("",""over_refusal",
    max_new_tokens: int = 512
    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching")#     , "llamaguard2")
    refusal_eval_methodologies: Tuple[str] = ("substring_matching",)
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048
    harmtype_1: str = "or_bench_hard" # over refusal
    harmtype_2: str = "harmless"
    harmtype_3: str = "harmful"
    
    def artifact_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias, 'over_refusal_vs_harmles_vs_harmful_add_coe_0.5')