from evals.base import Evaluator


class MYTOFUEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("MYTOFU", eval_cfg, **kwargs)
