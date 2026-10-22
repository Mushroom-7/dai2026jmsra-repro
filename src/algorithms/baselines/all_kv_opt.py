from src.algorithms.baselines.all_kv import run_all_kv
from src.algorithms.baselines.all_nl import BaselineResult
from src.algorithms.jmsra import JmsraProblem


def run_all_kv_opt(problem: JmsraProblem) -> BaselineResult:
    return run_all_kv(problem, optimized=True)
