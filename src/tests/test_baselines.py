from src.algorithms.baselines.all_kv import run_all_kv
from src.algorithms.baselines.all_kv_opt import run_all_kv_opt
from src.algorithms.baselines.all_nl import run_all_nl
from src.tests.test_jmsra import small_problem


def test_baselines_return_comparable_result_schema():
    problem = small_problem()
    results = [
        run_all_nl(problem),
        run_all_kv(problem),
        run_all_kv_opt(problem),
    ]
    for result in results:
        assert isinstance(result.name, str)
        assert len(result.modes) == problem.size
        assert len(result.rho) == problem.size
        assert result.latency_seconds > 0
        assert sum(result.rho) <= 1.000001


def test_optimized_all_kv_is_not_worse_than_average_all_kv():
    problem = small_problem()
    average = run_all_kv(problem, optimized=False)
    optimized = run_all_kv_opt(problem)
    assert optimized.latency_seconds <= average.latency_seconds
