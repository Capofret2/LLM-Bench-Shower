from .benchbase import BaseBench
from .c_eval import C_EvalBenchmarker
from .cmmmu import CMMMUBenchmarker
from .longbench import LongBenchBenchmarker
from .longbench_v2 import LongBenchV2Benchmarker
from .mr_gms8k import MR_GMS8KBenchmarker
from .needle_in_haystack import NeedleInHaystackBenchmarker

ALL_BENCHMARKERS = {
    "C-Eval": C_EvalBenchmarker,
    "CMMMUBench": CMMMUBenchmarker,
    "LongBench": LongBenchBenchmarker,
    "LongBenchV2": LongBenchV2Benchmarker,
    "MR-GMS8K": MR_GMS8KBenchmarker,
    "NeedleInHaystack": NeedleInHaystackBenchmarker,
}


def get_benchmarker(bench_name) -> BaseBench:
    return ALL_BENCHMARKERS.get(bench_name)
