"""Predeclared cross-device diagnostics. These never relax exact feature parity."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from atlasrank.retrieval.encoder import validate_vectors


def compare_cuda_vectors(
    cpu: NDArray[np.float32],
    cuda: NDArray[np.float32],
    *,
    component_rtol: float = 1e-4,
    component_atol: float = 1e-5,
    cosine_atol: float = 1e-4,
) -> dict[str, Any]:
    if cpu.shape != cuda.shape or cpu.ndim != 2 or not len(cpu):
        raise ValueError("CPU/CUDA identity-aligned shapes differ")
    validate_vectors(cpu, cpu.shape[1])
    validate_vectors(cuda, cpu.shape[1])
    np.testing.assert_allclose(cuda, cpu, rtol=component_rtol, atol=component_atol)
    probes = list(range(min(16, len(cpu))))
    reference = cpu[probes] @ cpu.T
    actual = cuda[probes] @ cuda.T
    np.testing.assert_allclose(actual, reference, rtol=0, atol=cosine_atol)
    rankings = []
    for a, b in zip(reference, actual, strict=True):
        order_a = np.lexsort((np.arange(len(a)), -a))
        order_b = np.lexsort((np.arange(len(b)), -b))
        for k in (20, 100):
            k = min(k, len(a))
            missing = set(order_a[:k]) - set(order_b[:k])
            if (
                missing
                and max(float(a[i]) for i in missing) - min(float(a[i]) for i in order_b[:k])
                > 2 * cosine_atol
            ):
                raise AssertionError("Clearly better CPU candidate omitted by CUDA")
            # Maximum CPU score inversion in CUDA top-k order.
            minimum = float("inf")
            inversion = 0.0
            for i in order_b[:k]:
                inversion = max(inversion, float(a[i]) - minimum)
                minimum = min(minimum, float(a[i]))
            if inversion > 2 * cosine_atol:
                raise AssertionError("CUDA ranking inversion exceeds declared numerical bound")
            rankings.append(
                {
                    "k": k,
                    "membership_exact": not missing,
                    "membership_overlap": len(set(order_a[:k]) & set(order_b[:k])) / k,
                    "order_exact": bool(np.array_equal(order_a[:k], order_b[:k])),
                    "max_cpu_inversion_gap": inversion,
                    "cpu_only_rows": sorted(int(i) for i in missing),
                    "cuda_only_rows": sorted(int(i) for i in set(order_b[:k]) - set(order_a[:k])),
                }
            )
    error = np.abs(cuda.astype(np.float64) - cpu)
    return {
        "rows": len(cpu),
        "dimensions": cpu.shape[1],
        "component_rtol": component_rtol,
        "component_atol": component_atol,
        "cosine_atol": cosine_atol,
        "component_max_absolute_error": float(error.max()),
        "component_median_absolute_error": float(np.median(error)),
        "cosine_max_absolute_error": float(np.max(np.abs(actual - reference))),
        "title_derived_probe_rows": probes,
        "retrieval_agreement": rankings,
        "near_tie_policy": "Report changes; reject inversions/membership gaps > 2*cosine_atol",
        "quality_evidence": False,
        "m6_exact_feature_parity_waived": False,
    }
