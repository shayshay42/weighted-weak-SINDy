from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import sha256_file
from .io import atomic_save_npz, atomic_write_json
from .library import polynomial_exponents
from .multifamily_data import DATASET_SCHEMA
from .multifamily_train import _load_source
from .tabpfn_pilot import _metrics, _rollout


ODEFORMER_GATE_SCHEMA = "amortized-sindy-odeformer-gate-v1"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def symbolic_polynomial_coefficients(
    equations: str,
    *,
    dimension: int,
    degree: int = 2,
) -> np.ndarray | None:
    """Convert symbolic equations to the matched SINDy library without fitting.

    ``None`` means that the symbolic result is outside the requested polynomial
    library. This conversion never reads observations or derivatives.
    """
    try:
        import sympy
        from sympy.polys.polyerrors import PolynomialError
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("SymPy is required for ODEFormer conversion") from error

    components = [value.strip() for value in equations.split("|")]
    if len(components) != dimension:
        return None
    variables = sympy.symbols(f"x_0:{dimension}", real=True, finite=True)
    local_symbols = {f"x_{index}": value for index, value in enumerate(variables)}
    exponents = polynomial_exponents(dimension, degree)
    exponent_index = {value: index for index, value in enumerate(exponents)}
    coefficients = np.zeros((len(exponents), dimension), dtype=np.float64)
    try:
        for output, component in enumerate(components):
            expression = sympy.sympify(component, locals=local_symbols)
            polynomial = sympy.Poly(sympy.expand(expression), *variables)
            for powers, coefficient in polynomial.terms():
                if sum(powers) > degree or powers not in exponent_index:
                    return None
                numeric = float(coefficient)
                if not np.isfinite(numeric):
                    return None
                coefficients[exponent_index[powers], output] = numeric
    except (PolynomialError, TypeError, ValueError, ZeroDivisionError):
        return None
    return coefficients


def symbolic_quadratic_taylor_coefficients(
    equations: str,
    *,
    expansion_point: np.ndarray,
) -> np.ndarray | None:
    """Analytically truncate a symbolic vector field to the degree-2 library."""
    try:
        import sympy
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("SymPy is required for ODEFormer conversion") from error

    point = np.asarray(expansion_point, dtype=np.float64)
    if point.ndim != 1 or not np.isfinite(point).all():
        raise ValueError("Taylor expansion point must be a finite state vector")
    dimension = point.size
    components = [value.strip() for value in equations.split("|")]
    if len(components) != dimension:
        return None
    variables = sympy.symbols(f"x_0:{dimension}", real=True, finite=True)
    local_symbols = {f"x_{index}": value for index, value in enumerate(variables)}
    substitutions = dict(zip(variables, point.tolist()))
    exponents = polynomial_exponents(dimension, 2)
    exponent_index = {value: index for index, value in enumerate(exponents)}
    result = np.zeros((len(exponents), dimension), dtype=np.float64)
    try:
        for output, component in enumerate(components):
            expression = sympy.sympify(component, locals=local_symbols)
            approximation = sympy.S.Zero
            for powers in exponents:
                derivative = expression
                denominator = 1
                shifted_monomial = sympy.S.One
                for variable, center, power in zip(variables, point, powers):
                    if power:
                        derivative = sympy.diff(derivative, variable, power)
                        denominator *= int(sympy.factorial(power))
                        shifted_monomial *= (variable - float(center)) ** power
                value = complex(derivative.subs(substitutions).evalf())
                if not np.isfinite(value.real) or not np.isfinite(value.imag):
                    return None
                if abs(value.imag) > 1e-10:
                    return None
                approximation += (value.real / denominator) * shifted_monomial
            polynomial = sympy.Poly(sympy.expand(approximation), *variables)
            for powers, coefficient in polynomial.terms():
                if powers not in exponent_index:
                    return None
                value = float(coefficient)
                if not np.isfinite(value):
                    return None
                result[exponent_index[powers], output] = value
    except Exception:  # noqa: BLE001 - symbolic domains have many failure types
        return None
    return result


def adapt_symbolic_equations_to_sindy(
    equations: np.ndarray,
    *,
    contexts: np.ndarray,
    forecast_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map frozen symbolic outputs to local quadratic SINDy and roll them out."""
    contexts = np.asarray(contexts, dtype=np.float64)
    equations = np.asarray(equations)
    if equations.shape != (contexts.shape[0],):
        raise ValueError("one symbolic equation is required per context")
    coefficient_shape = (
        contexts.shape[0],
        len(polynomial_exponents(contexts.shape[-1], 2)),
        contexts.shape[-1],
    )
    coefficients = np.full(coefficient_shape, np.nan, dtype=np.float64)
    conversion_failed = np.zeros(contexts.shape[0], dtype=bool)
    for index, (equation, context) in enumerate(zip(equations, contexts)):
        converted = symbolic_quadratic_taylor_coefficients(
            str(equation), expansion_point=np.mean(context, axis=0)
        )
        if converted is None:
            conversion_failed[index] = True
        else:
            coefficients[index] = converted
    safe_coefficients = coefficients.copy()
    safe_coefficients[conversion_failed] = 0.0
    predictions, rollout_failed = _rollout(
        contexts[:, -1],
        np.asarray(forecast_offsets, dtype=np.float64),
        safe_coefficients,
    )
    failed = conversion_failed | rollout_failed
    predictions[failed] = np.nan
    return predictions, failed, coefficients


def condition_odeformer_sindy(
    contexts: np.ndarray,
    *,
    context_times: np.ndarray,
    forecast_offsets: np.ndarray,
    regressor: Any,
    seed: int,
    forecast_initial_index: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select frozen symbolic candidates only after conversion to SINDy.

    Each candidate is ranked by rollout error on the observed prefix. A
    candidate must also integrate finitely on the public forecast grid, but no
    forecast values are read. No coefficients are fitted or refined.
    """
    contexts = np.asarray(contexts, dtype=np.float64)
    context_times = np.asarray(context_times, dtype=np.float64)
    forecast_offsets = np.asarray(forecast_offsets, dtype=np.float64)
    if contexts.ndim != 3 or context_times.shape != (contexts.shape[1],):
        raise ValueError("invalid context array or time grid")
    if forecast_offsets.ndim != 1 or not np.isclose(forecast_offsets[0], 0.0):
        raise ValueError("forecast offsets must be a vector starting at zero")
    if forecast_initial_index not in {-1, 0}:
        raise ValueError("forecast initial index must be -1 or 0")
    _seed_everything(seed)
    count, _, dimension = contexts.shape
    coefficients = np.full(
        (count, len(polynomial_exponents(dimension, 2)), dimension),
        np.nan,
        dtype=np.float64,
    )
    predictions = np.full(
        (count, forecast_offsets.size, dimension), np.nan, dtype=np.float64
    )
    failed = np.ones(count, dtype=bool)
    context_scores = np.full(count, np.inf, dtype=np.float64)
    equations: list[str] = []
    errors: list[str] = []
    context_offsets = context_times - context_times[0]

    for index, context in enumerate(contexts):
        selected_equation = ""
        error_message = "no candidate produced a finite SINDy rollout"
        try:
            candidates = regressor.fit(
                context_times,
                context,
                sort_candidates=False,
                verbose=False,
            ).get(0, [])
            scale = np.maximum(context.std(axis=0, keepdims=True), 1e-6)
            for candidate in candidates:
                if candidate is None:
                    continue
                equation = candidate.infix()
                converted = symbolic_quadratic_taylor_coefficients(
                    equation,
                    expansion_point=np.mean(context, axis=0),
                )
                if converted is None:
                    continue
                context_prediction, context_failed = _rollout(
                    context[None, 0], context_offsets, converted[None]
                )
                if context_failed[0] or not np.isfinite(context_prediction).all():
                    continue
                forecast, forecast_failed = _rollout(
                    context[None, forecast_initial_index],
                    forecast_offsets,
                    converted[None],
                )
                if forecast_failed[0] or not np.isfinite(forecast).all():
                    continue
                score = float(np.mean(((context_prediction[0] - context) / scale) ** 2))
                if score < context_scores[index]:
                    context_scores[index] = score
                    coefficients[index] = converted
                    predictions[index] = forecast[0]
                    selected_equation = equation
                    failed[index] = False
                    error_message = ""
        except Exception:  # noqa: BLE001 - isolate each external prediction
            error_message = traceback.format_exc(limit=8).strip()
        equations.append(selected_equation)
        errors.append(error_message)

    equation_width = max(1, max((len(value) for value in equations), default=0))
    error_width = max(1, max((len(value) for value in errors), default=0))
    return (
        predictions,
        failed,
        np.asarray(equations, dtype=f"<U{equation_width}"),
        coefficients,
        context_scores,
        np.asarray(errors, dtype=f"<U{error_width}"),
    )


def _finite_only_metrics(
    predictions: np.ndarray,
    validation: dict[str, np.ndarray],
    failed: np.ndarray,
) -> dict[str, float | int]:
    finite = ~failed & np.isfinite(predictions).all(axis=(1, 2))
    if not np.any(finite):
        return {
            "finite_trajectory_count": 0,
            "finite_only_mean_normalized_mse": float("inf"),
            "finite_only_median_normalized_mse": float("inf"),
        }
    contexts = np.asarray(validation["context_states"], dtype=np.float64)
    truth = np.asarray(validation["future_states"], dtype=np.float64)
    scale = np.maximum(contexts.std(axis=1, keepdims=True), 1e-6)
    per_row = np.mean(((predictions - truth) / scale) ** 2, axis=(1, 2))
    return {
        "finite_trajectory_count": int(np.sum(finite)),
        "finite_only_mean_normalized_mse": float(np.mean(per_row[finite])),
        "finite_only_median_normalized_mse": float(np.median(per_row[finite])),
    }


def forecast_odeformer(
    contexts: np.ndarray,
    *,
    context_times: np.ndarray,
    forecast_offsets: np.ndarray,
    regressor: Any,
    seed: int,
    context_rerank: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Infer and roll out one transformer-ranked equation per context.

    The function signature intentionally has no future-trajectory argument.
    Candidate sorting against the context is disabled, so selection is solely
    the frozen transformer's rank rather than a data-fit score.
    """
    contexts = np.asarray(contexts, dtype=np.float64)
    context_times = np.asarray(context_times, dtype=np.float64)
    forecast_offsets = np.asarray(forecast_offsets, dtype=np.float64)
    if contexts.ndim != 3 or contexts.shape[-1] < 1:
        raise ValueError("contexts must have shape [trajectory,time,state]")
    if context_times.shape != (contexts.shape[1],):
        raise ValueError("context time grid has the wrong shape")
    if forecast_offsets.ndim != 1 or forecast_offsets.size < 1:
        raise ValueError("forecast offsets must be a nonempty vector")
    if not np.isclose(forecast_offsets[0], 0.0):
        raise ValueError("forecast offsets must begin at zero")

    _seed_everything(seed)
    count, _, dimension = contexts.shape
    predictions = np.full(
        (count, forecast_offsets.size, dimension), np.nan, dtype=np.float64
    )
    equation_strings: list[str] = []
    error_strings: list[str] = []
    polynomial_coefficients = np.full(
        (count, len(polynomial_exponents(dimension, 2)), dimension),
        np.nan,
        dtype=np.float64,
    )
    failed = np.ones(count, dtype=bool)

    for index, context in enumerate(contexts):
        equation = ""
        error_message = ""
        try:
            candidates = regressor.fit(
                context_times,
                context,
                sort_candidates=context_rerank,
                verbose=False,
            )
            ranked = candidates.get(0, [])
            candidate = ranked[0] if ranked else None
            if candidate is None:
                equation_strings.append(equation)
                error_strings.append("no transformer candidate")
                continue
            equation = candidate.infix()
            rollout = regressor.integrate_prediction(
                forecast_offsets,
                context[-1],
                prediction=candidate,
            )
            rollout = np.asarray(rollout, dtype=np.float64)
            if rollout.shape != predictions[index].shape:
                equation_strings.append(equation)
                error_strings.append(
                    f"rollout shape {rollout.shape} != {predictions[index].shape}"
                )
                continue
            predictions[index] = rollout
            failed[index] = not np.isfinite(rollout).all()
            converted = symbolic_polynomial_coefficients(
                equation, dimension=dimension, degree=2
            )
            if converted is not None:
                polynomial_coefficients[index] = converted
        except Exception as error:  # noqa: BLE001 - isolate each external prediction
            error_message = traceback.format_exc(limit=8).strip()
        equation_strings.append(equation)
        error_strings.append(error_message)

    width = max(1, max((len(value) for value in equation_strings), default=0))
    error_width = max(1, max((len(value) for value in error_strings), default=0))
    return (
        predictions,
        failed,
        np.asarray(equation_strings, dtype=f"<U{width}"),
        polynomial_coefficients,
        np.asarray(error_strings, dtype=f"<U{error_width}"),
    )


def _load_odeformer(
    *,
    checkpoint_path: str | Path,
    external_source_path: str | Path,
    external_revision: str,
    device: str,
    beam_size: int,
) -> tuple[Any, int]:
    import sys

    source = str(Path(external_source_path).resolve())
    try:
        actual_revision = subprocess.run(
            ["git", "-C", source, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("external ODEFormer source must be a Git checkout") from error
    if actual_revision != external_revision:
        raise ValueError(
            f"ODEFormer source revision {actual_revision} != {external_revision}"
        )
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from odeformer.model import SymbolicTransformerRegressor
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("the pinned ODEFormer source and dependencies are required") from error

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ODEFormer checkpoint not found: {checkpoint_path}")
    # The authors distribute a pickled full model rather than a weights-only
    # state dict. Only load the pinned, checksum-recorded official artifact.
    try:
        model = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:  # pragma: no cover - older supported PyTorch
        model = torch.load(checkpoint_path, map_location="cpu")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    # The released checkpoint's search decoder casts integer token IDs to its
    # floating dtype before an embedding lookup. Its sampling path shares the
    # same generator and becomes deterministic greedy decoding when the
    # temperature is None. Use one candidate and do not patch external code.
    model.beam_type = "sampling"
    model.beam_size = beam_size
    model.beam_temperature = None if beam_size == 1 else 0.1
    parameter_count = int(sum(value.numel() for value in model.parameters()))
    return SymbolicTransformerRegressor(model=model, rescale=True), parameter_count


def run_odeformer_source_validation(
    *,
    source_validation_path: str | Path,
    dataset_manifest_path: str | Path,
    checkpoint_path: str | Path,
    external_source_path: str | Path,
    external_revision: str,
    output_dir: str | Path,
    device: str = "cuda:0",
    beam_size: int = 1,
    context_rerank: bool = False,
    sindy_context_rerank: bool = False,
    seed: int = 2026,
    target_pretraining_exposure: str = "unknown",
    force: bool = False,
) -> Path:
    if not external_revision or beam_size < 1:
        raise ValueError("the ODEFormer revision and positive candidate count are required")
    if beam_size > 1 and not context_rerank:
        if not sindy_context_rerank:
            raise ValueError("multiple ODEFormer candidates require context-only reranking")
    if context_rerank and sindy_context_rerank:
        raise ValueError("choose native-symbolic or post-adapter SINDy reranking")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"ODEFormer output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(dataset_manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("invalid multifamily dataset manifest")
    if manifest.get("artifacts", {}).get("source_validation", {}).get(
        "sha256"
    ) != sha256_file(source_validation_path):
        raise ValueError("source validation does not match the dataset manifest")
    source = Path(external_source_path)
    git_head_path = source / ".git" / "HEAD"
    if not git_head_path.exists():
        raise ValueError("external ODEFormer source must be a pinned Git checkout")
    validation = _load_source(source_validation_path)
    contexts = np.asarray(validation["context_states"], dtype=np.float64)

    regressor, parameter_count = _load_odeformer(
        checkpoint_path=checkpoint_path,
        external_source_path=external_source_path,
        external_revision=external_revision,
        device=device,
        beam_size=beam_size,
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if sindy_context_rerank:
        sindy_predictions, sindy_failed, equations, taylor_coefficients, (
            context_scores
        ), errors = condition_odeformer_sindy(
            contexts,
            context_times=validation["context_times"],
            forecast_offsets=validation["forecast_offsets"],
            regressor=regressor,
            seed=seed,
        )
        predictions = np.full_like(sindy_predictions, np.nan)
        failed = np.ones_like(sindy_failed)
        exact_coefficients = np.full_like(taylor_coefficients, np.nan)
        for index, equation in enumerate(equations):
            value = (
                symbolic_polynomial_coefficients(str(equation), dimension=3)
                if equation
                else None
            )
            if value is not None:
                exact_coefficients[index] = value
        coefficients = exact_coefficients
        symbolic_metrics = None
    else:
        predictions, failed, equations, coefficients, errors = forecast_odeformer(
            contexts,
            context_times=validation["context_times"],
            forecast_offsets=validation["forecast_offsets"],
            regressor=regressor,
            seed=seed,
            context_rerank=context_rerank,
        )
        symbolic_metrics = {
            **_metrics(predictions, validation, failed),
            **_finite_only_metrics(predictions, validation, failed),
        }
        sindy_predictions, sindy_failed, taylor_coefficients = (
            adapt_symbolic_equations_to_sindy(
                equations,
                contexts=contexts,
                forecast_offsets=validation["forecast_offsets"],
            )
        )
        context_scores = np.full(contexts.shape[0], np.nan, dtype=np.float64)
    inference_seconds = time.perf_counter() - started
    metrics = {
        **_metrics(sindy_predictions, validation, sindy_failed),
        **_finite_only_metrics(sindy_predictions, validation, sindy_failed),
    }
    polynomial_rows = np.isfinite(coefficients).all(axis=(1, 2))
    prediction_path = atomic_save_npz(
        output_dir / "source_validation_predictions.npz",
        predictions=sindy_predictions,
        symbolic_predictions=predictions,
        equations=equations,
        errors=errors,
        physical_coefficients=taylor_coefficients,
        exact_polynomial_coefficients=coefficients,
        observed_context_scores=context_scores,
        group_ids=np.asarray(validation["group_ids"], dtype=np.int64),
        trajectory_ids=np.asarray(validation["trajectory_ids"], dtype=np.int64),
        forecast_offsets=np.asarray(validation["forecast_offsets"], dtype=np.float64),
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    )
    return atomic_write_json(output_dir / "manifest.json", {
        "schema": ODEFORMER_GATE_SCHEMA,
        "status": "complete",
        "selection_role": "source_family_validation_only",
        "backend": "odeformer_symbolic_transformer",
        "model": {
            "external_source_revision": external_revision,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "parameter_count": parameter_count,
            "target_pretraining_exposure": target_pretraining_exposure,
        },
        "information_contract": {
            "context_steps": int(contexts.shape[1]),
            "forecast_steps_including_initial": int(
                validation["forecast_offsets"].size
            ),
            "target_optimizer_steps": 0,
            "target_sparse_regression_solves": 0,
            "candidate_ranking": (
                "observed_context_taylor_sindy_nmse_with_public_grid_stability"
                if sindy_context_rerank
                else
                "observed_context_snmse"
                if context_rerank
                else "frozen_transformer_greedy_only"
            ),
            "decoder_path": (
                "sampling_generator_temperature_0.1"
                if beam_size > 1
                else "sampling_generator_with_temperature_none"
            ),
            "trajectory_based_candidate_sorting": context_rerank,
            "observed_context_candidate_reranking": context_rerank,
            "post_adapter_sindy_candidate_reranking": sindy_context_rerank,
            "public_grid_stability_screening": sindy_context_rerank,
            "symbolic_to_sindy_conversion": "exact_sympy_polynomial_expansion",
            "nonpolynomial_conversion": (
                "analytic_degree_2_taylor_at_context_coordinate_mean"
            ),
            "validation_future_passed_to_conditioner": False,
            "heldout_lorenz_or_ctf_data_read": False,
        },
        "metrics": {
            **metrics,
            "degree_2_polynomial_output_count": int(np.sum(polynomial_rows)),
        },
        "native_symbolic_metrics": symbolic_metrics,
        "runtime": {
            "inference_seconds": inference_seconds,
            "peak_gpu_memory_bytes": peak_memory,
            "beam_size": beam_size,
            "seed": seed,
        },
        "artifacts": {
            "source_validation_sha256": sha256_file(source_validation_path),
            "predictions": {
                "path": prediction_path.name,
                "sha256": sha256_file(prediction_path),
            },
        },
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate frozen ODEFormer equation inference on source validation."
    )
    parser.add_argument("--source-validation", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--external-source", required=True)
    parser.add_argument("--external-revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--context-rerank", action="store_true")
    parser.add_argument("--sindy-context-rerank", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--target-pretraining-exposure",
        choices=("verified_excluded", "unknown", "known_exposed"),
        default="unknown",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(run_odeformer_source_validation(
        source_validation_path=args.source_validation,
        dataset_manifest_path=args.dataset_manifest,
        checkpoint_path=args.checkpoint,
        external_source_path=args.external_source,
        external_revision=args.external_revision,
        output_dir=args.output_dir,
        device=args.device,
        beam_size=args.beam_size,
        context_rerank=args.context_rerank,
        sindy_context_rerank=args.sindy_context_rerank,
        seed=args.seed,
        target_pretraining_exposure=args.target_pretraining_exposure,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
