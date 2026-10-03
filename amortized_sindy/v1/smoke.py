from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from .checkpoint import save_checkpoint
from .model import AmortizedSINDy, AmortizedSINDyConfig
from .training import source_training_loss, source_training_step


def exponential_family(
    rates: Sequence[float],
    *,
    context_steps: int = 21,
    forecast_steps: int = 11,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trajectory-only source data for dx/dt = a*x; equations are not inputs."""
    rate_tensor = torch.tensor(rates, dtype=torch.float32)[:, None, None]
    context_times = torch.linspace(0.0, 1.0, context_steps)
    contexts = torch.exp(rate_tensor * context_times[None, :, None])
    context_grid = context_times[None, :].expand(len(rates), -1).clone()
    offsets = torch.linspace(0.0, 0.5, forecast_steps)
    futures = torch.exp(rate_tensor * (1.0 + offsets[None, :, None]))
    return contexts, context_grid, offsets, futures


def run_smoke(*, epochs: int = 120, seed: int = 17) -> tuple[AmortizedSINDy, dict[str, float]]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = AmortizedSINDy(
        AmortizedSINDyConfig(
            state_dimension=1,
            library_degree=1,
            hidden_size=24,
            min_context_steps=8,
        )
    )
    train = exponential_family((-0.8, -0.5, -0.2, 0.2, 0.5, 0.8))
    test = exponential_family((-0.35, 0.35))
    model.train()
    with torch.no_grad():
        initial = source_training_loss(
            model,
            context_states=train[0],
            context_times=train[1],
            forecast_offsets=train[2],
            future_states=train[3],
        ).rollout_mse.item()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        source_training_step(
            model,
            optimizer,
            context_states=train[0],
            context_times=train[1],
            forecast_offsets=train[2],
            future_states=train[3],
            support_weight=1e-4,
            coefficient_weight=1e-6,
        )
    model.eval()
    with torch.no_grad():
        train_predictions, _ = model.rollout(train[0], train[1], train[2])
        test_predictions, test_dynamics = model.rollout(test[0], test[1], test[2])
        train_scale = train[0].std(dim=1, unbiased=False, keepdim=True)
        test_scale = test[0].std(dim=1, unbiased=False, keepdim=True)
        final_train = torch.mean(((train_predictions - train[3]) / train_scale) ** 2)
        final_test = torch.mean(((test_predictions - test[3]) / test_scale) ** 2)
        active_fraction = torch.mean(
            (
                test_dynamics.support_probabilities
                >= model.config.support_threshold
            ).float()
        )
    metrics = {
        "initial_train_normalized_mse": float(initial),
        "final_train_normalized_mse": float(final_train),
        "held_out_parameter_normalized_mse": float(final_test),
        "held_out_active_fraction": float(active_fraction),
        "epochs": float(epochs),
        "seed": float(seed),
    }
    return model, metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the amortized SINDy mechanism smoke on a held-out parameter family."
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    model, metrics = run_smoke(epochs=args.epochs, seed=args.seed)
    if args.checkpoint is not None:
        save_checkpoint(
            args.checkpoint,
            model,
            training_manifest={
                "schema": "amortized-sindy-training-manifest-v1",
                "source_dataset_ids": ["synthetic-exponential-family-v1"],
                "excluded_evaluation_dataset_ids": ["CTF4Science/ODE_Lorenz"],
                "target_evaluation_data_used_for_training": False,
                "training_seed": args.seed,
                "training_epochs": args.epochs,
            },
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
