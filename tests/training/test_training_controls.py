import math

import torch

from main import build_lr_scheduler
from training.config import parse_args


def test_cosine_scheduler_warms_up_then_decays():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = build_lr_scheduler(optimizer, "cosine", warmup_steps=2, total_steps=10)
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] == 0.5
    assert math.isclose(lrs[1], 1.0)
    assert lrs[2] == lrs[1]  # cosine phase starts at its maximum
    assert lrs[3] < lrs[2]
    assert lrs[-1] == 0.0


def test_constant_scheduler_stays_constant_after_warmup():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([param], lr=2.0)
    scheduler = build_lr_scheduler(optimizer, "constant", warmup_steps=2, total_steps=6)
    assert optimizer.param_groups[0]["lr"] == 1.0
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 2.0
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 2.0


def test_new_sweep_cli_controls_parse():
    args = parse_args([
        "--lr_scheduler", "cosine",
        "--warmup_steps", "200",
        "--max_grad_norm", "1.0",
        "--sampling_strategy", "balanced_cycle",
        "--checkpoint_every", "500",
        "--resume_from", "checkpoints/example",
    ])
    assert args.lr_scheduler == "cosine"
    assert args.warmup_steps == 200
    assert args.max_grad_norm == 1.0
    assert args.sampling_strategy == "balanced_cycle"
    assert args.checkpoint_every == 500
    assert args.resume_from == "checkpoints/example"
