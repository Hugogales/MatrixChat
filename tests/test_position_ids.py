import dataclasses

import torch

from model.matrix_qwen import MatrixQwenForCausalLM


def test_position_ids_flat(tiny_model, matrix_config):
    cfg = dataclasses.replace(matrix_config, position_mode="flat")
    model = MatrixQwenForCausalLM(tiny_model, cfg)

    a, t = 3, 4
    pos = model.build_position_ids(batch_size=1, num_agents=a, seq_len=t, device=torch.device("cpu"))
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert pos[0].tolist() == expected


def test_position_ids_column(tiny_model, matrix_config):
    cfg = dataclasses.replace(matrix_config, position_mode="column")
    model = MatrixQwenForCausalLM(tiny_model, cfg)

    a, t = 3, 4
    pos = model.build_position_ids(batch_size=1, num_agents=a, seq_len=t, device=torch.device("cpu"))
    expected = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert pos[0].tolist() == expected
