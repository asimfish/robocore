"""测试分布式训练工具。"""
import torch
from robocore.trainer.distributed import AMPTrainer, GradientAccumulator, is_main_process


def test_amp_disabled():
    amp = AMPTrainer(enabled=False)
    model = torch.nn.Linear(3, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    with amp.autocast():
        loss = model(torch.randn(2, 3)).sum()
    opt.zero_grad()
    amp.backward(loss)
    amp.step(opt)
    assert True


def test_gradient_accumulator():
    accum = GradientAccumulator(accumulation_steps=4)
    steps_taken = 0
    for i in range(8):
        accum.step()
        if accum.should_step():
            steps_taken += 1
    assert steps_taken == 2
    assert accum.scale_factor == 0.25


def test_is_main_process():
    assert is_main_process() is True
