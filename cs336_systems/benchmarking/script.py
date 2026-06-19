from pprint import pprint

from cs336_basics.model import BasicsTransformerLM
from cs336_basics import nn_utils
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
import click
import numpy as np
import timeit

import torch


@click.command()
@click.option("--vocab-size", "-v", type=int, default=10000)
@click.option("--batch-size", "-b", type=int, default=4)
@click.option("--seq-len", "-s", type=int, default=1024)
@click.option("--steps", "-e", type=int, default=10)
@click.option("--data-set", type=str, required=True)
@click.option("--d-model", "-d", type=int, required=True)
@click.option("--d-ff", "-f", type=int, required=True)
@click.option("--heads", "-h", type=int, required=True)
@click.option("--layers", "-l", type=int, required=True)
@click.option("--device", type=str, default="mps")
@click.option("--warm-up", type=int, default=5)
def run(vocab_size: int, batch_size: int, seq_len: int, steps: int, data_set: str, d_model: int, d_ff: int, heads: int, layers: int, device: str, warm_up: int):
    lm = BasicsTransformerLM(vocab_size, seq_len, d_model, layers, heads, d_ff).to(device)
    optim = AdamW(lm.parameters())
    dataset = np.memmap(data_set)
    times = []
    for i in range(warm_up + steps):
        step_times = {}
        optim.zero_grad()
        batch = get_batch(dataset, batch_size, seq_len, device)

        step_times["start"] = timeit.default_timer()

        res = lm(batch[0])
        torch.mps.synchronize()
        step_times["forward"] = timeit.default_timer()

        loss = nn_utils.cross_entropy(res, batch[1])
        loss.backward()
        torch.mps.synchronize()
        step_times["backward"] = timeit.default_timer()

        optim.step()
        torch.mps.synchronize()
        step_times["optim"] = timeit.default_timer()
        times.append(step_times)
        print("Step: ", i)
        print("Loss: ", loss.item())

    print("++++ Total warm up steps: ", warm_up)
    for t in times[warm_up:]:
        print("==== For step ", t)
        print("Forward time = ", t["forward"] - t["start"])
        print("Backward time = ", t["backward"] - t["forward"])
        print("Optimizer time = ", t["optim"] - t["backward"])
        print("Total step time = ", t["optim"] - t["start"])
        print("==== ====\n\n\n")


if __name__ == "__main__":
    run()
