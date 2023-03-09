# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


#fire.Fire(main)

with open("config.json", "r") as f:
    config = json.loads(f.read())


ckpt_dir = config['model_path'] + config['model_size']
tokenizer_path = config['model_path'] 
temperature = 0.8
top_p =  0.95
max_seq_len = 512
max_batch_size = 32

local_rank, world_size = setup_model_parallel()
if local_rank > 0:
    sys.stdout = open(os.devnull, "w")

generator = load(
    ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
)

prompts = ["I believe the meaning of life is","Simply put, the theory of relativity states that "]
results = generator.generate(prompts, max_gen_len=256, temperature=temperature, top_p=top_p )

for result in results:
    print(result)
    print("\n==================================\n")
