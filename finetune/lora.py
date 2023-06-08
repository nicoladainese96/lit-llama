"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys
from pathlib import Path
import os
import time

import lightning as L
import numpy as np
import torch
import argparse

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt

def main(
    data_dir, 
    pretrained_path, 
    tokenizer_path, 
    out_dir,
    instruction_tuning, 
    eval_interval, 
    save_interval, 
    eval_iters,
    log_interval, 
    learning_rate, 
    batch_size, 
    micro_batch_size,
    weight_decay, 
    max_seq_length, 
    lora_r, 
    lora_alpha,
    lora_dropout, 
    warmup_iters
):

    assert gradient_accumulation_iters > 0
    
    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

    config = LLaMAConfig.from_name("7B")
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)

    with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        model = LLaMA(config)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False)
    
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data, tokenizer_path, out_dir)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, "lit-llama-lora-finetuned.pth"), checkpoint)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer_path: str,
    out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0

    for iter_num in range(max_iters):

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits = model(input_ids)
            loss = loss_fn(logits, targets)
            fabric.backward(loss / gradient_accumulation_iters)

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
                
            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data, tokenizer_path)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"), checkpoint)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


def generate_response(model, instruction, tokenizer_path):
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if instruction_tuning:
        prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, tokenizer_path: str) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, instruction, tokenizer_path)
    fabric.print(instruction)
    fabric.print(output)

    model.train()
    return out.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    

def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument("--data_dir", type=str, default="data/alpaca")
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/lit-llama/7B/lit-llama.pth")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/lit-llama/tokenizer.model")
    parser.add_argument("--out_dir", type=str, default="out/lora/alpaca")

    # Additional arguments
    parser.add_argument("--instruction_tuning", type=bool, default=True)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_iters", type=int, default=100)
    
    parser.add_argument("--config_file", type=str, default=None)
    
    # Load command-line arguments from a file if specified
    if args.config_file:
        with open(args.config_file, "r") as file:
            config = json.load(file)
            parser.set_defaults(**config)
    
    args = parser.parse_args()

    # Call the main function with the command-line arguments
    main(
        args.data_dir, args.pretrained_path, args.tokenizer_path, args.out_dir,
        args.instruction_tuning, args.eval_interval, args.save_interval, args.eval_iters,
        args.log_interval, args.learning_rate, args.batch_size, args.micro_batch_size,
        args.weight_decay, args.max_seq_length, args.lora_r, args.lora_alpha,
        args.lora_dropout, args.warmup_iters
    )
