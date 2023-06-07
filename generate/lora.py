import os
import sys
import time
import warnings
import argparse
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer, LLaMA, LLaMAConfig
from lit_llama.lora import lora
from lit_llama.utils import EmptyInitOnDevice
from scripts.prepare_alpaca import generate_prompt

torch.set_float32_matmul_precision("high")
warnings.filterwarnings(
    # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
    "ignore",
    message="ComplexHalf support is experimental and many operators don't support it yet"
)

# Add root - not sure if this is needed
root_dir = os.curdir
print("root_dir", root_dir)
sys.path.append(root_dir)


# Argparse
parser = argparse.ArgumentParser(description='Chat with Lit-Alpaca!')

# Int entry template
# parser.add_argument('--x', type=int, help='description', default=1)
# Float entry template
# parser.add_argument('--x', type=float, help='description', default=1.0)
# String entry template
# parser.add_argument('--x', type=str, help='description', default='1')
# Bool entry template
# parser.add_argument('--x_flag', dest='x', help='description', default=False, action='store_true')
# Prompt
parser.add_argument('--prompt', type=str, help='Prompt in input to the model', default='What is a fine-tuned model?')
# Paths
parser.add_argument('--lora_path', type=str, help='Path to lora-finetuned model', default='out/lora/alpaca/lit-llama-lora-finetuned.pth')
parser.add_argument('--pretrained_path', type=str, help='Path to Lit-LLaMA pretrained model', default='checkpoints/lit-llama/7B/lit-llama.pth')
parser.add_argument('--tokenizer_path', type=str, help='Path to LLaMA tokenizer', default='checkpoints/lit-llama/tokenizer.model')

# Internal variables
parser.add_argument('--dtype', type=str, help='The dtype to use during generation.', default='bfloat16') # not sure this is supported by AMD
parser.add_argument('--max_new_tokens', type=int, help='The number of generation steps to take.', default=100)
parser.add_argument('--top_k', type=int, help='The number of top most probable tokens to consider in the sampling process.', default=200)
parser.add_argument('--temperature', type=float, help='A value controlling the randomness of the sampling process. Higher values result in more random samples.', default=0.8)
parser.add_argument('--accelerator', type=str, help='The hardware to run on. Possible choices are:``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.', default='auto')

# Lora args
parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_dropout', type=float, default=0.5)

args = parser.parse_args()

def main():
    # Load model once
    lora_path = Path(args.lora_path)
    pretrained_path = Path(args.pretrained_path)
    tokenizer_path = Path(args.tokenizer_path)

    assert lora_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    fabric = L.Fabric(accelerator=args.accelerator, devices=1)

    dt = getattr(torch, args.dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{args.dtype} is not a valid dtype.")
    dtype = dt

    with EmptyInitOnDevice(
        device=fabric.device, dtype=dtype, quantization_mode=None
    ), lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, enabled=True):
        print("Loading model ...", file=sys.stderr)
        t0 = time.time()
        model = LLaMA(LLaMAConfig())  # TODO: Support different model sizes

        # 1. Load the pretrained weights
        pretrained_checkpoint = torch.load(pretrained_path)
        model.load_state_dict(pretrained_checkpoint, strict=False)

        # 2. Load the fine-tuned LoRA weights
        lora_checkpoint = torch.load(lora_path)
        model.load_state_dict(lora_checkpoint, strict=False)

        print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)

    def truncate_output_to_eos(output, eos_id):
        # TODO: Make this more efficient, terminate generation early
        try:
            eos_pos = output.tolist().index(eos_id)
        except ValueError:
            eos_pos = -1
        
        output = output[:eos_pos]
        return output

    def reply(prompt):
        encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)
        encoded_prompt = encoded_prompt[None, :]  # add batch dimension
        print('encoded_prompt.shape ', encoded_prompt.shape, file=sys.stderr)

        sample = {"instruction": "Write completion of the following python program", "input": prompt}
        prompt = generate_prompt(sample)
        print("Actual prompt:", prompt, file=sys.stderr)
        encoded = tokenizer.encode(prompt, bos=True, eos=False)
        #encoded = encoded[None, :]  # add batch dimension -> no need for that if we use generate?
        encoded = encoded.to(model.device)
        print('encoded.shape', encoded.shape, file=sys.stderr) # now it works, shape (T,)

        t0 = time.perf_counter()
        output = generate(
            model,
            idx=encoded,
            max_seq_length=args.max_new_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print('output.shape', output.shape)

        # The end of the response is where the model generates the EOS token
        output = truncate_output_to_eos(output.cpu(), tokenizer.eos_id)
        output = tokenizer.decode(output)
        output = output.split("### Response:")[1].strip()
        output = output.split("\n")[0].strip() # quite rough way of dealing with that

        t = time.perf_counter() - t0

        return output
    print(f"Prompt: {args.prompt}")
    # Generate and print response
    response = reply(args.prompt)
    print(f"Answer: {response}")


if __name__ == "__main__":
    main()

