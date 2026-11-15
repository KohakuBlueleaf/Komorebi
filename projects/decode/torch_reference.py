"""The reference of a real prompt: the model in torch on the host CPU, bf16, as its
library runs it. Tokenizes the prompt, takes a greedy continuation and the hidden
state after each layer at the first positions, into one .npz for real.py.
Run with the environment that has torch and transformers; nothing on the card."""

import argparse
import glob
import os

import numpy as np


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--hub", default=os.path.expanduser("~/.cache/huggingface"))
    parser.add_argument("--prompt", default="The tallest mountain in the world is")
    parser.add_argument("--new", type=int, default=16, help="tokens continued")
    parser.add_argument("--positions", type=int, default=4, help="hidden states at")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--out", required=True, help=".npz written")
    parser.add_argument("--decode", help="real.py's .npz: print its tokens as text")
    return parser.parse_args()


def snapshot(hub, name):
    """The model's snapshot folder in a Hugging Face hub cache."""
    pattern = f"{hub}/hub/models--{name.replace('/', '--')}/snapshots/*/config.json"
    return os.path.dirname(sorted(glob.glob(pattern))[-1])


def main():
    args = parse()
    import torch  # noqa: PLC0415 (only this script's environment has them)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    torch.set_num_threads(args.threads)
    path = snapshot(args.hub, args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if args.decode:
        got = np.load(args.decode)
        for name in ("card", "reference"):
            print(name, repr(tokenizer.decode(got[name].tolist())))
        return
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    prompt = tokenizer(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        generated = model.generate(
            prompt, max_new_tokens=args.new, do_sample=False, top_p=None, top_k=None
        )
        hidden = model(prompt, output_hidden_states=True).hidden_states
    # hidden_states: the embedding, each layer's output, the last after the final norm
    states = torch.stack(hidden[1:], dim=0)[:, 0, : args.positions].float().numpy()
    np.savez(
        args.out,
        prompt=prompt[0].numpy(),
        tokens=generated[0, prompt.shape[1] :].numpy(),
        hidden=states.transpose(1, 0, 2),
    )
    text = tokenizer.decode(generated[0, prompt.shape[1] :])
    print(f"prompt {prompt.shape[1]} tokens, continued: {text!r}")


if __name__ == "__main__":
    main()
