# DeepSeek-V4.1-Flash on the Wormhole Galaxy box

The token's path of DeepSeek-V4.1-Flash (text, one request, greedy) on the 32-chip Wormhole Galaxy
box, piece by piece, each piece checked against the model's own reference code run on the host.

| Folder | What |
|---|---|
| `reference/` | the reference on a host CPU: the model's own `inference/model.py`, unchanged, with CPU stand-ins for its GPU kernels (`cpu_kernels.py`), weights memory-mapped from the released checkpoint (`model_loader.py`); the fixed prompts and the saved tensors (`golden.py`) |
| `driver/` | the host loop: the token's path written out as a chain of sites (`path.py`), each site on the host or on the box by a switch (`interface.py`), the request's state (`state.py`) |
| others | the pieces on the box, one folder a worker |

## Running

Needs Python with torch (CPU), safetensors, tokenizers, transformers, sympy, numpy, and the released
checkpoint (with its `inference/` and `encoding/` folders); komorebi for the box pieces.

    # the driver with every site on the host equals the reference, bit for bit
    PYTHONPATH=projects python -m galaxy.driver.check_path --ckpt <checkpoint>
    # the reference's saved inputs and outputs of every site, and its tokens
    PYTHONPATH=projects python -m galaxy.reference.golden --ckpt <checkpoint> --out <golden dir>
    # one piece against the saved tensors
    PYTHONPATH=src:projects python -m galaxy.driver.verify --golden <golden dir> --cards <cards> \
        --piece <module>:make [--load <layers>@<kinds>[@<formats>]]
    # a prompt through the driver, pieces on the box by the switch, tokens compared
    PYTHONPATH=src:projects python -m galaxy.driver.run --ckpt <checkpoint> --golden <golden dir> \
        --prompt en --new 16 --cards <cards> --piece <module>:make@<layers> ...

A piece (`driver/interface.py`): `make(session, weights, layers)` returns an object with `sites`
(the site names it implements) that is called as `piece(site, layer, *inputs)` and returns what the
reference call returns, as host tensors of the reference's dtypes. `driver/host_piece.py` is the
example (the reference itself).

## The 18 pieces

States: `ref` (reference on the host), `box` (runs on the box), `ok` (verified on the box against the
reference). Times and errors are the owners' measurements.

| # | Piece | Sites (`driver/path.py`) | State |
|---|---|---|---|
| 1 | Weights placed and loaded across 31 chips | (loader) | - |
| 2 | Host driver loop, token by token | `TokenPath.forward` | ref |
| 3 | Embedding row (host) | `embed` | ref |
| 4 | Engram: host lookup, projections on the box | `engram_hash`, `engram_rows` (host), `engram_mix` | ref |
| 5 | The four residual streams and their mixing | `hc_*_mixes`, `hc_*_pre`, `hc_*_post` | ref |
| 6 | Norms | `attn_norm`, `ffn_norm`, `q_norm`, `kv_norm` | ref |
| 7 | Attention projections | `wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b` | box, 40 layers |
| 8 | Rotary positions | `rope_q`, `rope_kv`, `rope_ckv`, `rope_o` | box, 40 layers (`rope_ckv` host) |
| 9 | Local attention (SWA), 128-token window | `sparse_attn` | box, 40 layers |
| 10 | Compressor for the global cache | `compressor` | ref |
| 11 | Indexer and top-k selection | `indexer` | ref |
| 12 | Global attention over the selected entries | `sparse_attn` (with global entries) | ref |
| 13 | Router | `gate` | ref |
| 14 | Shared expert | `shared` | ref |
| 15 | Routed experts across the islands | `routed` | ref |
| 16 | Per-request cache and state | `window_store`, `compress_store` (`driver/state.py`) | box: the window of layers 0-1 (inside `attn`); host otherwise |
| 17 | Final norm | `final_norm` | ref |
| 18 | Head and the choice of the token | `head`, `choose` | ref |

"box": switched in the driver and run with real weights for the English prompt, tokens the
same as the reference's. The run (one Wormhole chip, 4 tokens):

    python -m projects.galaxy.driver.run --ckpt <checkpoint> --golden <golden dir> \
        --cards <card> --prompt en --new 4 \
        --piece "projects.galaxy.attention_local.piece:make@0,1@attn+2-39@wq_a,wq_b,\
    wkv,wo_a,wo_b,rope_q,rope_kv,rope_o,sparse_attn"

Logits against the reference's: rel RMS 0.180 (prefill), 0.088, 0.089, 0.149 (decode);
box time a decode step 1.55 s (projections 0.59, rotations 0.58, attention 0.39), the
host's about 10 s (routed experts 9.4 s).

## Piece 16 on the box: what an island holds between the pieces

A layer L runs its sublayers on its home island H(L); row tensors are row tiles of 32 rows
(a decode token in row 0, a prefill block in all 32). Each island holds, made once:

| name | shape | written by | read by |
|---|---|---|---|
| `streams` | [4 x 32, 5120] bf16, stream k = tile row k | `hc_*_post`, `engram_mix`, the embed row (layer 0) | `hc_*_mixes`, `hc_*_pre`, `engram_mix`, `hc_head_pre` |
| `pre_mix` | [32, 32] f32, columns 0-3 | `hc_ffn_mixes` of L | `hc_attn_pre` of L + 1 |
| `h` | [32, 5120] bf16 | `hc_*_pre`, then the norm in place | `attn`, `ffn` |
| `y` | [32, 5120] bf16 | `attn`, `ffn` | `hc_*_post` |

The mixes' `post` and `comb`, the attention's window ring and the compressor's partial group
stay inside their piece. When H(L + 1) differs from H(L), `streams` and `pre_mix` move to it.
The compressed cache and the index keys are copied to every island that reads them, each new
entry sent once when it is made, so the gather and the top-k are local to the reader. In the
first all-box run the host carries these tensors between the pieces and keeps the embed row,
the Engram lookup, the compressed cache, the index keys and the top-k.

## The modules

### `driver/check_path.py`

Check that the driver's site chain with every site on the host is the reference, bit
for bit.

    python -m galaxy.driver.check_path --ckpt <checkpoint dir> [--steps 3]

Runs the reference `Transformer.forward` for the English prompt (prefill + decode
steps), clears the state, runs `TokenPath.forward` on the same tokens, and compares the
logits of every step exactly.

### `driver/host_piece.py`

A "box piece" that runs the reference on the host: the example of the interface, and
the check of `verify.py` itself (every stateless site against golden/ gives relative RMS
0).

    python -m galaxy.driver.verify --golden <golden> --ckpt <checkpoint> \
        --piece galaxy.driver.host_piece:make --layers 3 --prompts en --steps decode0

### `driver/interface.py`

The one interface every box piece offers the driver.

A piece's module (in its worker's folder) exposes a factory:

    def make(session, weights, layers) -> BoxPiece

- `session`: the driver's open Komorebi session (the islands the piece may use are the
  session's);
- `weights`: where the piece finds its weights: the weights loader's handle (by
  checkpoint tensor name, see `weights/README.md`), or, until the loader is there, a
  `galaxy.reference.model_loader.Checkpoint` (host memory-mapped views of the released
  tensors, by the same names) from which the piece converts and uploads what it needs
  itself;
- `layers`: the layer numbers the driver will call it for (a piece may prepare per layer
  at make time).

The returned object names the sites it implements and is called once per site call:

    piece.sites                      # e.g. ("wq_a", "q_norm", "wq_b")
    piece(site, layer, *inputs)      # -> what the reference call of that site returns

Inputs and outputs are host torch tensors with the reference's shapes and dtypes
(bfloat16 stays bfloat16, float32 stays float32, int32 stays int32); the arguments of
each site are those of the reference call and are stored in golden/ under the same names. A piece
that keeps state on the box (the attention window, the global cache) reads `start_pos`
from its inputs and owns that state; the driver then never calls the host state of that
layer.

Host hooks (`interface.HOOKS`): a piece that needs a host function declares the
attribute as None and the driver sets it after `make` (in a piece process, to a call back
over the pipe); no other attribute of a piece is ever set. `compress_hook(layer, x, qr,
start_pos, offset)` returns the reference's `compress_kv` outputs (the compressed entries
and the chosen ones), for an attention piece until the box makes and chooses them.

Decode shape: one token (seqlen 1, start_pos >= 32). Prefill shape: one block of 32
tokens (start_pos 0). The driver calls exactly the reference's sequence of sites, so a
piece sees the same calls as golden/ records.

### `driver/path.py`

The token's path as a chain of sites, each site one call of the reference code.

`TokenPath.forward` is the reference `Transformer.forward` (and `Block.forward`,
`Attention.forward`, `Attention._window_kv`, `Attention._compress_kv`, `MoE.forward`,
`Engram.forward`) written out line by line, with every call of a reference function or
module going through `TokenPath.site`. A site is named, belongs to one of the 18 pieces,
and has a layer (or none). For each site the path either calls the reference (host) or
the box implementation registered for it (`TokenPath.put`), and can record the site's
inputs and outputs (`Recorder`).

With nothing registered, `TokenPath.forward` gives bit for bit the reference's logits
(checked by `python -m galaxy.driver.check_path`).

Sites that wrap other sites (chains): `attn` (the whole attention of a layer, pieces
7-12 and 16), `window_kv`, `compress_kv`, `ffn` (pieces 13-15), `engram_mix`.
Registering a box implementation for a chain takes all the sites inside it to the box at
once.

### `driver/remote.py`

A piece given as `module:make@layers...#cards` runs in a process of its own, started with
only those cards visible (two sessions in one process are refused: the second card is
opened under another driver number than its lock). The driver's side, `RemotePiece`, is
called as a piece; each site call's inputs and outputs, and each call of a host hook the
piece makes (the attention's `compress_hook`), go over a pipe as pickled host tensors. The
piece process writes its output to `<report>.piece<n>.log`.

### `driver/run.py`

The driver: a prompt in, tokens out, every site on the host or on the box by a
switch.

    python -m galaxy.driver.run --ckpt <checkpoint> --golden <golden dir> \
        --prompt en --new 16 \
        [--cards 0-15,17-31] [--load all | --load layers@kinds[@formats]] \
        [--piece module:make@layers[@sites][+layers@sites][#cards]] ... [--force]

--piece: a box piece (`driver/interface.py`): `module:make` is its factory, `layers` a
list like `0-39` or `3,20` or `-` (the global sites), the optional last part the sites
to route (default: all the piece offers); `+` adds another group of layers and sites
for the same piece (one `make` with every layer named); a last `#cards` runs the piece in a
process of its own with only those cards visible (its session, the Checkpoint as its
weights, or with `#cards%layers@kinds[@formats]` the loader's W of those weights;
`driver/remote.py`): two sessions in one process are refused. `--kernel-bytes`:
the kernel buffer of a joined session. Without any --piece no session is opened:
everything is the reference.

Free running (default): the driver's own tokens; it reports where they first differ from
the reference's (golden/<prompt>/tokens.json) and by how much the logits differ there.
--force: the reference's tokens are fed (teacher forcing) so that every step's logits
are compared. Times: prefill and decode by wall clock; tokens a second = decode steps /
their seconds.

### `driver/state.py`

Piece 16 on the host: the request's state, where it lives in the reference, and how
to clear it.

Per layer `i` (reference `model.layers[i]`):
- `attn.window_kv_cache` [1, 128, 512] bf16: the ring of the layer's own last 128 K (=
  V) rows, each already rotated and E4M3-rounded (quantized and dequantized, scale a 32,
  power of two). Position p lives in slot p % 128. Prefill of s <= 128 tokens writes
  slots 0..s-1.
- `attn.compress_kv_cache` [1, max_seq_len // ratio, 512] bf16, only in the kv source
  layers 2, 8, 14 (ratio 2) and 20 (ratio 1): compressed entry j (tokens j*ratio ..
  j*ratio+ratio-1), rotated at position j*ratio, FP4-rounded with an E4M3 scale a 16.
  Layers 3-19 read the cache of the latest source among 2, 8, 14; layers 21-39 read
  layer 20's.
- `attn.indexer.k_cache` [1, max_seq_len // ratio, 128] bf16, in the same four source
  layers: index key j, rotated (last 64), FP4-rounded with an E8M0 scale a 32.
- `attn.compressor.kv_state` / `score_state` [1, ratio, 512] f32, ratio-2 sources only:
  the first token of a group still waiting for its second (score_state -inf where
  empty).
- `model.engram_hash.cache` [1, max_seq_len] int64: the compressed token ids of the
  request (host).
- the shared choice of the latest index source (`mod.shared_attn.topk_idxs`,
  `candidates`): per step, recomputed by layers 2, 8, 14, 20, 24, 28, 32, 36 and read by
  the layers after them.

### `driver/verify.py`

Verify a box piece against golden/: the same inputs, its outputs against the
reference's.

    python -m galaxy.driver.verify --golden <golden dir> --piece module:make \
        [--cards 24-31] [--load all | --load layers@kinds[@formats]] \
        [--sites wq_a,wq_b] [--layers 3,20] \
        [--prompts en,zh] [--steps prefill,decode0,decode1]

For every golden file of the piece's sites (and the chosen layers, prompts and steps) it
rebuilds the site's inputs as the reference's tensors (dtypes from manifest.json), calls
`piece(site, layer, *inputs)`, and prints one JSON line: relative RMS (||box - ref|| /
||ref||), the largest absolute error, exact equality for integer outputs, and the call's
wall time.

Note: a piece that keeps state (window, global cache) sees the golden steps in order
(prefill, decode0, decode1) for one prompt at a time, as the driver would call it.

### `reference/cpu_kernels.py`

Host (CPU) stand-ins for the GPU kernels of the model's reference code.

The model's own `inference/kernel.py` is written for a GPU kernel language. This module
offers the same six functions with the same arguments and the same arithmetic, in plain
torch on the CPU, so that the model's own `inference/model.py` runs unchanged on a host:
`reference/model_loader.py` installs this module under the name `kernel` before it
imports `model.py`.

What is the same as the GPU kernels, by construction:
- the 8-bit and 4-bit quantizers: the same block sizes, the same lower bound of the
  block's largest value, the same power-of-two scale (2 ** ceil(log2(amax / max))) or
  E4M3 scale, the same clamp, rounding to nearest even into E4M3 / E2M1;
- the matrix products: every product of a quantized activation and a quantized weight,
  times both scales, summed in float32, the result rounded to the default dtype
  (bfloat16), as the kernels' float32 accumulators do (only the order of the float32 sum
  differs). Both operands, a quantized value times a power-of-two scale, are exact in
  bfloat16, so the products are taken as bfloat16 matrices (the CPU's bfloat16 matrix
  product sums in float32);
- the sparse attention: the gather by index (-1 = no entry), the scale, a running
  maximum over blocks of 64 entries starting at -1e30, the probabilities rounded to
  bfloat16 before they meet the values, the sink added to the denominator at the end,
  the output rounded to bfloat16;
- the hyper-connection split and Sinkhorn passes, in float32.

### `reference/expert_counts.py`

How often each routed expert of each layer is picked for the fixed prompts.

    python -m galaxy.reference.expert_counts --ckpt <checkpoint> --out <golden dir>

Runs the reference (through the driver's site chain, every site on the host) for each
prompt: the prefill block and the decode steps of its generated tokens, and counts the
router's picks (6 a token) per layer and expert: <out>/<prompt>/expert_counts.npy, int64
[40, 384], and <out>/expert_counts.npy, the sum over the prompts.

### `reference/golden.py`

Write the reference's saved inputs and outputs of every site, for the two fixed
prompts.

    python -m galaxy.reference.golden --ckpt <checkpoint> --out <golden dir> [--new 32]

For each prompt: the prefill block (32 tokens, start_pos 0) and the first two decode
steps (start_pos 32 and 33) are recorded at every site of layers 0, 1, 2, 3, 14, 20, 21,
24 (one layer of each kind of the schedule, both Engram layers) and at the global sites;
then the generation goes on, unrecorded, to `--new` tokens or the end-of-sentence token.
`tokens.json` and `logits.npy` (the logits of every step, [steps, 129280] float32) are
written for the tokens the run made; `--no-record` writes only those two. `--prompts
long --steps decode0,decode1 --layers 2,3,20,21,24` records the long prompt's decode
steps, where the indexer's choice is a real one. Layout: `reference/golden.py`.

### `reference/model_loader.py`

Load the model's own reference code and its released checkpoint on a host CPU.

The model's `inference/model.py` is imported unchanged from the checkpoint folder; its
GPU kernel module is replaced by `cpu_kernels` (same functions, same arithmetic, plain
torch). The model is built on the meta device and every parameter is then pointed at the
checkpoint's bytes through a memory map, so a token reads from disk (or the page cache)
only the weights it touches. Three kinds of parameter are converted once at load, as the
model's own `convert.py` and `load_model` would: `wo_a` (E4M3 + scales -> bfloat16), the
head (bfloat16 -> float32), and the ratio-2 compressors' `wkv`/`wgate` (bfloat16 ->
float32).

Text only: the vision tower and the speculative-decoding blocks are not built.

### `reference/prompts.py`

The fixed prompts. `en` and `zh`: each exactly 32 tokens with the model's chat
encoding (one prefill block of 32). `long`: a summary request over the start of the
checkpoint's own README.md, cut to at most LONG_TOKENS tokens (the largest cut; 1,100 or
a token less), so that a decode step's indexer chooses 512 of more than 512 entries
(ratio 2: 550 entries at 1,100 positions; layer 20: 1,100).
