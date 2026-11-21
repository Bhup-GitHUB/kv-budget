# kv-budget

kv-budget estimates how many concurrent users fit on one GPU at a given model context length.
It reports a per-user decode speed ceiling from memory bandwidth.

Build with `go build .` (Go 1.22+). GB and GB/s use decimal units.

```text
kv_per_token = 2 × layers × kv_heads × head_dim × bytes_per_kv_value
kv_per_user  = kv_per_token × (context_tokens − shared_prefix)
shared_kv    = kv_per_token × shared_prefix
weights      = params × bytes_per_weight
free         = vram − weights − reserve − shared_kv
users        = max(0, floor(free / kv_per_user))
ceiling      = bandwidth / (weights + kv_per_user + shared_kv)
```

If the entire context is shared and fits, the memory formula gives an unlimited user count.

```console
$ ./kv-budget fit --model llama-3.1-8b --gpu h100-sxm --context 131072
KV per token: 131072 bytes
KV per user: 17.180 GB
Shared KV: 0.000 GB
Users: 3
Speed ceiling: 100.78 tokens/sec
```

```console
$ ./kv-budget sweep --gpu h100-sxm
GPU       Context  KV/user (GB)  Shared KV (GB)  Users  Ceiling (tokens/sec)
h100-sxm  2048     0.268         0.000           238    205.16
h100-sxm  8192     1.074         0.000           59     195.52
h100-sxm  32768    4.295         0.000           14     164.58
h100-sxm  131072   17.180        0.000           3      100.78
```

```console
$ ./kv-budget compare --context 8192
GPU        Context  KV/user (GB)  Shared KV (GB)  Users  Ceiling (tokens/sec)
h100-sxm   8192     1.074         0.000           59     195.52
h100-pcie  8192     1.074         0.000           59     116.73
h200-sxm   8192     1.074         0.000           116    280.15
a100-80    8192     1.074         0.000           59     112.94
l4         8192     1.074         0.000           7      17.51
```

Sweep writes `sweep.csv` and includes the model maximum once. It skips contexts above that maximum or below the shared prefix.

Use `--config config.json` for a Hugging Face model. The loader derives the parameter count for Llama when the config has the necessary dimensions. Supply `--params` if the config lacks both those dimensions and `num_parameters`.
For a custom card, supply both `--vram` and `--bandwidth`.

Against a running vLLM server, use `python3 bench/check.py --help` to measure GPU memory and request rejections. vLLM can queue requests and preallocate KV memory.

The built-in model numbers come from [Meta's Llama 3.1 8B config.json](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json). H100 SXM values come from the [NVIDIA H100 specs](https://www.nvidia.com/en-us/data-center/h100/#specifications).

These are upper bounds. Real servers typically fit fewer users and decode more slowly.
