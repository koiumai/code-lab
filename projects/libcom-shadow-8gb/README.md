# libcom shadow generation (8 GiB target)

A **single replacement Python file**, not a copy of libcom. Install libcom and
its normal dependencies first, then replace only:

```text
libcom/shadow_generation/shadow_generation.py
```

Use the `shadow_generation.py` in this directory. Back up your original file.
Do not place this file at your project's top level or run it as a standalone
script: it uses the rest of the installed libcom package.

Tested against upstream [bcmi/libcom](https://github.com/bcmi/libcom), commit
`32afffddf985e3c2832262d41960c8702b8c0fcb`. Other libcom revisions are not verified.
No changes to `source/attention_processor.py`, no `source/low_vram.py`, and no
changes to `utils/model_download.py` are needed. The added helper code and the
corrected Hugging Face downloader are contained in this one replacement file.

```python
from libcom import ShadowGenerationModel

net = ShadowGenerationModel(device="cuda:0", low_vram=True)
images = net("composite.png", "mask.png", number=1)
```

Images use the upstream BGR convention; output size is 512 × 512. Additional
outputs are generated sequentially. The fixed prompt and upstream sampling
schedule are retained.

## Requirements and limits

- CUDA low-memory mode automatically selects **BF16 on Ampere or newer** and
  **FP32 with sequential CPU offload on older GPUs, including RTX 2080**.
  The new RTX 2080/Turing path has passed local functional tests but has **not
  yet been measured on a physical Turing GPU or with the full checkpoints**.
  CPU mode stays FP32.
- The FP32 low-memory path needs `accelerate` (tested with 1.15.0). If missing,
  install it with `pip install accelerate`. It keeps weights in system RAM and
  uploads only called layers, rather than trying to fit the entire FP32 UNet
  in VRAM. Attention is processed in 128-query chunks to bound memory even
  when Turing uses PyTorch's math attention kernel. Expect slower execution.
  Allow substantial system RAM, preferably 32 GB or more; this is guidance,
  not a measured minimum for the Turing path.
- FP16 produced non-finite output in the initial test. `precision="fp16"` is an
  explicit experimental option, not the default. Non-finite decoder outputs
  now raise an error rather than becoming a corrupted image.
- `low_vram=False` retains the FP32 all-resident approach and needs more VRAM.
- CPU offload trades performance for memory. Expect substantial system RAM
  usage; the test Pod had approximately 31 GB assigned RAM. Do not use the
  same model instance concurrently.
- Use libcom's normal dependencies. Blackwell testing retained a compatible
  PyTorch 2.8.0+cu128 container instead of the upstream PyTorch 2.6.0 pin.
- Weights are downloaded from the usual official sources on first use. No
  model weights are included here. Only load trusted official checkpoints.

The following numbers apply to the **BF16 path only**, not the new Turing path.
Two BF16 images completed on a **24 GB GPU under an 8 GiB PyTorch allocator
cap**. Peak PyTorch allocation was 5.64 GiB; peak reservation was 6.96 GiB.
Physical 8 GB hardware is not tested. See [MEASUREMENTS.md](MEASUREMENTS.md) for
method, limitations and the initial failed FP16 test.

## RTX 2080: how to try the experimental fallback

1. Install the normal libcom dependencies and a CUDA-enabled PyTorch build
   compatible with your driver. A CPU-only PyTorch build cannot use your GPU.
2. In the same Python environment, run `python -m pip install accelerate`.
3. Back up and replace `libcom/shadow_generation/shadow_generation.py` with
   this version. Restart the Python process or notebook kernel so the old
   imported class is no longer cached.
4. Start with one output and the default settings:

```python
from libcom import ShadowGenerationModel
net = ShadowGenerationModel(device="cuda:0", low_vram=True)
images = net("composite.png", "mask.png", number=1)
```

On RTX 2080, `net.weight_dtype` should be `torch.float32` and
`net.sequential_cpu_offload` should be `True`. Do not force BF16 or FP16.
Do not call `.to("cuda")` on the internal hooked models; that would defeat
sequential offload or fail on offloaded parameters.

This removes the BF16-only compatibility restriction, but **does not guarantee
8 GB fit or a successful full image on RTX 2080**. That hardware was unavailable
for testing. It may be much slower than the BF16 path, and system RAM can be
the next limit. If it fails, keep the full traceback and report the PyTorch,
Diffusers and Accelerate versions; do not treat the earlier BF16 measurements
as measurements for your card.

## License

Modified from Apache-2.0-licensed libcom; the upstream [LICENSE](LICENSE) is
retained. Changes include staged CPU offload, BF16 precision selection, SDPA
attention, a finite-output guard, CPU text/postprocessing, FP32 VAE handling,
an in-file Hugging Face download compatibility fix, and an experimental
FP32 sequential-offload path for pre-Ampere GPUs.

Code licensing does not replace model-weight terms. Review the current model
cards for [SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
and [libcom weights](https://huggingface.co/BCMIZB/Libcom_pretrained_models).
