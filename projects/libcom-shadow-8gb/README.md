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

- CUDA low-memory mode defaults to **BF16**, requiring NVIDIA Ampere or newer
  (for example, RTX 30-series or newer). CPU mode stays FP32.
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

Two BF16 images completed on a **24 GB GPU under an 8 GiB PyTorch allocator
cap**. Peak PyTorch allocation was 5.64 GiB; peak reservation was 6.96 GiB.
Physical 8 GB hardware is not tested. See [MEASUREMENTS.md](MEASUREMENTS.md) for
method, limitations and the initial failed FP16 test.

## License

Modified from Apache-2.0-licensed libcom; the upstream [LICENSE](LICENSE) is
retained. Changes include staged CPU offload, BF16 precision selection, SDPA
attention, a finite-output guard, CPU text/postprocessing, FP32 VAE handling,
and an in-file Hugging Face download compatibility fix.

Code licensing does not replace model-weight terms. Review the current model
cards for [SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
and [libcom weights](https://huggingface.co/BCMIZB/Libcom_pretrained_models).
