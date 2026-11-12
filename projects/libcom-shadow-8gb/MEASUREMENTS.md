# Measurements and verification

## GPU test (2026-11-12 UTC)

- GPU: NVIDIA RTX PRO 4000 Blackwell, reported total 25,153,044,480 bytes.
- Software: PyTorch 2.8.0+cu128, Python 3.12, diffusers 0.34.0.
- Workload: upstream composite/mask example #1, 512 × 512, batch size 1,
  50 denoising steps, two successive calls on the same model instance.
- Precision: BF16 diffusion models, FP32 VAE, CPU text encoders/postprocessor.
- An 8 GiB PyTorch allocator cap was set **before model initialization**.
- Cumulative peak allocation: **6,057,103,360 bytes (5.64 GiB)**.
- Cumulative peak reservation: **7,474,249,728 bytes (6.96 GiB)**.
- Whole-session nvidia-smi peak: **7,428 MiB** across 645 samples, including
  initial failed FP16 tests. This is not an isolated BF16-only measurement.
- Cached initialization plus both BF16 outputs: **214.6356 seconds**.
- Both decoded tensors passed a finite-value check. Output PNGs were recovered
  and visually inspected: the original subject and scene remain visible with
  generated cast shadows. No quantitative comparison against FP32 was made.

Raw measurements: [measurements.json](measurements.json),
[driver-peak.json](driver-peak.json).

## What this does and does not establish

The capped run passed; **a physical 8 GB GPU was not tested**. PyTorch's allocator
cap does not include every driver/display allocation. Different hardware,
kernels, driver state and other applications can affect memory fit and speed.
This single input does not establish image quality across diverse scenes.

The initial FP16 run produced a non-finite decoded output warning and was not
counted as a quality pass. BF16 fixed the tested case. A BF16-to-NumPy conversion
issue was also corrected with an explicit FP32 conversion.

## Single-file packaging checks

The GPU run used the same implementation before the helpers were moved into
the replacement file. The final single-file layout was not rerun on the paid
GPU. Its model class, SDPA functions, offload/precision helpers and downloader
were checked by AST comparison against that validated implementation; their
executable definitions are unchanged.

Eighteen CPU tests passed against the **inlined** helper definitions, covering
attention equivalence, masks, IP weights, error cleanup and precision selection.
The file compiles. A full `from libcom import ShadowGenerationModel` import
also passed in a pristine checkout of the stated upstream commit with **only
this file replaced**. The inlined processors and offload helper resolved from
the replacement module itself. No extra helper file or altered upstream
attention module is required. The normal libcom package and dependencies are still required.

Upstream contains unrelated syntax errors in `bsrgan.py` and `bsrgan_light.py`
outside the tested inference path. No whole-project test pass is claimed.
