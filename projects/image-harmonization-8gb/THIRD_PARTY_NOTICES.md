# Provenance and third-party notices

## Scope of this source release

The project-local `LICENSE` is Apache License 2.0, consistent with the monorepo
root license. It applies only to the original code and documentation
in `projects/image-harmonization-8gb`, to the extent the contributors can grant
those rights. It does not change the licensing of other projects in this
monorepo, downloaded models, dependencies, input images, or generated outputs.
No model weights, third-party package source/binaries, sample photographs,
reference images, or generated image examples are included in this source release.
Dependencies and model weights are obtained separately at installation/runtime.

## Project code provenance

- `harmonize.py` is an original command-line wrapper, validation, and output-audit
  implementation written for this project.
- `compositing.py` is an original NumPy/SciPy implementation of masked
  low-frequency RGB gain transfer and luminance attenuation. It evolved from
  the project's own experimental `klein_compose.py`; it does not embed libcom,
  IC-Light, or another published harmonization implementation.
- `_klein_backend.py` evolved from the project's own `klein_probe.py`, written
  for the experimental evaluation. It orchestrates Hugging Face API calls,
  staged model loading, NF4 quantization, FP32 VAE decoding, and local telemetry.
  Its integration was informed by the official Diffusers
  [FLUX.2 Klein pipeline source at v0.43.0](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)
  and the official model's usage examples. It imports that pipeline at runtime;
  it does not vendor its implementation. In particular, prompt embedding logic
  is delegated to `Flux2KleinPipeline._get_qwen3_prompt_embeds`, not copied here.
  The local FP32 decode adapter wraps, rather than replaces, the upstream VAE.
- `preflight.py` and `tests/` are original diagnostic and test code. Tests create
  synthetic arrays and do not require or distribute third-party photographs.

These descriptions identify the source of this implementation, not ownership
of the underlying algorithms, model architecture, package APIs, or trademarks.
The project is not affiliated with or endorsed by the model or dependency authors.

## Model downloaded separately

Default model: **Black Forest Labs FLUX.2-klein-4B** (the distilled 4B model,
not the 9B model and not an interchangeable license claim for the whole family).

- [Official model repository and model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [Model LICENSE.md: Apache License 2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/LICENSE.md)

The model card declares `apache-2.0`. The backend downloads components from this
repository, including the text encoder, tokenizer, transformer, and VAE. The
model's license and any accompanying notices remain applicable independently
of this project's Apache-2.0 license. Runtime NF4 quantization does not replace those
terms. No quantized or original checkpoints are redistributed here. If you
redistribute model files or modified weights, review and retain their applicable
licenses/notices and meet the upstream redistribution requirements.

Apache-2.0 is not a non-commercial-only license. This does not certify that every
input, output, or deployment is legally cleared. Users remain responsible for
image rights, privacy, workplace policies, and any other applicable restrictions.
Changing the model identifier or revision may change the applicable terms.

## Software dependencies installed separately

The following are upstream license identifiers/summaries, not replacement
license texts or an exhaustive inventory of transitive and bundled components.
The license/notice files accompanying the exact installed distributions are
authoritative. Binary wheels may contain additional third-party notices.

| Component | Upstream license | License source |
|---|---|---|
| PyTorch | BSD-style / BSD-3-Clause, plus bundled-component notices | [v2.8.0 LICENSE](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE) |
| torchvision (README setup) | BSD-3-Clause | [v0.23.0 LICENSE](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE) |
| Hugging Face Diffusers | Apache-2.0 | [v0.43.0 LICENSE](https://github.com/huggingface/diffusers/blob/v0.43.0/LICENSE) |
| Hugging Face Transformers | Apache-2.0 | [v5.22.2 LICENSE](https://github.com/huggingface/transformers/blob/v5.22.2/LICENSE) |
| Hugging Face Accelerate | Apache-2.0 | [v1.15.0 LICENSE](https://github.com/huggingface/accelerate/blob/v1.15.0/LICENSE) |
| bitsandbytes | MIT | [LICENSE](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/LICENSE) |
| NumPy | BSD-3-Clause, plus bundled-component notices | [LICENSE.txt](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| SciPy | BSD-3-Clause, plus bundled-component notices | [LICENSE.txt](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| Pillow / PIL | MIT-CMU, with upstream notices | [LICENSE](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| safetensors | Apache-2.0 | [LICENSE](https://github.com/huggingface/safetensors/blob/main/LICENSE) |
| SentencePiece | Apache-2.0 | [LICENSE](https://github.com/google/sentencepiece/blob/master/LICENSE) |

Python, the NVIDIA driver, CUDA runtime/libraries, and packages installed
transitively (for example, Hugging Face Hub and tokenizers) also have their own
terms. NVIDIA components are not relicensed under this project's Apache-2.0 license.
When distributing a runnable environment or container rather than this source
alone, preserve the licenses and notices for what you actually redistribute.

## Evaluation inputs and outputs

The evaluation described in `EVALUATION.md` used publicly accessible libcom
samples. Public accessibility is not a grant of unrestricted image rights.
The original project is [bcmi/libcom](https://github.com/bcmi/libcom); sample
images and derived evaluation images are deliberately not bundled here.
Benchmark measurements and textual evaluation notes do not grant rights to the
underlying images. Use your own authorized images when running the tool.
