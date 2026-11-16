HINet / INR-Harmonization official source:
Jianqi Chen, Yilan Zhang, Zhengxia Zou, Keyan Chen, Zhenwei Shi.
Dense Pixel-to-Pixel Harmonization via Continuous Image Representation.
https://github.com/WindVChen/INR-Harmonization

upstream/ contains only inference-reachable Python sources and the unchanged
Apache-2.0 LICENSE from commit 40fa55a03ec5f2f832c48f3f98b8b83176b255dc.
Upstream source files have not been modified in this distribution.

Added files: hinet_probe.py, extract_weights_only.py, output_utils.py,
tests/test_cpu.py and packaging documentation. These additions are also
provided under Apache-2.0. Runtime adapters skip redundant ImageNet loading,
handle the legacy numpy alias, safely extract the pinned model-only checkpoint,
restore exact outside-mask pixels, and reject overwriting user output files.
