"""TTSVS: local text-to-speech with voice cloning."""

import os

# AMD ROCm: MIOpen prints harmless "IsEnoughWorkspace" warnings for every
# convolution. Show errors only, unless the user set their own level.
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Deprecation chatter from Chatterbox's dependencies (pkg_resources, diffusers
# LoRA, torch sdp_kernel). Harmless, and not something users can act on.
import warnings  # noqa: E402

for _pattern in (r".*pkg_resources is deprecated.*", r".*LoRACompatibleLinear.*",
                 r".*sdp_kernel.*", r".*output_attentions.*"):
    warnings.filterwarnings("ignore", message=_pattern)

__version__ = "0.1.0"
