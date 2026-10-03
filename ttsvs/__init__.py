"""TTSVS: local text-to-speech with voice cloning."""

import os

# AMD ROCm: MIOpen prints harmless "IsEnoughWorkspace" warnings for every
# convolution. Show errors only, unless the user set their own level.
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")

__version__ = "0.1.0"
