# SPDX-License-Identifier: Apache-2.0
"""Iluvatar config overrides."""

import importlib

# A worker can unpickle the plugin config before importing vllm.config itself.
# Load the community namespace before entering the subclass module so its
# post-import callback can install that subclass without a circular import.
importlib.import_module("vllm.config.speculative")
