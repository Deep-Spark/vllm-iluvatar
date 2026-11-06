# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remove the "developer preview" announcement banner on tagged releases.

On Read the Docs, a tagged (released) version should not show the banner that
tells readers they are viewing the development preview. We detect the tagged
build via the READTHEDOCS_VERSION_TYPE environment variable and delete the
overrides/main.html that injects the banner.
"""

import os
from pathlib import Path
from typing import Literal


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool):
    # https://docs.readthedocs.io/en/stable/reference/environment-variables.html
    if os.getenv("READTHEDOCS_VERSION_TYPE") == "tag":
        repo_root = Path(__file__).parent.parent
        announcement_path = repo_root / "overrides/main.html"
        # The file might already be removed if the build runs multiple times
        # (Read the Docs builds HTML and PDF separately).
        if announcement_path.exists():
            os.remove(announcement_path)
