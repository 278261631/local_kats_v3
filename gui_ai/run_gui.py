#!/usr/bin/env python3
"""gui_ai 启动入口。"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from main_window import main  # noqa: E402

if __name__ == "__main__":
    main()
