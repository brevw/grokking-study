#!/usr/bin/env python
"""Entry point: ``python run.py <command> ...`` (see ``python run.py --help``)."""

import sys

from grokking.cli import main

if __name__ == "__main__":
    sys.exit(main())
