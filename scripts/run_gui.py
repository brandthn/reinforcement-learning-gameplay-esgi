#!/usr/bin/env python3
"""Lancer l'interface graphique Pygame."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gui.app import run

if __name__ == "__main__":
    run()
