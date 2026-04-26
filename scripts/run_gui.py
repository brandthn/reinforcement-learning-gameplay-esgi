#!/usr/bin/env python3
"""Lance l'interface graphique Pygame.

Usage :
    uv run scripts/run_gui.py            # mode normal
    uv run scripts/run_gui.py --debug    # mode debug (affiche l'encoding)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gui.app import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI du projet DRL")
    parser.add_argument("--debug", action="store_true",
                        help="Affiche les infos d'encoding dans le panneau lateral")
    args = parser.parse_args()
    run(debug=args.debug)
