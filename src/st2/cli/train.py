"""ST2 train command-line entry."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from st2.engine.train import main

if __name__ == '__main__':
    sys.exit(main())
