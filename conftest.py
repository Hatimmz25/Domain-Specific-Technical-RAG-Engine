import sys
from pathlib import Path

# Add project root directory to sys.path for Pytest module discovery
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))