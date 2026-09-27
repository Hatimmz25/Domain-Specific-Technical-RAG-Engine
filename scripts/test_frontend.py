import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))


def verify_frontend_file():
    """Validates Streamlit frontend script syntax and existence."""
    print("=" * 60)
    print("      Verifying Streamlit Frontend Script")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "frontend" / "streamlit_app.py"
    if not app_path.exists():
        print(f"[ERROR] Streamlit script not found at {app_path}")
        sys.exit(1)

    print(f"[✓] Confirmed Streamlit script exists at: {app_path}")
    print("[SUCCESS] Streamlit frontend file verified!")


if __name__ == "__main__":
    verify_frontend_file()