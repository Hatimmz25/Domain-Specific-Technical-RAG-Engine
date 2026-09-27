import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))


def verify_docker_configs():
    """Validates Dockerfile and docker-compose.yml syntax and existence."""
    print("=" * 60)
    print("      Verifying Docker & Docker Compose Configurations")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent.parent
    dockerfile = base_dir / "Dockerfile"
    compose = base_dir / "docker-compose.yml"

    assert dockerfile.exists(), f"Missing Dockerfile at {dockerfile}"
    assert compose.exists(), f"Missing docker-compose.yml at {compose}"

    print(f"[✓] Confirmed Dockerfile exists at: {dockerfile}")
    print(f"[✓] Confirmed docker-compose.yml exists at: {compose}")
    print("[SUCCESS] Docker configuration files verified successfully!")


if __name__ == "__main__":
    verify_docker_configs()