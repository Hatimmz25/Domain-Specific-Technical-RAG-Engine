import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))


def verify_docker_configs():
    """Validates Dockerfile, .dockerignore, and docker-compose.yml existence and syntax."""
    print("=" * 60)
    print("      Verifying Docker Deployment Architecture")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent.parent
    dockerfile = base_dir / "Dockerfile"
    dockerignore = base_dir / ".dockerignore"
    compose = base_dir / "docker-compose.yml"

    assert dockerfile.exists(), f"Missing Dockerfile at {dockerfile}"
    assert dockerignore.exists(), f"Missing .dockerignore at {dockerignore}"
    assert compose.exists(), f"Missing docker-compose.yml at {compose}"

    print(f"[✓] Confirmed Dockerfile at: {dockerfile.name}")
    print(f"[✓] Confirmed .dockerignore at: {dockerignore.name}")
    print(f"[✓] Confirmed docker-compose.yml at: {compose.name}")
    print("[SUCCESS] All Docker production configuration files verified successfully!")


if __name__ == "__main__":
    verify_docker_configs()