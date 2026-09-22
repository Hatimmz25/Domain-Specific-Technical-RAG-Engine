import sys
import time
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import app


def run_api_tests():
    """
    Validates FastAPI endpoints (/health, /stats, /query) using FastAPI TestClient.
    """
    print("=" * 60)
    print("      Testing FastAPI REST API Server")
    print("=" * 60)

    client = TestClient(app)

    # 1. Test Health Endpoint
    print("\n[1] Testing GET /api/v1/health...")
    health_res = client.get("/api/v1/health")
    print(f"Status Code: {health_res.status_code}")
    print(f"Response Payload: {health_res.json()}")
    assert health_res.status_code == 200

    # 2. Test Stats Endpoint
    print("\n[2] Testing GET /api/v1/stats...")
    stats_res = client.get("/api/v1/stats")
    print(f"Status Code: {stats_res.status_code}")
    print(f"Response Payload: {stats_res.json()}")
    assert stats_res.status_code == 200

    # 3. Test POST /query Endpoint (Domain Query)
    print("\n[3] Testing POST /api/v1/query (Domain Question)...")
    payload = {"question": "How do I create a POST endpoint in FastAPI?"}
    query_res = client.post("/api/v1/query", json=payload)
    print(f"Status Code: {query_res.status_code}")
    data = query_res.json()
    print(f"Generated Answer:\n{data['answer']}\n")
    print(f"Source Citations Count: {len(data['sources'])}")
    print(f"Total Latency: {data['latency_ms']['total']} ms")
    assert query_res.status_code == 200

    # 4. Test Invalid Request Validation
    print("\n[4] Testing POST /api/v1/query (Short Invalid Question)...")
    bad_res = client.post("/api/v1/query", json={"question": "a"})
    print(f"Status Code: {bad_res.status_code} (Expected 422 Unprocessable Entity)")
    assert bad_res.status_code == 422

    print("\n[SUCCESS] All FastAPI endpoint tests passed!")


if __name__ == "__main__":
    run_api_tests()