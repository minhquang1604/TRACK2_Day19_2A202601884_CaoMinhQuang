"""bonus/demo.py — 5 queries minh hoạ HybridMemoryAgent.

Seed vài episodic memory cho user u_001 (đã có profile sẵn trong Feast từ
NB4: app/feast_repo, u_000..u_099), rồi hỏi 5 câu bám theo brief:
  1. vector hit đơn giản
  2. cần profile (topic_affinity)
  3. cần fresh activity (queries_last_hour)
  4. paraphrase (vector wins)
  5. mixed — hybrid + profile

Run: python bonus/demo.py   (exit 0)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import HybridMemoryAgent  # noqa: E402


def main() -> int:
    agent = HybridMemoryAgent()

    for text in [
        "Đã đọc bài về Kubernetes autoscaling cho microservices.",
        "Ghi chú: so sánh EKS vs GKE cho workload AI.",
        "Đọc security checklist cho container registry.",
        "Bài viết về tự động mở rộng hạ tầng theo lưu lượng traffic.",
        "Tóm tắt cloud security best practices 2026.",
    ]:
        agent.remember(text, user_id="u_001")

    queries = [
        ("Tôi đã đọc gì về Kubernetes?", "vector hit đơn giản"),
        ("Recommend đọc gì tiếp", "cần profile (topic_affinity)"),
        ("Tôi đang quan tâm gì gần đây?", "cần fresh activity (queries_last_hour)"),
        ("Tài liệu về tự động mở rộng hạ tầng?", "paraphrase — vector wins"),
        ("Cho tôi summary cloud security", "mixed — hybrid + profile"),
    ]

    for i, (q, note) in enumerate(queries, 1):
        print(f"\n=== Query {i} ({note}) ===")
        print(agent.recall(q, user_id="u_001"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
