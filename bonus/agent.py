"""bonus/agent.py — HybridMemoryAgent: episodic memory + stable profile.

Deliberately *not* a new subsystem: it wires together the two halves of
Lab 19 that already exist —

  * episodic memory  -> same pattern as app/search.py (BM25 + vector + RRF,
                         k=60, rank 1-based), scoped per user_id via a
                         Qdrant payload filter (same isolation mechanism
                         NB7 shows can leak if you forget it)
  * stable profile    -> the exact 3 feature views applied + materialized
                         in NB4 (app/feast_repo), reused as-is

See bonus/ARCHITECTURE.md for the design decisions and tradeoffs behind
the choices below (chunking, feature schema, freshness).

Run the demo: `python bonus/demo.py`
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter,
                                  MatchValue, PointStruct, VectorParams)
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
FEAST_REPO = ROOT / "app" / "feast_repo"  # reuses NB4's applied+materialized store
COLLECTION = "episodic_memory"
RRF_K = 60  # same default as app/search.py / NB2


@dataclass
class _Memory:
    text: str
    user_id: str
    ts: float


class HybridMemoryAgent:
    """Episodic memory (Qdrant, filtered per user) + stable profile (Feast)
    assembled into one context string for a downstream LLM. No LLM is
    called here — this class only retrieves and formats context.
    """

    def __init__(self, feast_repo_path: Path = FEAST_REPO) -> None:
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        # Kept in-process too, so BM25 can run per-user without a Qdrant
        # scroll round-trip -- fine at demo scale (see ARCHITECTURE.md).
        self._memories: dict[str, list[_Memory]] = {}
        self._next_id = 0

        self._feature_store = None
        try:
            from feast import FeatureStore
            if (feast_repo_path / "registry.db").exists():
                self._feature_store = FeatureStore(repo_path=str(feast_repo_path))
        except Exception:
            self._feature_store = None

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user.

        Chunking: per-message (the whole `text` is one chunk). See
        ARCHITECTURE.md §Decision 1 for why, and its stated limit (no
        automatic split for long pasted documents in this POC).
        """
        vec = next(self.embedder.embed([text])).tolist()
        pid = self._next_id
        self._next_id += 1
        ts = time.time()
        self.client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(
                id=pid,
                vector=vec,
                payload={"user_id": user_id, "text": text, "ts": ts},
            )],
        )
        self._memories.setdefault(user_id, []).append(
            _Memory(text=text, user_id=user_id, ts=ts)
        )

    # ── read path ───────────────────────────────────────────────────────
    def _profile(self, user_id: str) -> dict:
        """Stable profile + recent activity from Feast (NB4's feature views)."""
        if self._feature_store is None:
            return {}
        try:
            out = self._feature_store.get_online_features(
                features=[
                    "user_profile_features:reading_speed_wpm",
                    "user_profile_features:preferred_language",
                    "user_profile_features:topic_affinity",
                    "query_velocity_features:queries_last_hour",
                    "query_velocity_features:distinct_topics_24h",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
        except Exception:
            return {}
        return {k: v[0] for k, v in out.items()}

    def _hybrid_search(self, query: str, user_id: str, top_k: int = 3) -> list[str]:
        """BM25 (this user's memories only) + vector (Qdrant, filtered by
        user_id) fused with RRF -- same formula as app/search.py / NB2:
        score(d) = sum_r 1/(RRF_K + rank_r(d)), rank 1-based.
        """
        mems = self._memories.get(user_id, [])
        if not mems:
            return []

        tokenized = [m.text.lower().split() for m in mems]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())
        kw_order = sorted(range(len(mems)), key=lambda i: -scores[i])
        kw_ids = [mems[i].text for i in kw_order[: top_k * 5]]

        q_vec = next(self.embedder.embed([query])).tolist()
        vec_hits = self.client.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k * 5,
        ).points
        sem_ids = [h.payload["text"] for h in vec_hits]

        rrf: dict[str, float] = {}
        for rank, doc in enumerate(kw_ids, start=1):
            rrf[doc] = rrf.get(doc, 0.0) + 1.0 / (RRF_K + rank)
        for rank, doc in enumerate(sem_ids, start=1):
            rrf[doc] = rrf.get(doc, 0.0) + 1.0 / (RRF_K + rank)

        return [d for d, _ in sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]]

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features, return an
        assembled context string (no LLM call)."""
        profile = self._profile(user_id)
        top_memories = self._hybrid_search(query, user_id, top_k=3)

        lines = [f"User {user_id} query: {query!r}"]
        if profile:
            lines.append(
                f"Profile: thích chủ đề '{profile.get('topic_affinity', '?')}', "
                f"đọc {profile.get('reading_speed_wpm', '?')} wpm, "
                f"ngôn ngữ ưu tiên '{profile.get('preferred_language', '?')}'."
            )
            lines.append(
                f"Recent activity: {profile.get('queries_last_hour', '?')} query/giờ qua, "
                f"{profile.get('distinct_topics_24h', '?')} chủ đề khác nhau trong 24h."
            )
        else:
            lines.append(
                "Profile: (chưa có -- cần 'feast apply' + 'materialize-incremental' "
                "trong app/feast_repo trước, xem NB4)."
            )

        if top_memories:
            lines.append("Top memories:")
            for i, m in enumerate(top_memories, 1):
                lines.append(f"  {i}. {m}")
        else:
            lines.append("Top memories: (chưa có memory nào cho user này -- gọi remember() trước).")

        return "\n".join(lines)
