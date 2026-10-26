"""JSON-file persistence for chunks (and optional embeddings)."""
import json
import os
import threading
from typing import Dict, List


class Store:
    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "index.json")
        os.makedirs(data_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.chunks: List[Dict] = []
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.chunks = json.load(f)

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def replace_document(self, source: str, new_chunks: List[Dict]) -> None:
        with self._lock:
            self.chunks = [c for c in self.chunks if c["source"] != source] + new_chunks
            self._save()

    def delete_document(self, source: str) -> int:
        with self._lock:
            n = len(self.chunks)
            self.chunks = [c for c in self.chunks if c["source"] != source]
            self._save()
            return n - len(self.chunks)

    def documents(self) -> List[Dict]:
        docs: Dict[str, int] = {}
        for c in self.chunks:
            docs[c["source"]] = docs.get(c["source"], 0) + 1
        return [{"source": s, "chunks": n} for s, n in sorted(docs.items())]
