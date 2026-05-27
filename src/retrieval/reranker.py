import logging
import os

from fastembed.rerank.cross_encoder import TextCrossEncoder

from settings import settings
from src.shared.langfuse import langfuse
from src.shared.model.RetrievedDocument import RetrievedDocument

logger = logging.getLogger(__name__)
FASTEMBED_CACHE_PATH = os.getenv("FASTEMBED_CACHE_PATH", "/cache/fastembed")


class CrossEncoderReranker:
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: TextCrossEncoder | None = None
        self._load_failed = False

    def _ensure_model(self) -> bool:
        if self._load_failed:
            return False
        if self._model is None:
            logger.info(f"Loading reranker model: {self.model_name}...")
            try:
                self._model = TextCrossEncoder(
                    model_name=self.model_name, cache_dir=FASTEMBED_CACHE_PATH
                )
                logger.info("Reranker model loaded successfully")
            except Exception as e:
                self._load_failed = True
                logger.warning(f"Reranker unavailable: {e}")
                return False
        return True

    def rerank(
        self, query: str, docs: list[RetrievedDocument], k_final: int = 3
    ) -> list[RetrievedDocument]:
        if not docs:
            return docs

        if not self._ensure_model() or self._model is None:
            logger.warning("Reranker model not available, using original order")
            return docs[:k_final]

        texts = [doc.content for doc in docs]

        try:
            logger.info(f"reranking: query={query}, texts={texts}")

            trace = langfuse.trace(name="reranker")
            span = trace.span(
                name="cross-encoder-rerank",
                input={"query": query, "num_docs": len(docs), "k_final": k_final},
            )

            scores = list(self._model.rerank(query, texts))
            logger.info(f"scores: {scores}")

            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            result = [doc for doc, _ in ranked[:k_final]]

            span.end(
                output={
                    "num_results": len(result),
                    "top_score": float(scores[0]) if scores else None,
                }
            )
            langfuse.flush()

            return result
        except Exception as e:
            print(f"[WARN] Reranker failed {e}")
            return docs[:k_final]


reranker = CrossEncoderReranker(model_name=settings.reranker_model)


def rerank(
    query: str, docs: list[RetrievedDocument], k_final: int = 3
) -> list[RetrievedDocument]:
    # logger.info(f"docs: {docs}")

    return reranker.rerank(query, docs, k_final)
