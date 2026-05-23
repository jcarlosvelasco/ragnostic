import logging

from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.shared.model.RetrievedDocument import RetrievedDocument

logger = logging.getLogger(__name__)


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
                    model_name=self.model_name, cache_dir="/cache/fastembed"
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
            scores = list(self._model.rerank(query, texts))
            logger.info(f"scores: {scores}")

            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked[:k_final]]
        except Exception as e:
            print(f"[WARN] Reranker failed {e}")
            return docs[:k_final]


reranker = CrossEncoderReranker()


def rerank(
    query: str, docs: list[RetrievedDocument], k_final: int = 3
) -> list[RetrievedDocument]:
    logger.info(f"docs: {docs}")

    return reranker.rerank(query, docs, k_final)
