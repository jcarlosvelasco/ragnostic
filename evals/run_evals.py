import asyncio
from pathlib import Path
from statistics import mean
from typing import List

import httpx
from datasets.info import json
from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas import experiment
from ragas.embeddings import BaseRagasEmbedding, OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

DATASET_PATH = "evals/golden_dataset.json"
API_URL = "http://localhost:8000/retrieve"

MODEL_NAME = "gemma4:e2b"
EMBEDDINGS_MODEL = "nomic-embed-text"

THRESHOLDS = {
    "faithfulness": 0.5,
    "answer_relevancy": 0.5,
    "context_precision": 0.5,
    "context_recall": 0.5,
}


class ExperimentResult(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


class EvaluationRow(BaseModel):
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str


@experiment(ExperimentResult)
async def run_evaluation(
    row: EvaluationRow, llm: InstructorBaseRagasLLM, emb: BaseRagasEmbedding
) -> ExperimentResult:
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=emb)
    context_precision = ContextPrecision(llm=llm, embeddings=emb)
    context_recall = ContextRecall(llm=llm, embeddings=emb)

    faith_result = await faithfulness.ascore(
        response=row.response,
        retrieved_contexts=row.retrieved_contexts,
        user_input=row.user_input,
    )

    answer_result = await answer_relevancy.ascore(
        user_input=row.user_input,
        response=row.response,
    )

    context_precision_result = await context_precision.ascore(
        user_input=row.user_input,
        reference=row.reference,
        retrieved_contexts=row.retrieved_contexts,
    )

    context_recall_result = await context_recall.ascore(
        user_input=row.user_input,
        retrieved_contexts=row.retrieved_contexts,
        reference=row.reference,
    )

    return ExperimentResult(
        faithfulness=faith_result.value,
        answer_relevancy=answer_result.value,
        context_precision=context_precision_result.value,
        context_recall=context_recall_result.value,
    )


async def run_ragas(rows: list[EvaluationRow]) -> List[ExperimentResult]:
    llm_client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    llm = llm_factory(model=MODEL_NAME, provider="openai", client=llm_client)

    emb_client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    embeddings = OpenAIEmbeddings(client=emb_client, model=EMBEDDINGS_MODEL)

    results: list[ExperimentResult] = []
    for i, row in enumerate(rows):
        print(f"Running question {i + 1}/{len(rows)}")
        try:
            result = await run_evaluation(row, llm, embeddings)
            results.append(result)
        except Exception as e:
            print(f"  ⚠️  Error en pregunta {i + 1}: {e} — skipping")
    return results


async def query_rag(question: str) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(API_URL, json={"query": question})
        response.raise_for_status()
        return response.json()


async def collect_results(golden: list[dict]) -> list[EvaluationRow]:
    rows = []
    for i, item in enumerate(golden):
        print(f"Running question {i + 1}/{len(golden)}: {item['question'][:60]}...")
        try:
            result = await query_rag(item["question"])
            contexts = result["context"]
            if isinstance(contexts, str):
                contexts = json.loads(contexts)

            rows.append(
                EvaluationRow(
                    user_input=item["question"],
                    response=result["answer"],
                    retrieved_contexts=contexts,
                    reference=item["ground_truth"],
                )
            )
        except Exception as e:
            print(f"  ⚠️  Error: {e} — skipping")
            continue
    return rows


async def main():
    with open(DATASET_PATH) as f:
        golden = json.load(f)

    print(f"Loaded {len(golden)} Q&A pairs\n")
    rows = await collect_results(golden)
    print(f"\nCollected {len(rows)} results, running RAGAS...\n")

    scores = await run_ragas(rows)

    print("\n── RESULTS BY QUESTION ─────────────────────")

    for i, score in enumerate(scores, 1):
        print(f"\nQuestion {i}")
        print(f"  Faithfulness:      {score.faithfulness:.3f}")
        print(f"  Answer relevancy:  {score.answer_relevancy:.3f}")
        print(f"  Context precision: {score.context_precision:.3f}")
        print(f"  Context recall:    {score.context_recall:.3f}")

    aggregated = {
        "faithfulness": mean(s.faithfulness for s in scores),
        "answer_relevancy": mean(s.answer_relevancy for s in scores),
        "context_precision": mean(s.context_precision for s in scores),
        "context_recall": mean(s.context_recall for s in scores),
    }

    print("\n── AVERAGE SCORES ─────────────────────")

    failed = []

    for metric, threshold in THRESHOLDS.items():
        value = aggregated[metric]

        status = "✅" if value >= threshold else "❌"

        print(f"{status} {metric}: {value:.3f} (threshold {threshold})")

        if value < threshold:
            failed.append(metric)

    result_data = {
        "num_questions": len(scores),
        "average_scores": aggregated,
        "individual_results": [score.model_dump() for score in scores],
    }

    Path("evals/results").mkdir(parents=True, exist_ok=True)

    with open("evals/results/latest.json", "w") as f:
        json.dump(result_data, f, indent=2)

    print("\nSaved results to evals/results/latest.json")

    if failed:
        print(f"\n❌ Failed metrics: {failed}")
        raise SystemExit(1)

    print("\n✅ All evals passed")


if __name__ == "__main__":
    asyncio.run(main())
