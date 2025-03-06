import io
import os
import zipfile
import requests
import numpy as np
import openai
import asyncio
import json
from typing import List
from langsmith import Client, traceable
from langsmith.wrappers import wrap_openai
from langsmith.schemas import Run
from langchain.smith import RunEvalConfig
from ragas.integrations.langchain import EvaluatorChain
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# Initialize LangSmith client
client = Client()
dataset_url = "https://smith.langchain.com/public/56fe54cd-b7d7-4d3b-aaa0-88d7a2d30931/d"
dataset_name = "BaseCamp Q&A"
client.clone_public_dataset(dataset_url)

# Fetch the source documents
url = "https://storage.googleapis.com/benchmarks-artifacts/basecamp-data/basecamp-data.zip"
response = requests.get(url)

data_dir = os.path.join(os.getcwd(), "data")

with io.BytesIO(response.content) as zipped_file:
    with zipfile.ZipFile(zipped_file, "r") as zip_ref:
        zip_ref.extractall(data_dir)

docs = []
for filename in os.listdir(data_dir):
    if filename.endswith(".md"):
        with open(os.path.join(data_dir, filename), "r") as file:
            docs.append({"file": filename, "content": file.read()})


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    async def from_docs(cls, docs, oai_client):
        embeddings = await oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    @traceable
    async def query(self, query: str, k: int = 5) -> List[dict]:
        embed = await self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


class NaiveRagBot:
    def __init__(self, retriever, model: str = "gpt-4-turbo-preview"):
        self._retriever = retriever
        self._client = wrap_openai(openai.AsyncClient())
        self._model = model

    @traceable
    async def get_answer(self, question: str):
        similar = await self._retriever.query(question)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                    " Use the following docs to help answer the user's question.\n\n"
                    f"## Docs\n\n{[doc['content'] for doc in similar]}",
                },
                {"role": "user", "content": question},
            ],
        )
        return {
            "answer": response.choices[0].message.content,
            "contexts": [doc["content"] for doc in similar],  # Ensure plain text
        }


# Set up evaluation metrics
evaluators = [
    EvaluatorChain(metric)
    for metric in [
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    ]
]
eval_config = RunEvalConfig(custom_evaluators=evaluators)



async def main():
    # Create retriever and bot
    retriever = await VectorStoreRetriever.from_docs(docs, openai.AsyncClient())
    rag_bot = NaiveRagBot(retriever)

    # Ask a test question
    question = "How much time off do we get?"

    response = await rag_bot.get_answer(question)

    # Print answer preview
    print("Generated Answer:", response["answer"][:150])

    # Manually log the run to LangSmith using create_run (FIX)
    run = client.create_run(name="rag_eval", run_type="chain", inputs={"question": question})

    if run is None:
        raise ValueError("Failed to create a LangSmith run. Check API key and parameters.")
    client.update_run(run.id, outputs=response)

    # Ensure dataset exists before running evaluation
    datasets = client.list_datasets()
    if dataset_name not in [d["name"] for d in datasets]:
        print(f"Dataset '{dataset_name}' not found. Please check LangSmith.")
        return

    # Run evaluation
    results = await client.arun_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=rag_bot.get_answer,
        evaluation=eval_config,
        num_examples=5,  # Reduce dataset size
    )


    # Print the evaluation scores
    print("Evaluation Results:", json.dumps(results, indent=2))


# Run the async main function
asyncio.run(main())
