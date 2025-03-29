from typing import List
from dotenv import load_dotenv
load_dotenv()

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersTextEmbedder,
    SentenceTransformersDocumentEmbedder
)
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from pathlib import Path
import time


def create_document_store(path_to_documents: List[str]):
    # Start store
    _document_store = InMemoryDocumentStore()
    # Start embedder
    document_embedder = SentenceTransformersDocumentEmbedder()
    # Create pipeline
    p = Pipeline()
    p.add_component(instance=PyPDFToDocument(), name="converter")
    p.add_component(instance=DocumentCleaner(), name="cleaner")
    p.add_component(instance=DocumentSplitter(
        split_by="period", split_length=3, split_overlap=1
    ), name="splitter")
    p.add_component(instance=document_embedder,  name="embedder")
    p.add_component(instance=DocumentWriter(document_store=_document_store), name="writer")
    p.connect("converter.documents", "cleaner.documents")
    p.connect("cleaner.documents", "splitter.documents")
    p.connect("splitter.documents", "embedder.documents")
    p.connect("embedder.documents", "writer.documents")

    # Write documents into store
    p.run(
        {
            "converter": {
                "sources": [Path(path_to_document) for path_to_document in path_to_documents]
            }
        }
    )
    return _document_store


def create_retriever_pipeline(document_store: InMemoryDocumentStore) -> Pipeline:
    # Create retrievers
    bm25_retriever = InMemoryBM25Retriever(
        document_store,
        scale_score=True,
        top_k=5
    )
    embedding_retriever = InMemoryEmbeddingRetriever(
        document_store,
        top_k=5
    )

    # Define embedder (same model as for the document store)
    text_embedder = SentenceTransformersTextEmbedder()
    text_embedder.warm_up()

    # Create joiner
    document_joiner = DocumentJoiner(
        join_mode="merge",
        weights=[0.7, 0.3],
        sort_by_score=True
    )

    # Create ThresholdFilter
    from haystack import component
    from haystack.dataclasses import Document
    from typing import List, Optional, Dict, Any

    @component
    class DocumentFilter:
        """
        Filter documents based on a specified threshold value.

        This component examines a specific field in each document (e.g., score,
        confidence) and only keeps documents that meet or exceed the threshold.
        """

        def __init__(self, threshold: float = 0.5, field_name: str = "score", top_k: int = 2):
            """
            Initialize the ThresholdFilter component.

            Args:
                threshold: The minimum value to keep a document (default: 0.5)
                field_name: The metadata field to check against the threshold (default: "score")
            """
            self.threshold = threshold
            self.field_name = field_name
            self.top_k = top_k

        @component.output_types(documents=List[Document])
        def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
            """
            Filter the input documents based on the threshold. Return only top_k.

            Args:
                documents: List of documents to filter

            Returns:
                Dict containing the filtered documents
            """
            filtered_docs = []

            for doc in documents:
                if hasattr(doc, self.field_name):
                    field_value = getattr(doc, self.field_name)
                    if field_value >= self.threshold:
                        filtered_docs.append(doc)
                else:
                    raise ValueError(f"{self.field_name} is not an attribute of the input.")

            # Return only top_k
            filtered_docs = filtered_docs[:self.top_k] if len(filtered_docs) >= self.top_k else filtered_docs
            return {"documents": filtered_docs}

    document_filter = DocumentFilter(
        threshold=0.6,
        top_k=2
    )
    # Rank both retrievers
    # ranker = TransformersSimilarityRanker()

    hybrid_retrieval = Pipeline()
    hybrid_retrieval.add_component("text_embedder", text_embedder)
    hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
    hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
    hybrid_retrieval.add_component("document_joiner", document_joiner)
    hybrid_retrieval.add_component("document_filter", document_filter)
    # hybrid_retrieval.add_component("ranker", ranker)

    hybrid_retrieval.connect("text_embedder", "embedding_retriever")
    hybrid_retrieval.connect("bm25_retriever", "document_joiner")
    hybrid_retrieval.connect("embedding_retriever", "document_joiner")
    hybrid_retrieval.connect("document_joiner", "document_filter")
    # hybrid_retrieval.connect("document_joiner", "ranker")

    return hybrid_retrieval


def get_top_k_results(query: str, retriever: Pipeline) -> List[str]:
    result = retriever.run(
        {
            "text_embedder": {"text": query},
            "bm25_retriever": {"query": query},
            # "ranker": {"query": query, "top_k": top_k}
        }
    )
    return result['document_filter']['documents']


start = time.time()

# What to search for
query = "What is the computational complexity of the transformer model"

# Create document store
document_store = create_document_store(['docs/attention.pdf'])

# Create pipeline to extract best documents
hybrid_retriever = create_retriever_pipeline(document_store)

# Extract best documents
top_k_results = get_top_k_results(
    query=query,
    retriever=hybrid_retriever
)

# print("Printing raw results from retriever")
# for doc in top_k_results:
#     print(f"{doc.score}: {doc.content}")
#     print("####")

# Create PromptBuilder
from haystack.components.builders import PromptBuilder

prompt = """
Act as a very accurate Q&A helper. You will be given some passages delimited by ``` and a question about them.
Answer the question by using solely the information provided in the passages. Take your time and read the
passages carefully before answering the question. Reason as much as you need to get to your answer.
At the very end of your answer return a JSON containing your final concise answer. For example:
```json
{n_days: 31}
```

If you are unsure, do not hesitate and return a JSON with an empty string.

Passage 1: ```{{passage_1}}```
Passage 2: ```{{passage_2}}```

Question: {{question}}

Answer:
"""

builder = PromptBuilder(
    template=prompt
)

# Create LLM Generators
from haystack.components.generators import HuggingFaceAPIGenerator

verbose_generator = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={
        'model':"meta-llama/Llama-3.2-3B-Instruct"
    },
    generation_kwargs={
        "max_new_tokens": 250,
        "temperature": 0.5,
    }
)

# Create prompt formatter
from haystack import component

@component
class PromptFormatter:
    @component.output_types(prompt=str)
    def run(self, replies: List[str]):
        # Format the output from first generator to create prompt for second generator
        reply = replies[0]
        new_prompt = (
            f"Act as JSON formatter. Format the following text into JSON."
            f"For example {{n_days: 12}}"
            f"Text: {reply}"
            f"JSON:"
        )
        return {"prompt": new_prompt}

json_generator = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={
        'model':"meta-llama/Llama-3.2-3B-Instruct"
    },
    generation_kwargs={
        "max_new_tokens": 250,
        "temperature": 0.1,
    }
)

# Create pipeline
generation_pipeline = Pipeline()
generation_pipeline.add_component("builder", builder)
generation_pipeline.add_component("verbose_generator", verbose_generator)
# generation_pipeline.add_component("formatter", PromptFormatter())
# generation_pipeline.add_component("json_generator", json_generator)
generation_pipeline.connect("builder.prompt", "verbose_generator.prompt")
# generation_pipeline.connect("verbose_generator.replies", "formatter")
# generation_pipeline.connect("formatter.prompt", "json_generator.prompt")

print(
    generation_pipeline.run(
        {
            'builder': {
                "question": query,
                "passage_1": top_k_results[0],
                "passage_2": top_k_results[1]
            }
        }
    )
)

stop = time.time()
elapsed_time = stop - start
print(f"Pipeline took {elapsed_time:.6f} seconds to run")
