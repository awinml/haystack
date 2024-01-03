import logging
from typing import List, Literal, Optional, Union, Dict, Any

from haystack import ComponentError, Document, component, default_to_dict
from haystack.lazy_imports import LazyImport
from haystack.utils import get_device

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install \"sentence-transformers>=2.2.0\"'") as torch_and_transformers_import:
    import torch
    from sentence_transformers import SentenceTransformer


class DiversityRanker:
    """
    Implements a document ranking algorithm that orders documents in such a way as to maximize the overall diversity
    of the documents.
    """

    def __init__(
        self,
        model_name_or_path: str = "all-MiniLM-L6-v2",
        top_k: int = 10,
        device: Optional[str] = "cpu",
        token: Union[bool, str, None] = None,
        similarity: Literal["dot_product", "cosine"] = "dot_product",
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Initialize a DiversityRanker.

        :param model_name_or_path:  The name or path of a pretrained sentence-transformers model.
        :param top_k: The maximum number of Documents to return per query.
        :param device: The torch device (for example, cuda:0, cpu, mps) to which you want to limit model inference.
        :param token: The API token used to download private models from Hugging Face.
        :param similarity: Whether to use dot product or cosine similarity. Can be set to "dot_product" (default) or "cosine".
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document content.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document content.
        """
        torch_and_transformers_import.check()

        self.model_name_or_path = model_name_or_path
        if top_k is None or top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        self.top_k = top_k
        self.device = device
        self.token = token
        self.model = None
        self.similarity = similarity
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def warm_up(self):
        """
        Warm up the model used for scoring the Documents.
        """
        if self.model is None:
            if self.device is None:
                self.device = get_device()
            self.model = SentenceTransformer(
                model_name_or_path=self.model_name_or_path, device=self.device, use_auth_token=self.token
            )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": str(self.model_name_or_path)}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            model_name_or_path=self.model_name_or_path,
            device=self.device,
            token=self.token if not isinstance(self.token, str) else None,  # don't serialize valid tokens
            top_k=self.top_k,
            similarity=self.similarity,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Rank the documents based on their diversity and return the top_k documents.

        :param query: The query.
        :param documents: A list of Document objects that should be ranked.
        :param top_k: The maximum number of documents to return.

        :return: A list of top_k documents ranked based on diversity.
        """
        if query is None or len(query) == 0:
            raise ValueError("Query is empty")

        if not documents:
            return {"documents": []}

        if top_k is None:
            top_k = self.top_k
        elif top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        # If a model path is provided but the model isn't loaded
        if self.model_name_or_path and not self.model:
            raise ComponentError(
                f"The component {self.__class__.__name__} wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

        diversity_sorted = self.greedy_diversity_order(query=query, documents=documents)
        return {"documents": diversity_sorted[:top_k]}

    def greedy_diversity_order(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Orders the given list of documents to maximize diversity. The algorithm first calculates embeddings for
        each document and the query. It starts by selecting the document that is semantically closest to the query.
        Then, for each remaining document, it selects the one that, on average, is least similar to the already
        selected documents. This process continues until all documents are selected, resulting in a list where
        each subsequent document contributes the most to the overall diversity of the selected set.

        :param query: The search query.
        :param documents: The list of Document objects to be ranked.

        :return: A list of documents ordered to maximize diversity.
        """

        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key]
            ]
            text_to_embed = self.embedding_separator.join(meta_values_to_embed + [doc.content or ""])
            texts_to_embed.append(text_to_embed)

        # Calculate embeddings
        doc_embeddings: torch.Tensor = self.model.encode(texts_to_embed, convert_to_tensor=True)  # type: ignore
        query_embedding: torch.Tensor = self.model.encode([query], convert_to_tensor=True)  # type: ignore

        if self.similarity == "dot_product":
            doc_embeddings /= torch.norm(doc_embeddings, p=2, dim=-1).unsqueeze(-1)
            query_embedding /= torch.norm(query_embedding, p=2, dim=-1).unsqueeze(-1)

        n = len(documents)
        selected: List[int] = []

        # Compute the similarity vector between the query and documents
        query_doc_sim: torch.Tensor = query_embedding @ doc_embeddings.T

        # Start with the document with the highest similarity to the query
        selected.append(int(torch.argmax(query_doc_sim).item()))

        selected_sum = doc_embeddings[selected[0]] / n

        while len(selected) < n:
            # Compute mean of dot products of all selected documents and all other documents
            similarities = selected_sum @ doc_embeddings.T
            # Mask documents that are already selected
            similarities[selected] = torch.inf
            # Select the document with the lowest total similarity score
            index_unselected = int(torch.argmin(similarities).item())

            selected.append(index_unselected)
            # It's enough just to add to the selected vectors because dot product is distributive
            # It's divided by n for numerical stability
            selected_sum += doc_embeddings[index_unselected] / n

        ranked_docs: List[Document] = [documents[i] for i in selected]

        return ranked_docs
