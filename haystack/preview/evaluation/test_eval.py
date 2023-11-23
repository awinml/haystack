import os

from haystack.preview import Pipeline
from haystack.preview.dataclasses.document import Document
from haystack.preview.components.retrievers import InMemoryBM25Retriever
from haystack.preview.document_stores import InMemoryDocumentStore
from haystack.preview.components.builders.prompt_builder import PromptBuilder

docstore = InMemoryDocumentStore()

data = [
    Document(content="This is not the answer you are looking for.", meta={"name": "Obi-Wan Kenobi"}),
    Document(content="This is the way.", meta={"name": "Mandalorian"}),
    Document(content="The answer to life, the universe and everything is 42.", meta={"name": "Deep Thought"}),
    Document(content="When you play the game of thrones, you win or you die.", meta={"name": "Cersei Lannister"}),
    Document(content="Winter is coming.", meta={"name": "Ned Stark"}),
]

# Write some fake documents
docstore.write_documents(data)

# Create our retriever, we set top_k to 3 to get only the best 3 documents otherwise by default we get 10
retriever = InMemoryBM25Retriever(document_store=docstore, top_k=3)

# Create our prompt template
template = """Given the context please answer the question.
Context:
{# We're receiving a list of lists, so we handle it like this #}

{% for doc in documents %}
    {{- doc -}};
{% endfor %}

Question: {{ question }};
Answer:
"""

prompt_builder = PromptBuilder(template)

# Build the pipeline
pipe = Pipeline()

pipe.add_component("docs_retriever", retriever)
pipe.add_component("builder", prompt_builder)


pipe.connect("docs_retriever.documents", "builder.documents")
query = "What is the answer to life, the universe and everything?"


print(type(pipe))
print(type(pipe.run({"docs_retriever": {"query": query}, "builder": {"question": query}})))
