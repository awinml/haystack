from haystack import Document
from haystack.components.routers.document_joiner import DocumentJoiner

joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion", weights=[0.5, 0.5])


docs_set_1 = [Document(content="2"), Document(content="1"), Document(content="3")]
docs_set_2 = [Document(content="3"), Document(content="2")]

output = joiner.run([docs_set_1, docs_set_2])
print(output)

for doc in output["documents"]:
    print(f"""Document(content="{doc.content}", score={doc.score}),""")

final_ordered_docs = [
    Document(content="2", score=0.03252247488101534),
    Document(content="3", score=0.032266458495966696),
    Document(content="1", score=0.016129032258064516),
]


final_ordered_docs = [
    Document(content="2", score=0.9919354838709679),
    Document(content="3", score=0.9841269841269842),
    Document(content="1", score=0.4919354838709677),
]
