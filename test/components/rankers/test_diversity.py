import pytest

from typing import List

from haystack import Document
from haystack.components.rankers import DiversityRanker


class TestDiversityRanker:
    def test_init(self):
        component = DiversityRanker()
        assert component.model_name_or_path == "all-MiniLM-L6-v2"
        assert component.top_k == 10
        assert component.device == "cpu"
        assert component.similarity == "dot_product"
        assert component.token is None
        assert component.meta_fields_to_embed == []
        assert component.embedding_separator == "\n"

    def test_init_with_custom_init_parameters(self):
        component = DiversityRanker(
            model_name_or_path="msmarco-distilbert-base-v4",
            top_k=5,
            device="cuda",
            token="hf_token",
            similarity="cosine",
            meta_fields_to_embed=["meta_field"],
            embedding_separator="--",
        )
        assert component.model_name_or_path == "msmarco-distilbert-base-v4"
        assert component.top_k == 5
        assert component.device == "cuda"
        assert component.similarity == "cosine"
        assert component.token == "hf_token"
        assert component.meta_fields_to_embed == ["meta_field"]
        assert component.embedding_separator == "--"

    def test_to_dict(self):
        component = DiversityRanker()
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.diversity.DiversityRanker",
            "init_parameters": {
                "model_name_or_path": "all-MiniLM-L6-v2",
                "top_k": 10,
                "device": "cpu",
                "similarity": "dot_product",
                "token": None,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = DiversityRanker(
            model_name_or_path="msmarco-distilbert-base-v4",
            top_k=5,
            device="cuda",
            token="hf_token",
            similarity="cosine",
            meta_fields_to_embed=["meta_field"],
            embedding_separator="--",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.diversity.DiversityRanker",
            "init_parameters": {
                "model_name_or_path": "msmarco-distilbert-base-v4",
                "top_k": 5,
                "device": "cuda",
                "token": None,  # Valid tokens are not serialized
                "similarity": "cosine",
                "meta_fields_to_embed": ["meta_field"],
                "embedding_separator": "--",
            },
        }

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_returns_list_of_documents(self, similarity: str):
        """
        Tests that run method returns a list of Document objects
        """
        ranker = DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", similarity=similarity)  # type: ignore
        ranker.warm_up()
        query = "test query"
        documents = [Document(content="doc1"), Document(content="doc2")]
        result = ranker.run(query=query, documents=documents)
        ranked_docs = result["documents"]

        assert isinstance(ranked_docs, list)
        assert len(ranked_docs) == 2
        assert all(isinstance(doc, Document) for doc in ranked_docs)

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_returns_correct_number_of_documents(self, similarity: str):
        """
        Tests that run method returns the correct number of documents
        """
        ranker = DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", similarity=similarity)  # type: ignore
        ranker.warm_up()
        query = "test query"
        documents = [Document(content="doc1"), Document(content="doc2")]
        result = ranker.run(query=query, documents=documents, top_k=1)
        ranked_docs = result["documents"]

        assert len(ranked_docs) == 1

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_diversity_ranker_with_documents_less_than_top_k(self, similarity: str):
        """
        Tests that run method returns the correct order of documents for edge cases
        """
        ranker = DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", similarity=similarity, top_k=5)  # type: ignore
        ranker.warm_up()
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
        result = ranker.run(query=query, documents=documents)

        assert len(result["documents"]) == 3

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_diversity_ranker_negative_top_k(self, similarity: str):
        """
        Tests that run method raises an error for negative top-k.
        """
        ranker = DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", similarity=similarity, top_k=10)  # type: ignore
        ranker.warm_up()
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]

        # Setting top_k at runtime
        with pytest.raises(ValueError):
            ranker.run(query=query, documents=documents, top_k=-5)

        # Setting top_k at init
        with pytest.raises(ValueError):
            DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", similarity=similarity, top_k=-5)  # type: ignore

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_diversity_ranker_none_top_k(self, similarity: str):
        """
        Tests that run method returns the correct order of documents for top-k set to None.
        Setting top_k to None is ignored during runtime, it should use top_k set at init.
        """

        ranker = DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", similarity=similarity, top_k=2)  # type: ignore
        ranker.warm_up()
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
        result = ranker.run(query=query, documents=documents, top_k=None)

        assert len(result["documents"]) == 2

        # Setting top_k to None at init should raise error
        with pytest.raises(ValueError):
            DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", similarity=similarity, top_k=None)  # type: ignore

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_returns_documents_in_correct_order(self, similarity: str):
        """
        Tests that run method returns documents in the correct order
        """
        ranker = DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", similarity=similarity)  # type: ignore
        ranker.warm_up()
        query = "city"
        documents = [
            Document(content="France"),
            Document(content="Germany"),
            Document(content="Eiffel Tower"),
            Document(content="Berlin"),
            Document(content="Bananas"),
            Document(content="Silicon Valley"),
            Document(content="Brandenburg Gate"),
        ]
        result = ranker.run(query=query, documents=documents)
        ranked_docs = result["documents"]
        ranked_order = ", ".join([doc.content for doc in ranked_docs])
        expected_order = "Berlin, Bananas, Eiffel Tower, Silicon Valley, France, Brandenburg Gate, Germany"

        assert ranked_order == expected_order

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_single_document_corner_case(self, similarity: str):
        """
        Tests that run method returns the correct number of documents for a single document
        """
        ranker = DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", similarity=similarity)  # type: ignore
        ranker.warm_up()
        query = "test"
        documents = [Document(content="doc1")]
        result = ranker.run(query=query, documents=documents)

        assert len(result["documents"]) == 1

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_raises_value_error_if_query_is_empty(self, similarity: str):
        """
        Tests that run method raises ValueError if query is empty
        """
        ranker = DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", similarity=similarity)  # type: ignore
        ranker.warm_up()
        query = ""
        documents = [Document(content="doc1"), Document(content="doc2")]

        with pytest.raises(ValueError):
            ranker.run(query=query, documents=documents)

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_returns_empty_list_if_no_documents_are_provided(self, similarity: str):
        """
        Tests that run method returns empty list if documents is empty
        """
        ranker = DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", similarity=similarity)  # type: ignore
        ranker.warm_up()
        query = "test query"
        documents: List[Document] = []
        results = ranker.run(query=query, documents=documents)

        assert len(results["documents"]) == 0

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_real_world_use_case(self, similarity: str):
        ranker = DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", similarity=similarity)  # type: ignore
        ranker.warm_up()
        query = "What are the reasons for long-standing animosities between Russia and Poland?"

        doc1 = Document(
            "One of the earliest known events in Russian-Polish history dates back to 981, when the Grand Prince of Kiev , "
            "Vladimir Svyatoslavich , seized the Cherven Cities from the Duchy of Poland . The relationship between two by "
            "that time was mostly close and cordial, as there had been no serious wars between both. In 966, Poland "
            "accepted Christianity from Rome while Kievan Rus' —the ancestor of Russia, Ukraine and Belarus—was "
            "Christianized by Constantinople. In 1054, the internal Christian divide formally split the Church into "
            "the Catholic and Orthodox branches separating the Poles from the Eastern Slavs."
        )

        doc2 = Document(
            "Since the fall of the Soviet Union , with Lithuania , Ukraine and Belarus regaining independence, the "
            "Polish Russian border has mostly been replaced by borders with the respective countries, but there still "
            "is a 210 km long border between Poland and the Kaliningrad Oblast"
        )

        doc3 = Document(
            "As part of Poland's plans to become fully energy independent from Russia within the next years, Piotr "
            "Wozniak, president of state-controlled oil and gas company PGNiG , stated in February 2019: 'The strategy of "
            "the company is just to forget about Eastern suppliers and especially about Gazprom .'[53] In 2020, the "
            "Stockholm Arbitrary Tribunal ruled that PGNiG's long-term contract gas price with Gazprom linked to oil prices "
            "should be changed to approximate the Western European gas market price, backdated to 1 November 2014 when "
            "PGNiG requested a price review under the contract. Gazprom had to refund about $1.5 billion to PGNiG."
        )

        doc4 = Document(
            "Both Poland and Russia had accused each other for their historical revisionism . Russia has repeatedly "
            "accused Poland for not honoring Soviet Red Army soldiers fallen in World War II for Poland, notably in "
            "2017, in which Poland was thought on 'attempting to impose its own version of history' after Moscow was "
            "not allowed to join an international effort to renovate a World War II museum at Sobibór , site of a "
            "notorious Sobibor extermination camp."
        )

        doc5 = Document(
            "President of Russia Vladimir Putin and Prime Minister of Poland Leszek Miller in 2002 Modern Polish Russian "
            "relations begin with the fall of communism in1989 in Poland ( Solidarity and the Polish Round Table "
            "Agreement ) and 1991 in Russia ( dissolution of the Soviet Union ). With a new democratic government after "
            "the 1989 elections , Poland regained full sovereignty, [2] and what was the Soviet Union, became 15 newly "
            "independent states , including the Russian Federation . Relations between modern Poland and Russia suffer "
            "from constant ups and downs."
        )

        doc6 = Document(
            "Soviet influence in Poland finally ended with the Round Table Agreement of 1989 guaranteeing free elections "
            "in Poland, the Revolutions of 1989 against Soviet-sponsored Communist governments in the Eastern Block , and "
            "finally the formal dissolution of the Warsaw Pact."
        )

        doc7 = Document(
            "Dmitry Medvedev and then Polish Prime Minister Donald Tusk , 6 December 2010 BBC News reported that one of "
            "the main effects of the 2010 Polish Air Force Tu-154 crash would be the impact it has on Russian-Polish "
            "relations. [38] It was thought if the inquiry into the crash were not transparent, it would increase "
            "suspicions toward Russia in Poland."
        )

        doc8 = Document(
            "Soviet control over the Polish People's Republic lessened after Stalin's death and Gomulka's Thaw , and "
            "ceased completely after the fall of the communist government in Poland in late 1989, although the "
            "Soviet-Russian Northern Group of Forces did not leave Polish soil until 1993. The continuing Soviet military "
            "presence allowed the Soviet Union to heavily influence Polish politics."
        )

        documents = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8]
        result = ranker.run(query=query, documents=documents)
        expected_order = [doc5, doc7, doc3, doc1, doc4, doc2, doc6, doc8]
        expected_content = " ".join([doc.content or "" for doc in expected_order])
        result_content = " ".join([doc.content or "" for doc in result["documents"]])

        # Check the order of ranked documents by comparing the content of the ranked documents
        # This is done by concatenating the content from each document and comparing it to concatenated expected content
        assert result_content == expected_content
