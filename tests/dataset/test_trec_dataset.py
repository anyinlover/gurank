import unittest
from gurank.dataset.trec_dataset import TrecDataset, QueryTrecDataset

class TrecDatasetTest(unittest.TestCase):
    def setUp(self):
        root_path = "tests/test_data/trec_data"
        doc_path = f"{root_path}/collections.tsv"
        query_path = f"{root_path}/queries.train.tsv"
        qrel_path = f"{root_path}/qrels.train.tsv"
        self._dataset = TrecDataset(doc_path=doc_path, query_path=query_path, qrel_path=qrel_path)
    
    def test_len(self):
        self.assertEqual(len(self._dataset), 8)
    
    def test_getitem(self):
        expect_return = {
            "idx": 0,
            "query": "What's the weather today?",
            "doc": "It's raining now.",
            "label": 1
        }
        self.assertEqual(self._dataset[0], expect_return)

    def test_ids_mapping(self):
        expect_return = [
            (0, "query1", "doc1"),
            (1, "query1", "doc5"),
            (2, "query2", "doc3"),
            (3, "query2", "doc6"),
            (4, "query3", "doc5"),
            (5, "query3", "doc2"),
            (6, "query4", "doc6"),
            (7, "query4", "doc4")
        ]
        self.assertEqual(self._dataset.ids_mapping, expect_return)
    
    def test_qrels_dict(self):
        expect_return = {
            "query1": {"doc1": 1, "doc5": 0},
            "query2": {"doc3": 1, "doc6": 0},
            "query3": {"doc5": 1, "doc2": 0},
            "query4": {"doc6": 1, "doc4": 0}
            }
        self.assertEqual(self._dataset.qrels_dict, expect_return)

class QueryTrecDatasetTest(unittest.TestCase):
    def setUp(self):
        root_path = "tests/test_data/trec_data"
        doc_path = f"{root_path}/collections.tsv"
        query_path = f"{root_path}/queries.train.tsv"
        qrel_path = f"{root_path}/qrels.train.tsv"
        self._dataset = QueryTrecDataset(doc_path=doc_path, query_path=query_path, qrel_path=qrel_path)
    
    def test_len(self):
        self.assertEqual(len(self._dataset), 4)

    def test_getitem(self):
        expect_return = {
            "query": "What's the weather today?",
            "docs": ["It's raining now.", "You really should read this interesting book."],
            "labels": [1, 0]
        }
        self.assertEqual(self._dataset[0], expect_return)

    def test_qrels_dict(self):
        expect_return = {
            "query1": {"doc1": 1, "doc5": 0},
            "query2": {"doc3": 1, "doc6": 0},
            "query3": {"doc5": 1, "doc2": 0},
            "query4": {"doc6": 1, "doc4": 0}
            }
        self.assertEqual(self._dataset.qrels_dict, expect_return)

if __name__ == "__main__":
    unittest.main()