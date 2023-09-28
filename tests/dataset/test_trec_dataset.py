import unittest
from gurank.dataset.trec_dataset import TrecDataset, QueryTrecDataset, DataCollator, QueryDataCollator
from transformers import AutoTokenizer

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

class DataCollatorTest(unittest.TestCase):
    def setUp(self):
        self.features =[
            {"idx": 0,
            "query": "What's the weather today?",
            "doc": "It's raining now.",
            "label": 1
            },
            {"idx": 1,
            "query": "What's the weather today?",
            "doc": "You really should read this interesting book.",
            "label": 0
            },
            {"idx": 2,
            "query": "Will future be good?",
            "doc": "Tomorrow will be better.",
            "label": 1
            },
            {"idx": 3,
            "query": "Will future be good?",
            "doc": "AI is rapidly changing the world.",
            "label": 0
            },]
    def test_bert_collator(self):
        model_path="tests/test_data/models/tiny-bert"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        collator = DataCollator(tokenizer=tokenizer)
        idxs, batch, labels = collator(self.features)
        self.assertEqual(idxs.tolist(), [0, 1, 2, 3])
        self.assertEqual(labels.tolist(), [1, 0, 1, 0])
        self.assertEqual(batch["input_ids"].tolist(), [
            [101, 2054, 1005, 1055, 1996, 4633, 2651, 1029, 102, 2009, 1005, 1055, 24057, 2085, 1012, 102, 0, 0],
            [101, 2054, 1005, 1055, 1996, 4633, 2651, 1029, 102, 2017, 2428, 2323, 3191, 2023, 5875, 2338, 1012, 102],
            [101, 2097, 2925, 2022, 2204, 1029, 102, 4826, 2097, 2022, 2488, 1012, 102, 0, 0, 0, 0, 0],
            [101, 2097, 2925, 2022, 2204, 1029, 102, 9932, 2003, 5901, 5278, 1996, 2088, 1012, 102, 0, 0, 0]])

    def test_t5_collator(self):
        model_path="tests/test_data/models/tiny-t5"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        prompt = "Query: {} Document: {} Relevant:"
        label_text = ("false", "true")
        collator = DataCollator(tokenizer=tokenizer, prompt=prompt, label_text=label_text)
        idxs, batch, labels = collator(self.features)
        self.assertEqual(idxs.tolist(), [0, 1, 2, 3])
        self.assertEqual(labels.tolist(), [[1176, 1], [6136, 1], [1176, 1], [6136, 1]])
        self.assertEqual(batch["input_ids"].tolist(), [
            [3, 27569, 10, 363, 31, 7, 8, 1969, 469, 58, 11167, 10, 94, 31, 7, 3412, 53, 230, 5, 31484, 17, 10, 1, 0],
            [3, 27569, 10, 363, 31, 7, 8, 1969, 469, 58, 11167, 10, 148, 310, 225, 608, 48, 1477, 484, 5, 31484, 17, 10, 1],
            [3, 27569, 10, 2003, 647, 36, 207, 58, 11167, 10, 22365, 56, 36, 394, 5, 31484, 17, 10, 1, 0, 0, 0, 0, 0],
            [3, 27569, 10, 2003, 647, 36, 207, 58, 11167, 10, 7833, 19, 7313, 2839, 8, 296, 5, 31484, 17, 10, 1, 0, 0, 0]])

class QueryDataCollatorTest(unittest.TestCase):
    def setUp(self):
        self.features =[
            {"query": "What's the weather today?",
            "docs": ["It's raining now.", "You really should read this interesting book."],
            "labels": [1, 0]
            },
            {"query": "Will future be good?",
            "docs": ["Tomorrow will be better.", "AI is rapidly changing the world."],
            "labels": [1, 0]
            }]
    def test_bert_collator(self):
        model_path="tests/test_data/models/tiny-bert"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        collator = QueryDataCollator(tokenizer=tokenizer, docs_per_query=2)
        batch, labels = collator(self.features)
        self.assertEqual(labels.tolist(), [1, 0, 1, 0])
        self.assertEqual(batch["input_ids"].tolist(), [
            [101, 2054, 1005, 1055, 1996, 4633, 2651, 1029, 102, 2009, 1005, 1055, 24057, 2085, 1012, 102, 0, 0],
            [101, 2054, 1005, 1055, 1996, 4633, 2651, 1029, 102, 2017, 2428, 2323, 3191, 2023, 5875, 2338, 1012, 102],
            [101, 2097, 2925, 2022, 2204, 1029, 102, 4826, 2097, 2022, 2488, 1012, 102, 0, 0, 0, 0, 0],
            [101, 2097, 2925, 2022, 2204, 1029, 102, 9932, 2003, 5901, 5278, 1996, 2088, 1012, 102, 0, 0, 0]])

    def test_t5_collator(self):
        model_path="tests/test_data/models/tiny-t5"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        prompt = "Query: {} Document: {} Relevant:"
        label_text = ("false", "true")
        collator = QueryDataCollator(tokenizer=tokenizer, prompt=prompt, label_text=label_text, docs_per_query=2)
        batch, labels = collator(self.features)
        self.assertEqual(labels.tolist(), [[1176, 1], [6136, 1], [1176, 1], [6136, 1]])
        self.assertEqual(batch["input_ids"].tolist(), [
            [3, 27569, 10, 363, 31, 7, 8, 1969, 469, 58, 11167, 10, 94, 31, 7, 3412, 53, 230, 5, 31484, 17, 10, 1, 0],
            [3, 27569, 10, 363, 31, 7, 8, 1969, 469, 58, 11167, 10, 148, 310, 225, 608, 48, 1477, 484, 5, 31484, 17, 10, 1],
            [3, 27569, 10, 2003, 647, 36, 207, 58, 11167, 10, 22365, 56, 36, 394, 5, 31484, 17, 10, 1, 0, 0, 0, 0, 0],
            [3, 27569, 10, 2003, 647, 36, 207, 58, 11167, 10, 7833, 19, 7313, 2839, 8, 296, 5, 31484, 17, 10, 1, 0, 0, 0]])

if __name__ == "__main__":
    unittest.main()