from torch.utils.data import Dataset
from torch import Tensor
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass
import ir_datasets
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict, List, Any

def create_sample(query: str, doc: str, prompt: Optional[str]) -> Union[str, Tuple]:
    """Create sample by prompt"""
    if prompt:
        return prompt.format(query, doc)
    
    return [query, doc]

class TrecDatasetBase(Dataset, ABC):
    """
    DatasetBase of dataset on Trec Format
    """

    def __init__(
        self,
        name: Optional[str] = None,
        doc_path: Optional[str] = None,
        query_path: Optional[str] = None,
        qrel_path: Optional[str] = None,
    ):
        if name:
            self._dataset = ir_datasets.load(name)
        elif doc_path and query_path and qrel_path:
            # Trec format file with no header
            self._dataset = ir_datasets.create_dataset(
                doc_path, query_path, qrel_path
            )
        else:
            raise ValueError("Either name or doc/query/qrel path not exist.")
        
        if name and qrel_path:
            tmp_dataset = ir_datasets.create_dataset(qrels_trec=qrel_path)
            self._qrels = tmp_dataset.qrels
        else:
            self._qrels = self._dataset.qrels

        self._qrels_dict = self._qrels.asdict()
        
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pass

    @property
    def qrels_dict(self):
        """qrels as dict"""
        return self._qrels_dict


class TrecDataset(TrecDatasetBase):
    """
    Dataset of Trec format, every record has a query and a doc.
    """

    def __init__(self,
        name: Optional[str] = None,
        doc_path: Optional[str] = None,
        query_path: Optional[str] = None,
        qrel_path: Optional[str] = None,
    ):
        super().__init__(name, doc_path, query_path, qrel_path)
        self._ids = dict(enumerate(self._qrels))

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, int]]:
        qrel = self._ids[idx]
        rel = qrel.relevance
        qid = qrel.query_id
        did = qrel.doc_id
        query = self._dataset.queries.lookup(qid).text
        doc = self._dataset.docs.lookup(did).text

        return {"idx": idx, "query": query, "doc": doc, "label": rel}

    @property
    def ids_mapping(self) -> List[Tuple[int, str, str]]:
        """Get mapping query_id and doc_id of idx"""
        return [(k, v.query_id, v.doc_id) for k, v in self._ids.items()]


class QueryTrecDataset(TrecDatasetBase):
    """
    Dataset of Trec format, every record has a query and all its docs.
    """
    def __init__(self,
        name: Optional[str] = None,
        doc_path: Optional[str] = None,
        query_path: Optional[str] = None,
        qrel_path: Optional[str] = None,
    ):
        super().__init__(name, doc_path, query_path, qrel_path)
        self._ids = dict(enumerate(self._qrels_dict.keys()))

    def __len__(self) -> int:
        return len(self._ids)
    
    def __getitem__(self, idx:int) -> Dict[str, Union[str, List[Union[str,int]]]]:
        qid = self._ids[idx]
        query = self._dataset.queries.lookup(qid).text
        rels = self._qrels_dict[qid]
        docs = []
        labels = []
        for did, label in rels.items():
            docs.append(self._dataset.docs.lookup(did).text)
            labels.append(label)
        
        return {"query": query, "docs": docs, "labels": labels}


@dataclass
class DataCollator:
    """
    DataCollator for TrecDataset
    """
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    padding: Optional[bool] = True
    truncation: Optional[Union[str, bool]] = "only_second"
    prompt: Optional[str] = None
    label_text: Optional[Tuple[str]] = None

    def __call__(self, features: List[Dict[str, Union[str, int]]]) -> Dict[str, Tensor]:
        idxs = torch.LongTensor([f["idx"] for f in features])
        texts = [create_sample(f["query"], f["doc"], self.prompt) for f in features]
        rels = [self.label_text[f["label"]] if self.label_text else f["label"] for f in features]
        batch = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = self.tokenizer(rels, return_tensors="pt").input_ids if self.label_text else torch.LongTensor(rels)
        return idxs, batch, labels

@dataclass
class QueryDataCollator:
    """
    DataCollator for QueryDataset
    """
    tokenizer: PreTrainedTokenizerBase
    docs_per_query: int
    max_length: Optional[int] = None
    padding: Optional[bool] = True
    truncation: Optional[Union[str, bool]] = "only_second"
    prompt: Optional[str] = None
    label_text: Optional[Tuple[str]] = None

    def __call__(self, features: List[Dict[str, Union[str, List[Union[str,int]]]]]) -> Dict[str, Tensor]:
        texts = []
        rels = []
        for qds in features:
            if not (len(qds["docs"] == qds("labels") == docs_per_query)):
                raise ValueError(f"The docs or labels num under query {qds['query']} is not equal to docs_per_query")
            for doc, label in zip(qds["docs"], qds["labels"]):
                texts.append(create_sample(qds["query"], doc, self.prompt))
                rels.append(self.label_text[label] if self.label_text else label)

        batch = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt"
        )

        
        labels = self.tokenizer(rels, return_tensors="pt").input_ids if self.label_text else torch.LongTensor(rels)

        return batch, labels