from torch.utils.data import Dataset
import ir_datasets
from abc import ABC, abstractmethod
from typing import Optional

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
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int):
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

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx: int):
        qrel = self._ids[idx]
        rel = qrel.relevance
        qid = qrel.query_id
        did = qrel.doc_id
        query = self._dataset.queries.lookup(qid).text
        doc = self._dataset.docs.lookup(did).text

        return {"idx": idx, "query": query, "doc": doc, "label": rel}

    @property
    def ids_mapping(self):
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

    def __len__(self):
        return len(self._ids)
    
    def __getitem__(self, idx:int):
        qid = self._ids[idx]
        query = self._dataset.queries.lookup(qid).text
        rels = self._qrels_dict[qid]
        docs = []
        labels = []
        for did, label in rels.items():
            docs.append(self._dataset.docs.lookup(did).text)
            labels.append(label)
        
        return {"query": query, "docs": docs, "labels": labels}
