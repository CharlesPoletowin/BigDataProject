#!/usr/local/bin/python

from openclean.cluster.knn import knn_clusters
from openclean.function.similarity.base import SimilarityConstraint
from openclean.function.similarity.text import LevenshteinDistance
from openclean.function.value.threshold import GreaterThan
from openclean.function.eval.domain import Lookup
from openclean.operator.transform.update import update
import pandas

def cleanSimilarStringEvaluate(ds: pandas.core.frame.DataFrame, colName: str) -> dict:
  listNeedToChange = ds[colName].unique().tolist()
  # edit distance cluster
  clusters = knn_clusters(values=listNeedToChange,
      sim=SimilarityConstraint(func=LevenshteinDistance(),
                               pred=GreaterThan(0.8))
      )
  mapping = {}
  for cluster in clusters: 
    mapping.update(cluster.to_mapping())
  return mapping

def updateSimilarStringWithMapping(ds: pandas.core.frame.DataFrame, colName: str, mapping: dict):
  return update(
      ds, 
      colName, 
      Lookup(columns=[colName], mapping=mapping, default=colName)
      )