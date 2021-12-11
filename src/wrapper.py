import pandas as pd
from openclean.function.value.null import is_empty
from openclean.operator.transform.update import update
from openclean.cluster.knn import knn_collision_clusters
from openclean.function.similarity.base import SimilarityConstraint
from openclean.function.similarity.text import LevenshteinDistance
from openclean.function.value.threshold import GreaterThan
from openclean.function.eval.domain import Lookup


def location_to_zip(lat, lng, data):
    res = min(data, key=lambda x: abs(x[1] - lat) + abs(x[2] - lng))
    return str(int(res[0]))


def zip_to_borough(zipcode, data):
    res = min(data, key=lambda x: abs(x[1] - zipcode))
    return str(res[0])


class Wrapper:
    def __init__(self, path_to_dataset: str):
        self.ds = pd.read_csv(path_to_dataset)

    def clean_string_typos(self, col_target: str):
        # edit distance cluster
        clusters = knn_collision_clusters(values=self.ds[col_target].tolist(),
                                          sim=SimilarityConstraint(func=LevenshteinDistance(), pred=GreaterThan(0.8))
                                          )
        mapping = {}
        for cluster in clusters:
            mapping.update(cluster.to_mapping())

        return update(self.ds,
                      col_target,
                      Lookup(columns=[col_target], mapping=mapping, default=col_target)
                      )

    def clean_zip_from_location(self, col_zipcode: str, col_latitude: str, col_longitude: str):
        url = "https://raw.githubusercontent.com/CharlesPoletowin/BigDataProject/main/data/nyc_zipcodes.csv"
        df = pd.read_csv(url, index_col=0)
        zipcodes = df.values

        return update(self.ds,
                      [col_zipcode, col_latitude, col_longitude],
                      lambda a, b, c: (location_to_zip(float(b), float(c), zipcodes), b, c)
                      if (is_empty(a) and not is_empty(b) and not is_empty(c)) else (a, b, c)
                      )

    def clean_borough_from_zip(self, col_zipcode: str, col_borough: str):
        url = "https://raw.githubusercontent.com/CharlesPoletowin/BigDataProject/main/data/nyc_zipcodes.csv"
        df = pd.read_csv(url)
        df = df[['BOR', 'ZIP']]
        boroughs = df.values

        return update(self.ds,
                      [col_borough, col_zipcode],
                      lambda a, b: (zip_to_borough(int(b), boroughs), b)
                      if (is_empty(a) and not is_empty(b)) else (a, b)
                      )

    def clean_empty_values(self, target_cols: list, new_value: str = "Unknown", empty_value: str = ""):
        for col in target_cols:
            self.ds = update(self.ds,
                             col,
                             lambda x: new_value if is_empty(x.strip()) or x == empty_value else x.strip().upper())
