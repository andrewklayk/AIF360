import pandas as pd


class RankingDetector:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.uniques = {attr: data[attr].unique().tolist() for attr in data.columns}
        
    def _make_children(self, pattern: dict):
        # generate the children of pattern
        attrs = self.data.columns.tolist()
        if len(pattern) == 0:
            # start with all key-value pairs for individual attributes (columns)
            return [{attr: val} for attr in attrs for val in self.uniques[attr]]
        else:
            # patterns are dicts with attribute-value pairs
            idx_pat = max([attrs.index(attr) for attr in pattern.keys()])
            # add next attributes
            ch = []
            for i in range(idx_pat+1, len(attrs)):
                attr = attrs[i]
                for val in self.uniques[attr]:
                    child = pattern.copy()
                    child[attr] = val
                    ch.append(child)
            return ch
        

def topdown(
        data: pd.DataFrame,
        ranking: pd.DataFrame,
        min_group_size: int,
        k: int,
        bound_k: int
        ):
    pass
