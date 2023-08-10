import pandas as pd


class RankingDetector:
    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data
        self._uniques = {attr: data[attr].unique().tolist() for attr in data.columns}
        
    def _make_children(self, pattern: dict) -> list[dict]:
        # generate the children of pattern
        attrs = self._data.columns.tolist()
        if len(pattern) == 0:
            # start with all key-value pairs for individual attributes (columns)
            return [{attr: val} for attr in attrs for val in self._uniques[attr]]
        else:
            # patterns are dicts with attribute-value pairs
            idx_pat = max([attrs.index(attr) for attr in pattern.keys()])
            # add next attributes
            ch = []
            for i in range(idx_pat+1, len(attrs)):
                attr = attrs[i]
                for val in self._uniques[attr]:
                    child = pattern.copy()
                    child[attr] = val
                    ch.append(child)
            return ch
    
    def topdown(
            self,
            ranking: pd.DataFrame,
            min_group_size: int,
            k: int,
            bound_k: int
            ):
        
        res = []
        nodes = self._make_children({})
          
        def _filter_pattern(dataset: pd.DataFrame, pattern: dict):
                return dataset.query(
                    " & ".join([f'{attr} == {repr(val) if isinstance(val, str) else val}' for attr, val in pattern.items()]))
        
        def _is_ancestor(ancestor: dict, child: dict):
            # checks if an p1 is an ancestor of p2 
            return all(it in child.items() for it in ancestor.items())
        
        while len(nodes) > 0:
            pat = nodes.pop()
            # maybe try groupby????
            pat_items = _filter_pattern(dataset=self._data, pattern=pat)
            pat_size = pat_items.shape[0]
            if pat_size >= min_group_size:
                in_rank = _filter_pattern(dataset=ranking.iloc[:k],pattern=pat)
                if in_rank.shape[0] < bound_k:
                    if not any([_is_ancestor(pat, other) for other in res]):
                        res.append(pat)
                else:
                    nodes.extend(self._make_children(pat))
        return res
