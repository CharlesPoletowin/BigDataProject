import pandas as pd
from openclean.function.value.null import is_empty
from openclean.operator.transform.update import update

def fillEmptyColumnListWithValue(ds: pd.core.frame.DataFrame, columnList: list, newValue: str,optionForgetValue: str = "") -> pd.core.frame.DataFrame:
  for i in columnList:
    ds = update(ds, 
                i, 
                lambda x: newValue if is_empty(x.strip()) or x == optionForgetValue else x.strip().upper())
  return ds