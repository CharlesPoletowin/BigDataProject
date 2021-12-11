import pandas as pd
from openclean.function.value.null import is_empty
from openclean.operator.transform.update import update

# Given location, return a zipcode.
# Reading a borough-zipcode-location list, then filling the zipcode by location data. 
url = "https://raw.githubusercontent.com/CharlesPoletowin/BigDataProject/main/nyc_zipcodes.csv"
df = pd.read_csv(url, index_col=0)
zipcodes = df.values

def location_to_zip(lat, lng, data):
  res = min(data, key = lambda x: abs(x[1] - lat) + abs(x[2] - lng))
  return str(int(res[0]))

# this is the function which should be used by others
def locationToZip(ds: pd.core.frame.DataFrame, 
                  zipCodeColumn: str, 
                  latitudeColumn: str, 
                  longitudeColumn: str) -> pd.core.frame.DataFrame:
  return update(
    ds, 
    [zipCodeColumn, latitudeColumn, longitudeColumn], 
    lambda a,b,c: (location_to_zip(float(b), float(c), zipcodes), b, c) if (is_empty(a) and not is_empty(b) and not is_empty(c)) else (a, b, c)
    )


# Filling the borough.
df2 = pd.read_csv(url)
df2 = df2[['BOR', 'ZIP']]
boroughs = df2.values

def zip_to_borough(zip, data):
  res = min(data, key = lambda x: abs(x[1] - zip))
  return str(res[0])

# this is the function where others should use outside this file
def zipToBorough(ds: pd.core.frame.DataFrame, 
                 zipCodeColumn: str, 
                 boroughColumn: str) -> pd.core.frame.DataFrame:
  return update(ds, 
                [boroughColumn, zipCodeColumn], 
                lambda a,b: (zip_to_borough(int(b), boroughs), b) if (is_empty(a) and not is_empty(b)) else (a, b))