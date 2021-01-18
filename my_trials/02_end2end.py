# %%
import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
# %%
fetch_housing_data()
# %%
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
# %%
housing = load_housing_data()
housing.head()
# %%
housing.info()
# %%
housing["ocean_proximity"].value_counts()
# %%
housing.describe()

# %%
#modified for an interactive python script
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
# %%
housing.hist(bins=50, figsize=(10,10))
# %%
