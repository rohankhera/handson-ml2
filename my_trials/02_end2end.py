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
# %% {Focusing on scikit learn sampling}
# %%
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# %%
train_set.info()
# %%
test_set.info()
# %%
housing["median_income"].hist()
# %%
import numpy as np 
housing["income_cat"] = pd.cut(housing["median_income"],
                             bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# %%
housing["income_cat"].hist()
# %%
housing["income_cat"].value_counts()
# %%
#Stratified sampling on income_cat variable
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# %%
# proportion values
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
# %%
#Plain numbers
strat_test_set["income_cat"].value_counts()
# %%
# percentage values
100*(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# %%
#percentage value sorted by index
100*(strat_test_set["income_cat"].value_counts().sort_index() / len(strat_test_set))
# %%
#Training set - percentage value sorted by index
100*(strat_train_set["income_cat"].value_counts().sort_index() / len(strat_train_set))
# %%
#dropping variable income_cat as only needed for stratified sampling
for dfs in (strat_test_set, strat_train_set):
    dfs.drop("income_cat", axis=1, inplace=True)
# %%
for dfs in (strat_test_set, strat_train_set):
    dfs.info()
# %%
