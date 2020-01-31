import os
import tarfile
from six.moves import urllib

# Downloading Data
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
  if not os.path.isdir(housing_path):
    os.makedirs(housing_path)
  tgz_path = os.path.join(housing_path, "housing.tgz")
  urllib.request.urlretrieve(housing_url, tgz_path)
  housing_tgz = tarfile.open(tgz_path)
  housing_tgz.extractall(path=housing_path)
  housing_tgz.close()

fetch_housing_data()


# Visualizing The Data
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
  csv_path = os.path.join(housing_path, "housing.csv")
  return pd.read_csv(csv_path)
housing = load_housing_data()
housing.head()


# Displaying Information
housing.info()


# Counting Values
housing["ocean_proximity"].value_counts()


# Showing Summary
housing.describe()


# Plotting Histograms
'exec(%matplotlib inline)'    # Invalid syntax: %matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# Creating Test set
import numpy as np
def split_train_test(data, test_ratio):
  shuffled_indices = np.random.permutation(len(data))
  test_set_size = int(len(data) * test_ratio)
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")


# Using md5 function
import hashlib
count = 0
for i in range (0, 256):
  lastByte = hashlib.md5(np.int64(i)).digest()[-1]
  if lastByte < int(256*0.2):
    count += 1
    #print (lastByte)
print ('Count', count)


# Using np.ceil function
for i in [1.27, 1.45, 1.58, 1.93, 2.11, 2.99]:
  print (i, ' ~= ', np.ceil(i))


# Creating A Test Set Using Id
import hashlib
def test_set_check(identifier, test_ratio, hash):
  # Put the instance in the test set if this value is lower or equal to 51 (~20% of 256).
  return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
  ids = data[id_column]
  in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
  return data.loc[~in_test_set], data.loc[in_test_set]
housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
print(len(train_set), "train +", len(test_set), "test")


# Using train_test_split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(len(train_set), "train +", len(test_set), "test")


# Important Attribute: 'median income'
housing["median_income"].value_counts()


# Plotting median_income Histogram
'exec(%matplotlib inline)' #%matplotlib inline
import matplotlib.pyplot as plt
housing["median_income"].hist(bins=50, figsize=(20,15))
plt.show()


# Creating new attribute (income_cat)
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
housing["income_cat"].value_counts()


# Verifying The Income Category Proportions
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print ('Training set:')
print (train_set["income_cat"].value_counts() / len(train_set))
print ('Test set:')
print (test_set["income_cat"].value_counts() / len(test_set))


# Using StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]


# Verifying The Income Category Proportions
print ('Full dataset:')
print (housing["income_cat"].value_counts() / len(housing))
print ('Training set:')
print (strat_train_set["income_cat"].value_counts() / len(strat_train_set))
print ('Test set:')
print (strat_test_set["income_cat"].value_counts() / len(strat_test_set))


# Reverting The Data To Original State: remove "income_cat"
for set in (strat_train_set, strat_test_set):
  if 'income_cat' in set.columns:
    set.drop(["income_cat"], axis=1, inplace=True)
# create a copy so you can play with it without
# harming the training set:
housing = strat_train_set.copy()


# Visualizing Geographical Data
housing.plot(kind="scatter", x="longitude", y="latitude")


# Showing High Density Of Data Points
housing.plot(kind="scatter", x="longitude", y="latitude")


# Showing population & median_house_value
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=housing["population"]/100, label="population",
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
# The radius of each circle represents the district’s population (option s), and
# the color represents the price (option c).
plt.legend()


# Looking for Correlations
# Compute the standard correlation coefficient (also called Pearson’s r)
# between every pair of attributes
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# Plotting Correlation
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# Experimenting with Attribute Combinations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# Preparing Data
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing.describe()


# Using SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
#print (imputer.statistics_)
print ('Median values:', housing_num.median().values)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
housing_tr.describe()


# Converting Text Labels To Numbers
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print ('Original text labels:', encoder.classes_)
print ('New values:', housing_cat_encoded)


# Using OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print (housing_cat_1hot.shape)
print ('One-hot values:')
print (housing_cat_1hot.toarray())


# Using LabelBinarizer
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
print (housing_cat_1hot.shape)
print ('One-hot values:')
print (housing_cat_1hot)


# Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
  def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
    self.add_bedrooms_per_room = add_bedrooms_per_room

  def fit(self, X, y=None):
    return self # nothing else to do

  def transform(self, X, y=None):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if self.add_bedrooms_per_room:
      bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
      return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
    else:
      return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
housing_extra_attribs


# Implementing Min-Max Scaling
import numpy as np
x1 = [89, 72, 94, 69]
x1_min = np.min(x1)
print ('x1 min:', x1_min)
x1_max = np.max(x1)
print ('x1 max:', x1_max)
x1_scaled = (x1 - x1_min)/(x1_max - x1_min)
print ('x1_scaled:', x1_scaled)


# Using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
x1 = [[89], [72], [94], [69]]
minMaxScaler = MinMaxScaler()
x1_scaled = minMaxScaler.fit_transform(x1)
#value(i) = (value(i) - min)/(max - min)
print ('x1_scaled:')
print (x1_scaled)
x2 = [[89, 0.8], [72, 0.12], [94, 1], [69, 0]]
x2_scaled = minMaxScaler.fit_transform(x2)
print ('x2_scaled:')
print (x2_scaled)


# Standardization
import numpy as np
x1 = [89, 72, 94, 69]
x1_mu = np.sum(x1)/len(x1)
print ('x1_mu:', x1_mu)
x1_std = np.sqrt(np.sum((x1 - x1_mu)**2)/4)
print ('x1_std:', x1_std)
x1_normalized = (x1 - x1_mu)/x1_std
print ('x1_normalized:', x1_normalized)


# Using StandardScaler
from sklearn.preprocessing import StandardScaler
x1 = [[89], [72], [94], [69]]
standardScaler = StandardScaler()
x1_normalized = standardScaler.fit_transform(x1)
#value(i) = (value(i) - mean)/(variance) ### unit variance
print ('x1_normalized:')
print (x1_normalized)
x2 = [[89, 0.8], [72, 0.12], [94, 1], [69, 0]]
x2_normalized = standardScaler.fit_transform(x2)
print ('x2_normalized:')
print (x2_normalized)


# Transformation Pipelines (I)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
print(housing_num.iloc[[0]])
num_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy="median")),
  ('attribs_adder', CombinedAttributesAdder()),
  ('std_scaler', StandardScaler()),
  ])
housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr [0]


# Transformation Pipelines (II)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
class DataFrameSelector(BaseEstimator, TransformerMixin):
  def __init__(self, attribute_names):
    self.attribute_names = attribute_names

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return X[self.attribute_names].values

class LabelBinarizerPipelineFriendly(LabelBinarizer):
  def fit(self, X, y=None):
    """This change would allow us to fit the model based on the X input."""
    super(LabelBinarizerPipelineFriendly, self).fit(X)

  def transform(self, X, y=None):
    return super(LabelBinarizerPipelineFriendly, self).transform(X)

  def fit_transform(self, X, y=None):
    return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)


# Transformation Pipelines (III)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
# housing_num = housing.drop("ocean_proximity", axis=1)
# print (list(housing_num))
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
  ('selector', DataFrameSelector(num_attribs)),
  ('imputer', SimpleImputer(strategy="median")),
  ('attribs_adder', CombinedAttributesAdder()),
  ('std_scaler', StandardScaler()),
  ])
cat_pipeline = Pipeline([
  ('selector', DataFrameSelector(cat_attribs)),
  ('label_binarizer', LabelBinarizerPipelineFriendly()),
  ])
full_pipeline = FeatureUnion(transformer_list=[
  ("num_pipeline", num_pipeline),
  ("cat_pipeline", cat_pipeline),
  ])


# Transformation Pipelines (IV)
housing_prepared = full_pipeline.fit_transform(housing)
print ('Shape:', housing_prepared.shape)
print ('First row:')
print (housing_prepared[0])


# Training A LinearRegression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
print (lin_reg.coef_)
print (lin_reg.intercept_)


# Evaluating On The Training Set
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,
housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print ('mean_squared_error:',lin_rmse)


# Training A DecisionTreeRegressor Model
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels,
housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print ('mean_squared_error:',tree_rmse)
#Wait, what!? No error at all?


# Implementing Cross-Validation for DecisionTreeRegressor Model
from sklearn.model_selection import cross_val_score
#K-fold cross-validation: it randomly splits the training set into 10 distinct
#subsets called folds, then it trains and evaluates the Decision Tree model 10 times,
#picking a different fold for evaluation every time and training on the other 9 folds.
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
#Scoring function is actually the opposite of the MSE
rmse_scores = np.sqrt(-scores)
print ('Scores:', rmse_scores,
      '\nMean:', rmse_scores.mean(),
      '\nStandard deviation:', rmse_scores.std())


# Implementing Cross-Validation for LinearRegression Model
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print ('Scores:', lin_rmse_scores,
  '\nMean:', lin_rmse_scores.mean(),
  '\nStandard deviation:', lin_rmse_scores.std())


# Training A RandomForestRegressor Model
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators = 10)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)
forest_scores = cross_val_score(forest_reg, housing_prepared,
housing_labels,
  scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print ('Scores:', forest_rmse_scores,
  '\nMean:', forest_rmse_scores.mean(),
  '\nStandard deviation:', forest_rmse_scores.std())


# Grid Search (I)
from sklearn.model_selection import GridSearchCV
param_grid = [
  {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
  {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

# Grid Search (II)
GridSearchCV(cv=5, error_score='raise-deprecating',
  estimator=RandomForestRegressor(bootstrap=True,
    criterion='mse', max_depth=None,
    max_features='auto', max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_impurity_split=None,
    min_samples_leaf=1, min_samples_split=2,
    min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
    oob_score=False, random_state=None, verbose=0,
    warm_start=False),
  # fit_params=None,
  iid='warn', n_jobs=None,
  param_grid=[{'n_estimators': [3, 10, 30], 'max_features':
    [2, 4, 6, 8]}, {'bootstrap': [False], 'n_estimators': [3, 10],
    'max_features': [2, 3, 4]}],
  pre_dispatch='2*n_jobs', refit=True,
  return_train_score='warn',
  scoring='neg_mean_squared_error', verbose=0)

# Best Params and Estimators
print ('Best params:', grid_search.best_params_)
print ('Best estimator:', grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
  print('Score:' + str(np.sqrt(-mean_score)) + '. Params:' + str(params))

### Fix for below
encoder = OneHotEncoder(categories='auto')
encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
# Analyzing The Best Mode
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.categories_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# Evaluating The Best Model On The Test Set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print ('RMSE:', final_rmse)