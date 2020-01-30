import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
# Load the data
# oecd_bli = pd.read_csv("./datasets/oecd_bli_2015.csv", thousands=',')
# gdp_per_capita = pd.read_csv("./datasets/gdp_per_capita.csv",thousands=',' ,delimiter='\t', encoding='latin1', na_values="n/a")
oecd_bli = pd.read_csv("D:/MLPython/pythons/datasets/oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("D:/MLPython/pythons/datasets/gdp_per_capita.csv",thousands=',' ,delimiter='\t', encoding='latin1', na_values="n/a")

# Prepare the data
def prepare_country_stats(oecd_bli, gdp_per_capita):
  oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
  oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
  gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
  gdp_per_capita.set_index("Country", inplace=True)
  full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
  full_country_stats.sort_values(by="GDP per capita", inplace=True)
  remove_indices = [0, 1, 6, 8, 33, 34, 35]
  keep_indices = list(set(range(36)) - set(remove_indices))
  return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

country_stats

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Preparing Training Data
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]
print ('X shape:', X.shape)
print ('y shape:', y.shape)
print('X[0]:', X[0])
print('y[0]:', y[0])

from sklearn.linear_model import LinearRegression
# Select a linear model
lin_reg_model = LinearRegression()
# Train the model
lin_reg_model.fit(X, y)
print ('coef:', lin_reg_model.coef_)
print ('intercept:', lin_reg_model.intercept_)

# Making Prediction
X_new = [[22587]] # Cyprus' GDP per capita
print(lin_reg_model.predict(X_new)) # outputs [[ 5.96242338]]
