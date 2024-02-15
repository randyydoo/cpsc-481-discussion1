import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import ipywidgets as widgets


gdp_raw = pd.read_csv("data/gdp.csv")
hls_raw = pd.read_csv("data/hls.csv")

life_cols = pd.DataFrame(hls_raw, columns =["Country", "Indicator", "Type of indicator","Time","Value"])
hls_life = life_cols.loc[hls_raw["Indicator"] == "Life satisfaction"]
hls_2018 = hls_life.loc[hls_life["Time"] == 2018]
hls_train = hls_2018.loc[hls_2018["Type of indicator"] == "Average"]

gdp_cols  = pd.DataFrame(gdp_raw, columns =["Country","Subject Descriptor", "Subject Notes", "2018"])
gdp_units = gdp_cols.loc[gdp_raw["Subject Notes"] == "Annual percentages of average consumer prices are year-on-year changes."]
gdp_train = gdp_units.loc[gdp_units["Subject Descriptor"] == "Inflation, average consumer prices"]


merged_train_data = pd.merge(hls_train, gdp_train, on="Country")
merged_train_data = merged_train_data.rename(columns={"Value": "Life Satisfaction", "2018": "Inflation Measurement"})
merged_train_data = pd.DataFrame(merged_train_data, columns=['Country','Life Satisfaction', 'Inflation Measurement'])

X = np.c_[merged_train_data["Inflation Measurement"]]
Y = np.c_[merged_train_data["Life Satisfaction"]]
x = X.tolist()
y = Y.tolist()

# plot data
out1 = widgets.Output()
with out1:
  plt.scatter(x, y)
  plt.xlabel('Inflation')
  plt.ylabel('Happiness')
  plt.title("Data Plot")
  plt.show()

# fit linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, Y)

# plot predictions
predict_x = [x for x in range(901)]
predict_x = [[x/100] for x in predict_x]
predict_y = model.predict(predict_x)

out2 = widgets.Output()
with out2:
  plt.scatter(predict_x, predict_y)
  plt.scatter(x, y)
  plt.xlabel('Inflation')
  plt.ylabel('Happiness')
  plt.title("Prediction Line")
  plt.show()

display(widgets.HBox([out1,out2]))
