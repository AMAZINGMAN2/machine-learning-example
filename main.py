import h2o
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from h2o.automl import H2OAutoML
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import gc
h2o.init()
print("importing data")
df = h2o.import_file('train.csv')
df = df.drop('id')
y = 'class'
X = df.columns
train, test = df.split_frame(ratios=[0.8], seed=42)

aml = H2OAutoML(max_models=25, seed=42, max_runtime_secs=1000, balance_classes=True, 
                exclude_algos=["GLM", "DeepLearning"], 
                stopping_tolerance=0.05, stopping_metric="MSE", nfolds=5)
print("training model")
aml.train(x=X, y=y, training_frame=train)

model_path = h2o.save_model(model=aml.leader, path="/models/", force=True)

leaderboard = aml.leaderboard
print(leaderboard)
