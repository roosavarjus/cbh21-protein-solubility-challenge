from feature_construction import compute_features
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import glob
import csv
import pickle


# train the random forrest with training data

filenames = glob.glob("data/training/crystal_structs/*.pdb")

features = compute_features(filenames)
features = features.sort_values('protIDs').reset_index(drop=True)

solubility = pd.read_csv("data/training/crystal_structs/solubility_values.csv")
solubility = solubility.sort_values('protein').reset_index(drop=True)

# save the proteins 
proteins = solubility['protein'].values

X = features.iloc[:, 1:].to_numpy()
y = solubility.iloc[:, 1:].to_numpy()

rfr = RandomForestRegressor(n_estimators=1000)
rfr.fit(X, np.ravel(y))

# save model
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(rfr, file)
