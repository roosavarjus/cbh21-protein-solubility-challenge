import glob 
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from feature_construction import compute_features
import pandas as pd
import numpy as np
import timeit
from sklearn import metrics
import matplotlib.pyplot as plt


########
# Hyperparameter training for random forrest
########

# Parameters to train 
# - number of trees
# - feature selection https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f
# - number of leave nodes 
# - depth of tree

def rmse(f, t):
    N = t.shape[0]
    rmse = np.sqrt(np.mean((t - f) ** 2))
    return np.array(rmse)

def rmspe(f, t):
    N = t.shape[0]
    # import ipdb; ipdb.set_trace()
    rmspe = np.sqrt(np.mean(((t - f) / t) ** 2 ))
    return np.array(rmspe)



# iterate over training files
filenames = glob.glob("data/training/crystal_structs/*.pdb")

features = compute_features(filenames)
features = features.sort_values('protIDs').reset_index(drop=True)

solubility = pd.read_csv("data/training/crystal_structs/solubility_values.csv")
solubility = solubility.sort_values('protein').reset_index(drop=True)

XTrain = features.iloc[:, 1:].to_numpy()
YTrain = solubility.iloc[:, 1:].to_numpy()

k = 5
cv = KFold(n_splits=k, shuffle=True)


#########################
# Feature selection

selected_features = []

repetitions = 3
for i in range(repetitions):
    for train_index, test_index in cv.split(XTrain):
        XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain[train_index], XTrain[test_index], YTrain[train_index], YTrain[test_index]
        
        rfr = RandomForestRegressor(n_estimators=100)
        rfr.fit(XTrainCV, np.ravel(YTrainCV))

        sel = SelectFromModel(rfr, prefit=True)

        selected_features.append(sel.get_support())

print(np.array(selected_features).sum(0))
print(features.columns[1:])

df_features = pd.DataFrame({'Features': features.columns[1:], 'Selected': np.array(selected_features).sum(0)})
df_features = df_features.sort_values('Selected')
plt.bar(df_features['Features'], df_features['Selected'])
# rotate labels
plt.show()



#########################
# Number of trees

num_trees = [50, 100, 500, 1000, 10000]

for i in range(len(num_trees)):

    RMSE_list = []
    RMSPE_list = []
    times = []

    for train_index, test_index in cv.split(XTrain):
        XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain[train_index], XTrain[test_index], YTrain[train_index], YTrain[test_index]
        
        start = timeit.default_timer()

        rfr = RandomForestRegressor(n_estimators=num_trees[i])
        rfr.fit(XTrainCV, np.ravel(YTrainCV))

        stop = timeit.default_timer()
        times.append(stop - start)

        rfr_prediction = rfr.predict(XTestCV)

        RMSE_list.append(rmse(np.squeeze(YTestCV), rfr_prediction))
        RMSPE_list.append(rmspe(np.squeeze(YTestCV), rfr_prediction))

    print('Trees', num_trees[i])
    print('Time', np.mean(times))
    print('RMSE', np.mean(RMSE_list))
    print('RMSPE', np.mean(RMSPE_list))

