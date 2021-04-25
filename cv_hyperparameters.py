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
import pickle


########
# Hyperparameter training for random forrest
########

# Parameters to train 
# - number of trees
# - feature selection https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f
# - number of leave nodes 
# - depth of tree

# RMSE and RMSPE
# Pearson correlation

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

# features = compute_features(filenames)
features = pd.read_csv('features.csv')
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
df_features = df_features.sort_values('Selected', ascending=False)
plt.bar(df_features['Features'], df_features['Selected'])
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('feature_selection.png')
plt.show()



# check feature importance in model 
with open('pickle_model.pkl', 'rb') as file:
    rfr = pickle.load(file)

print(rfr.feature_importances_)
df_features = pd.DataFrame({'Features': features.columns[1:], 'Selected': rfr.feature_importances_})
df_features = df_features.sort_values('Selected', ascending=False)
plt.bar(df_features['Features'], df_features['Selected'])
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()



num_features = [3, 6, 9, 12, 15, 18]

for i in range(len(num_features)):

    indices = df_features[:num_features[i]].index
    XTrain_reduced = XTrain[:, indices]

    RMSE_list = []
    RMSPE_list = []
    times = []

    for train_index, test_index in cv.split(XTrain):
        XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain_reduced[train_index], XTrain_reduced[test_index], YTrain[train_index], YTrain[test_index]
        
        start = timeit.default_timer()

        rfr = RandomForestRegressor(n_estimators=500)
        rfr.fit(XTrainCV, np.ravel(YTrainCV))

        stop = timeit.default_timer()
        times.append(stop - start)

        rfr_prediction = rfr.predict(XTestCV)

        RMSE_list.append(rmse(np.squeeze(YTestCV), rfr_prediction))
        RMSPE_list.append(rmspe(np.squeeze(YTestCV), rfr_prediction))

    print('Features', num_features[i])
    print('Time', np.mean(times))
    print('RMSE', np.mean(RMSE_list))
    print('RMSPE', np.mean(RMSPE_list))



#########################
# Number of trees

num_trees = [50, 100, 500, 1000, 5000]

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

