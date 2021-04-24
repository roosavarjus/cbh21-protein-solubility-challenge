import pickle 
import glob
from feature_construction import compute_features
from matplotlib.pyplot import plt


# 1) make bar plot with feature importances 
# 2) correlation coeffiitnts


# Load from file
with open('pickle_model.pkl', 'rb') as file:
    rfr = pickle.load(file)



# from sklearn import metrics


filenames_train = glob.glob("data/training/crystal_structs/*.pdb")

features_train = compute_features(filenames_train)
features_train = features_train.sort_values('protIDs').reset_index(drop=True)

solubility_train = pd.read_csv("data/training/crystal_structs/solubility_values.csv")
solubility_train = solubility_train.sort_values('protein').reset_index(drop=True)

print(features_train.columns)
print(rfr.feature_importances_)




# filenames_train = glob.glob("data/test/ecoli_modelled_structs/*.pdb")

# features_train = compute_features(filenames)
# features_train = features.sort_values('protIDs').reset_index(drop=True)

# solubility_train = pd.read_csv("data/training/crystal_structs/solubility_values.csv")
# solubility_train = solubility.sort_values('protein').reset_index(drop=True)

