import pickle
from feature_construction import compute_features
import numpy as np
import pandas as pd
import glob
import csv
import freesasa
import Bio.PDB as PDB 
import Bio.PDB.DSSP as DSSP 
import glob
import numpy as np
import pandas as pd
from calc_fractions import *
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as IP
from Bio.PDB.Polypeptide import PPBuilder
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import atomium
import matplotlib.pyplot as plt

with open('pickle_model.pkl', 'rb') as file:
    rfr = pickle.load(file)

filenames = glob.glob("data/training/crystal_structs/*.pdb")
features = compute_features(filenames)
features = features.sort_values('protIDs').reset_index(drop=True)
X = features.iloc[:, 1:]
feature_importances = rfr.feature_importances_
feature_names = X.columns
sorted_idx = rfr.feature_importances_.argsort()

plt.barh(feature_names[sorted_idx], feature_importances[sorted_idx])
plt.title("Feature importance")
plt.tight_layout()
plt.savefig("Feature_importance.png")
plt.show()