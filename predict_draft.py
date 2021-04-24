import argparse
from Bio import SeqIO
import pandas as pd
import pickle
import glob
import temppathlib
import zipfile

from feature_construction import compute_features





# set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, default="data/test.zip")
args = parser.parse_args()

predictions = []
# use a temporary directory so we don't pollute our repo
with temppathlib.TemporaryDirectory() as tmpdir:
    # unzip the file with all the test PDBs
    with zipfile.ZipFile(args.infile, "r") as zip_:
        zip_.extractall(tmpdir.path)

    filenames = [str(tmp) for tmp in tmpdir.path.glob("*.pdb")]
    # print(filenames)

    features = compute_features(filenames)

# Load from file
with open('pickle_model.pkl', 'rb') as file:
    rfr = pickle.load(file)


# print(features)

# proteins = features['protIDs'].values
# XTest = features.iloc[:, 1:].to_numpy()

# rfr_prediction = rfr.predict(XTest)

# predictions = pd.DataFrame({'protein': proteins, 'solubility': rfr_prediction})

# # output file, this will be used for benchmarking
# predictions_outfile = "predictions.csv"

# predictions.to_csv(predictions_outfile)


####################################
# garbage

# # set up argument parsing (make sure these match those in config.yml)
# parser = argparse.ArgumentParser()
# parser.add_argument("--infile", type=str, required=True)
# args = parser.parse_args()



# # process input
# fasta_dict = dict()
# with open(args.infile, 'r') as fh:
#     for record in SeqIO.parse(fh, 'fasta'):
#         fasta_dict[record.id] = str(record.seq)

# # save predictions to file
# with open(predictions_outfile, 'w') as fh:
#     fh.write("name,adenine_count\n")
#     for key, value in fasta_dict.items():
#         adenine_prediction = int(0.3 * len(value))
#         fh.write(f"{key},{adenine_prediction}\n")

# # print predictions to screen
# print(open(predictions_outfile, 'r').read())
