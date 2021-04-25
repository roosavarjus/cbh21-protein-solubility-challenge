import argparse
import csv
from Bio import SeqIO
import pandas as pd
import pickle
import glob
import pprint
import temppathlib
import zipfile

from feature_construction import compute_features


if __name__ == "__main__":

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

        # the following is adapted to how we build the feature construction 

        # convert the paths into a list of filename strings
        filenames = [str(tmp) for tmp in tmpdir.path.glob("*.pdb")]

        # compute the features for all proteins
        features = compute_features(filenames)

        # Load model from file
        with open('pickle_model.pkl', 'rb') as file:
            rfr = pickle.load(file)

        # separate the protein names from the features
        proteins = features['protIDs'].values
        XTest = features.iloc[:, 1:].to_numpy()

        # run the prediction on the features
        rfr_prediction = rfr.predict(XTest)

    # save the predictions in a dictionary and file
    predictions = [
        {'protein': proteins[i], 'solubility': rfr_prediction[i]}
        for i in range(len(proteins))
    ]

    # save to csv file, this will be used for benchmarking
    outpath = "predictions.csv"
    with open(outpath, "w") as fh:
        writer = csv.DictWriter(fh, fieldnames=["protein", "solubility"])
        writer.writeheader()
        writer.writerows(predictions)

    # print predictions to screen
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(predictions)

