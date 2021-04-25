# Bits Please

This is our submission for the CBH2021 hackathon. 

Our team is from Copenhagen and we think proteins are cool 🙌.

## Team Members

    Javier Marchena
    Roosa Varjus
    Caroline Linnea Elin Lennartsson
    Marida Ianni-Ravn
    Henrietta Holze

## Project Description

Our team created a Random Forest Regressor to predict protein solubility. We trained the model with a selected assortment of features and fine-tuned model parameters (number of trees, tree depth, etc). Our final Random Forest Regressor contains 1000 trees and is fed with 21 features.


## Features 
We chose a selected amount of features from the article "SOLart: a structure-based method to predict protein
solubility and aggregation" by Qingzhen Hou1, Jean Marc Kwasigroch,  Marianne Rooman and
Fabrizio Pucci.  In addition, we also added some new features to investigate the properties of solubility further. Some features are based on the amino acid sequence, while others are based on the protein structure.

The features we investigated are described below, with our new features being followed by (new):

* Accessible surface area in Å^2. 
* Radius of gyration (new).
* The length of the amino acid sequence. 
* Isoelectric point (new). 
* Charge of protein at pH 7. 
* Count of aromatic residues.
* Fraction of aromatic residues.
* The aromaticity of the protein (new).
* Molecular weights in kDa (new). 
* Instability index (new). 
* Average b-factor (new).
* Fraction of the surface divided by sequence length. 
* Fraction of moderatly buried beta residues. 
* Fraction of moderatly buried alfa residues. 
* Fraction of exposed buried alfa residues. 
* Fraction of K residues minus the fraction of R residues.   
* Fraction of negativly charged residues. 
* Fraction of positivly charged residues.  
* Fraction of charged residues.
* Fraction of positively minus negatively
* Charged residues.


![Workspace](https://github.com/roosavarjus/cbh21-protein-solubility-challenge/edit/main/figures/2021-04-24-084229353.jpg)
