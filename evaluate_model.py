import pickle 


# 1) make bar plot with feature importances 
# 2) correlation coeffiitnts


# Load from file
with open('pickle_model.pkl', 'rb') as file:
    rfr = pickle.load(file)



# from sklearn import metrics

print(rfr.feature_importances_)

