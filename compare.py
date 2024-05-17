import pickle

# Load the first pickle file
with open('GBM_model.pkl', 'rb') as f:
    obj1 = pickle.load(f)

# Load the second pickle file
with open('RF_model.pkl', 'rb') as f:
    obj2 = pickle.load(f)
    


# Compare the two objects
if obj1 == obj2:
    print("The pickle files are identical")
else:
    print("The pickle files are different")
