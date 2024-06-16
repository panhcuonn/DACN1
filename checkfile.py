from joblib import load
import pickle
import pandas as pd
from sklearn.decomposition import PCA


# model = load("D:\processingdataset\streamlit\lrmodel_1.joblib")
# # with open('D:\processingdataset\streamlit\lrmodel.pkl', 'rb') as file:
# #     model = pickle.load(file)
# print(model.feature_names_in_)


dataset = pd.read_csv('D:\processingdataset\streamlit\Medicalpremium.csv')
dataset = dataset.iloc[:,:-1]
print(dataset.head())
pca = PCA(n_components=7)
a = pca.fit_transform(pca)

