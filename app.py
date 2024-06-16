import pickle

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import warnings  as ws
ws.filterwarnings("ignore")

# Load scikit-learn libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline



# Load the data
# @st.cache_data
def load_data():
    return pd.read_csv('Medicalpremium.csv')


# Load the model
# @st.cache(allow_output_mutation=True)
def load_model(name_model):
    if name_model == "linear":
        path = "D:\processingdataset\streamlit\lrmodel.joblib"
    if name_model == "poly":
        path = "../streamlit/poly_pca_20.joblib"
    if name_model == "rf":
        path = "D:\processingdataset\streamlit/rfmodel.joblib"
    if name_model == "gb":
        path = "D:\processingdataset\streamlit\gbmodel.joblib"

    if name_model == "linear_PCA":
        path = "D:\processingdataset\streamlit\lrmodel.joblib"
    if name_model == "poly_PCA":
        path = "streamlit/poly_pca_20.joblib"
    if name_model == "rf_PCA":
        path = "D:\processingdataset\streamlit/rfmodel.joblib"
    if name_model == "gb_PCA":
        path = "D:\processingdataset\streamlit\gbmodel.joblib"
    return load(path)


# Make prediction
def predict(model, data):
    return model.predict(data)


def predict_poly(model, data):
    return None


# Preprocessing data
def preprocess_poly(x):
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(x)
    X_poly = pd.DataFrame(X_poly, columns=poly_features.get_feature_names_out(x.columns))
    return X_poly


def preprocess_PCA(x):
    pca = PCA(n_components=7)
    pca.fit_transform(x)
    return x 


def process_data(df):
    # Calculating BMI
    w = df['Weight'];
    h = df['Height'];

    df['BMI'] = 10000 * (w / (h * h))

    df['BMI_Status'] = np.select(
        [df['BMI'] < 18.499999,
         df['BMI'] >= 30,
         df['BMI'].between(18.5, 24.999999),
         df['BMI'].between(25, 29.9999999)],
        ['Underweight', 'Obesse', 'Normal', 'Overweight']
    )
    return df


def get_x(columns, values):
    X_df = {}
    X_dict = dict(zip(columns, values))
    X_df = pd.DataFrame([X_dict])
    return X_df

def main():
    st.set_page_config(layout="wide")

    st.sidebar.title("Menu")
    selected = st.sidebar.radio("", ["Home", "Predict","PCA", "Visualizations"])

    if selected == "Home":
        st.title("PYAR Insurance Company")
        st.header("Mission :")
        st.markdown(
            "**To be the most preferred choice of customers for General Insurance by building Relationships and grow profitably.**")
        st.header("Vision :")
        st.markdown("Leveraging technology to integrate people and processes.")
        st.markdown("To excel in service and performance.")
        st.markdown("To uphold the highest ethical standards in conducting our business.")
        st.header("What is Insurance?")
        st.markdown(
            "Most people have some kind of insurance: for their car, their house, or even their life. Yet most of us don’t stop to think too much about what insurance is or how it works.Put simply, insurance is a contract, represented by a policy, in which a policyholder receives financial protection or reimbursement against losses from an insurance company. The company pools clients’ risks to make payments more affordable for the insured.Insurance policies are used to hedge against the risk of financial losses, both big and small, that may result from damage to the insured or their property, or from liability for damage or injury caused to a third party.")
        st.header("KeyTakeaways")
        st.markdown(
            "Insurance is a contract (policy) in which an insurer indemnifies another against losses from specific contingencies or perils.")
        st.markdown(
            "There are many types of insurance policies. Life, health, homeowners, and auto are the most common forms of insurance")
        st.markdown(
            "The core components that make up most insurance policies are the deductible, policy limit, and premium.")
        st.header("How Insurance Works")
        st.markdown(
            "A multitude of different types of insurance policies is available, and virtually any individual or business can find an insurance company willing to insure them—for a price. The most common types of personal insurance policies are auto, health, homeowners, and life. Most individuals in the United States have at least one of these types of insurance, and car insurance is required by law.Businesses require special types of insurance policies that insure against specific types of risks faced by a particular business. For example, a fast-food restaurant needs a policy that covers damage or injury that occurs as a result of cooking with a deep fryer. An auto dealer is not subject to this type of risk but does require coverage for damage or injury that could occur during test drives.")
        st.subheader("Important Note:")
        st.markdown(
            "To select the best policy for you or your family, it is important to pay attention to the three critical components of most insurance policies: 1.deductible, 2.premium, and 3.policy limit.")

    elif selected == "Predict":
        st.title("Insurance Premium Prediction")
        st.subheader("Enter your information")

        # Load data

        df = load_data()

        # Load model
        model_Lrn = load_model("linear")
        model_Poly = load_model("poly")
        model_RF = load_model("rf").best_estimator_
        model_GB = load_model("gb").best_estimator_


        # --------------------------------------------------------------------------#

        # Collect user inputs
        Age = st.number_input("Age", min_value=18, max_value=70)
        Height = st.slider("Height(cm)", 140, 200)
        Weight = st.slider("Weight(kg)", 50, 140)
        NumberOfMajorSurgeries = st.slider("Number Of Major Surgeries", min_value=0, max_value=5)
        AnyChronicDiseases = st.checkbox('Any Chronic Diseases')
        HistoryOfCancerInFamily = st.checkbox('History Of Cancer In Family')
        AnyTransplants = st.checkbox('Any Transplants')
        BloodPressureProblems = st.checkbox('Blood Pressure Problems')
        Diabetes = st.checkbox('Diabetes')
        KnownAllergies = st.checkbox('Known Allergies')

        # --------------------------------------------------------------------------#

        # Convert checkbox values to binary
        Diabetes = 1 if Diabetes else 0
        BloodPressureProblems = 1 if BloodPressureProblems else 0
        AnyTransplants = 1 if AnyTransplants else 0
        AnyChronicDiseases = 1 if AnyChronicDiseases else 0
        KnownAllergies = 1 if KnownAllergies else 0
        HistoryOfCancerInFamily = 1 if HistoryOfCancerInFamily else 0
        BMI = Weight / ((Height / 100) ** 2)

        columns = ('Age', 'Diabetes', 'BloodPressureProblems',
                   'AnyTransplants', 'AnyChronicDiseases',
                   'KnownAllergies',
                   'HistoryOfCancerInFamily',
                   'NumberOfMajorSurgeries',
                   'BMI')

        values = (
            Age,
            Diabetes,
            BloodPressureProblems,
            AnyTransplants,
            AnyChronicDiseases,
            KnownAllergies,
            HistoryOfCancerInFamily,
            NumberOfMajorSurgeries,
            BMI,
        )

        # --------------------------------------------------------------------------#
        if st.button('Get Predict', ):
            # Make prediction

            X = get_x(columns=columns, values=values)

            prediction_Lrn = predict(model_Lrn,X)
            st.write(f"(Lrn)- Linear Regression model Your health insurance premium price is Rs. {prediction_Lrn[0]:.5f}")

            prediction_Poly = predict(model_Poly, X)
            st.write(f"(Poly)- Polynomial model Your health insurance premium price is Rs. {(prediction_Poly[0]):.5f}")

            prediction_RF = predict(model_RF, X)
            st.write(f"(RF)- Random Forest model Your health insurance premium price is Rs. {prediction_RF[0]:.5f}")

            prediction_GB = predict(model_GB,X)
            st.write(f"(GBM)- Gradient Boosting model Your health insurance premium price is Rs. {prediction_GB[0]:.5f}")

    elif selected == "PCA":
        df = load_data()
        # Visual data
        st.subheader("Data for Predicting")

        data_final = process_data(df)
        data_final = data_final.drop(['Height','Weight','BMI_Status'], axis=1)
        # Visual
        st.write(data_final)
        st.write(data_final.shape)

        X = data_final.drop(['PremiumPrice'],axis=1)
        y = data_final['PremiumPrice']

        # create PCA for observe
        scaler = StandardScaler()
        xsc = scaler.fit_transform(X)

        pca = PCA()
        pca.fit_transform(xsc)

        st.subheader('PCA')
        loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(1, len(data_final.columns))], index=data_final.columns[:-1])
        st.write(loadings)

        # Visual heatmap of PCA
        st.subheader("PCA Loadings Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(loadings, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Visual Scree Plot and Cumulative Explained Variance

        st.subheader("Scree Plot and Cumulative Explained Variance")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        # Plot scree plot in the first subplot
        ax1.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o',
                 linestyle='--')
        ax1.set_title('Scree Plot')
        ax1.set_xlabel('Number of components')
        ax1.set_ylabel('Explained variance ratio')

        # Calculate cumulative explained variance
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

        # Plot cumulative explained variance in the second subplot
        ax2.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o',
                 linestyle='--')
        ax2.set_title('Cumulative Explained Variance')
        ax2.set_xlabel('Number of components')
        ax2.set_ylabel('Cumulative explained variance')
        ax2.axhline(y=0.85, color='r', linestyle='-', label='85% Threshold')
        ax2.text(0.5, 0.85, '85% Cut-off Threshold', color='red', fontsize=16)
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # --------------------------------------------------------------------------#

        # Load model
        model_Lrn = load_model("linear")
        model_Poly = load_model("poly")
        model_RF = load_model("rf").best_estimator_
        model_GB = load_model("gb").best_estimator_

        # Get features
        st.write(f"linear {model_Lrn.feature_names_in_}")
        st.write(f"poly {model_Poly.feature_names_in_}")
        st.write(f"random forest {model_RF.feature_names_in_}")
        st.write(f"gradient boosting{model_GB.feature_names_in_}")

        # --------------------------------------------------------------------------#

        # Collect user inputs
        Age = st.number_input("Age", min_value=18, max_value=70)
        Height = st.slider("Height(cm)", 140, 200)
        Weight = st.slider("Weight(kg)", 50, 140)
        NumberOfMajorSurgeries = st.slider("Number Of Major Surgeries", min_value=0, max_value=5)
        AnyChronicDiseases = st.checkbox('Any Chronic Diseases')
        HistoryOfCancerInFamily = st.checkbox('History Of Cancer In Family')
        AnyTransplants = st.checkbox('Any Transplants')
        BloodPressureProblems = st.checkbox('Blood Pressure Problems')
        Diabetes = st.checkbox('Diabetes')
        KnownAllergies = st.checkbox('Known Allergies')

        # --------------------------------------------------------------------------#

        # Testing PCA
        st.title("Testing PCA")


        # Convert checkbox values to binary
        Diabetes = 1 if Diabetes else 0
        BloodPressureProblems = 1 if BloodPressureProblems else 0
        AnyTransplants = 1 if AnyTransplants else 0
        AnyChronicDiseases = 1 if AnyChronicDiseases else 0
        KnownAllergies = 1 if KnownAllergies else 0
        HistoryOfCancerInFamily = 1 if HistoryOfCancerInFamily else 0
        BMI = Weight / ((Height / 100) ** 2)

        dict_encode = {
            'Age': Age,
            'Diabetes': Diabetes,
            'BloodPressureProblems': BloodPressureProblems,
            'AnyTransplants': AnyTransplants,
            'AnyChronicDiseases': AnyChronicDiseases,
            'KnownAllergies': KnownAllergies,
            'HistoryOfCancerInFamily': HistoryOfCancerInFamily,
            'NumberOfMajorSurgeries': NumberOfMajorSurgeries,
            'BMI': BMI,
        }

        feature_columns_Lrn = model_Lrn.feature_names_in_
        feature_columns_Poly = model_Poly.feature_names_in_
        feature_columns_RF = model_RF.feature_names_in_
        feature_columns_GB = model_GB.feature_names_in_

        X_lrn = get_x(lst_order=feature_columns_Lrn, dict_encode=dict_encode)
        # X_Poly = get_x_poly(lst_order=feature_columns_Lrn, dict_encode=dict_encode)
        X_RF = get_x(lst_order=feature_columns_RF, dict_encode=dict_encode)
        X_GB = get_x(lst_order=feature_columns_GB, dict_encode=dict_encode)

        st.write(f"X_lrn",X_lrn)
        st.write(f"X_RF",X_RF)
        st.write(f"X_GB",X_GB)

        n_components = 7
        pca = PCA(n_components=n_components)
        st.write("xsc", xsc)
        pca.fit(xsc)
        st.write("total explained_variance_ratio_ with n_components is 7: ", sum(pca.explained_variance_ratio_))

        X_lrn_PCA = pca.transform(X_lrn)
        # X_Poly_PCA = pca.transform()
        X_RF_PCA = pca.transform(X_RF)
        X_GB_PCA = pca.transform(X_GB)

        st.write("X_lrn_PCA", X_lrn_PCA)
        st.write("X_RF_PCA", X_RF_PCA)
        st.write("X_GB_PCA", X_GB_PCA)



    elif selected == "Visualizations":
        st.title("Visualizations of the Data")
        st.subheader("DataFrame")
        df = load_data()
        st.write(df)

        st.subheader("DataFrame info")
        st.write(df.describe().T)


        # --------------------------------------------------------------------------#

        # Generate a mask for the upper triangle
        st.subheader('Correlation Heatmap')
        corr = df.corr()
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=1.0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

        # Display the heatmap in Streamlit
        st.pyplot(fig)
        palette = ["#82CCDD", "#F4D08F"]

        # --------------------------------------------------------------------------#

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x=df.Age, y=df.PremiumPrice, ax=ax)
        ax.set_title('Insurance Premium Price by Age')
        st.pyplot(fig)

        # 1,Distribution of the Insurance Premium Price
        st.subheader('1,Distribution of the Insurance Premium Price')

        fig, axes = plt.subplots(ncols=2, figsize=(14, 4))  # Create subplots with 2 columns

        # Distribution of the Insurance Premium Price
        sns.countplot(x='PremiumPrice', data=df, ax=axes[0])
        axes[0].set_title('Distribution of the Insurance Premium Price')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

        # Insurance Premium Price Label
        pr_lab = ['Low', 'Basic', 'Average', 'High', 'SuperHigh']
        df['PremiumLabel'] = pd.cut(df['PremiumPrice'], bins=5, labels=pr_lab, precision=0)
        sns.countplot(x='PremiumLabel', data=df, ax=axes[1])
        axes[1].set_title('Distribution of the Insurance Premium Price Label', fontsize=12, fontdict={"weight": "bold"})

        plt.tight_layout()
        st.pyplot(fig)

        # --------------------------------------------------------------------------#

        # 2,Distribution of the Diabetic vs Non-Diabetic Patients
        st.subheader('2,Distribution of Diabetic vs Non-Diabetic Patients')

        # Insurance Premium Price for Diabetic vs Non-Diabetic Patients
        fig, axes = plt.subplots(ncols=2, figsize=(14, 4))  # Create subplots with 2 columns

        # Insurance Premium Price for Diabetic vs Non-Diabetic Patients
        sns.barplot(data=df, x="Diabetes", y="PremiumPrice", palette=palette, ax=axes[0]).set_title(
            'Insurance Premium Price for Diabetic vs Non-Diabetic Patients',
            )

        # Density plot for Diabetic vs Non-Diabetic Patients
        sns.kdeplot(data=df, x="PremiumPrice", hue="Diabetes", palette=palette, fill=True, ax=axes[1])
        plt.title('Density plot for Diabetic vs Non-Diabetic Patients', fontsize=12, fontdict={"weight": "bold"})

        st.pyplot(fig)

        # --------------------------------------------------------------------------#

        # 3,Distribution of the Patients with/without Blood Pressure Problems
        st.subheader('3,Distribution of Patients with/without Blood Pressure Problems')

        fig, axes = plt.subplots(ncols=2, figsize=(14, 4))  # Create subplots with 2 columns

        # Insurance Premium Price for Patients with/without Blood Pressure Problems
        sns.barplot(data=df, x="BloodPressureProblems", y="PremiumPrice", palette=palette, ax=axes[0]).set_title(
            'Insurance Premium Price for Patients with/without Blood Pressure Problems')

        # Density plot for Patients with/without Blood Pressure Problems
        sns.kdeplot(data=df, x="PremiumPrice", hue="BloodPressureProblems", palette=palette, fill=True, ax=axes[1])
        plt.title('Density plot for Patients with/without Blood Pressure Problems', fontsize=12,
                  fontdict={"weight": "bold"})
        # Display the plots in Streamlit
        st.pyplot(fig)

        # --------------------------------------------------------------------------#

        # 4,Distribution of the Patients with/without Any Transplants
        st.subheader('4,Distribution of Patients with/without Any Transplants')

        fig, axes = plt.subplots(ncols=2, figsize=(14, 4))  # Create subplots with 2 columns

        sns.barplot(data=df, x="AnyTransplants", y="PremiumPrice", palette=palette, ax=axes[0]).set_title(
            'Insurance Premium Price for Patients with/without Any Transplants',
            )
        sns.kdeplot(data=df, x="PremiumPrice", hue="AnyTransplants", palette=palette, fill=True, ax=axes[1])
        plt.title('Density plot for Patients with/without Any Transplants', fontsize=12,
                          fontdict={"weight": "bold"})
        # Display the plots in Streamlit
        st.pyplot(fig)

        # --------------------------------------------------------------------------#

        # 5,Distribution of the Patients with/without Any Chronic Disease
        st.subheader('5,Distribution of Patients with/without Any Chronic Disease')

        fig, axes = plt.subplots(ncols=2, figsize=(14, 4))  # Create subplots with 2 columns

        sns.barplot(data=df, x="AnyChronicDiseases", y="PremiumPrice", palette=palette, ax=axes[0]).set_title(
            'Insurance Premium Price for Patients with/without Any Chronic Diseases',
            )
        sns.kdeplot(data=df, x="PremiumPrice", hue="AnyChronicDiseases", palette=palette, fill=True, ax=axes[1])
        plt.title('Density plot for Patients with/without Any Chronic Diseases', fontsize=12,
                          fontdict={"weight": "bold"})
        # Display the plots in Streamlit
        st.pyplot(fig)

        # --------------------------------------------------------------------------#

        # 6,Distribution of the Patients with/without Any Chronic Disease
        st.subheader('6,Distribution of Patients with/without Any Known Allergies')

        fig, axes = plt.subplots(ncols=2, figsize=(14, 4))  # Create subplots with 2 columns

        sns.barplot(data=df, x="KnownAllergies", y="PremiumPrice", palette=palette, ax=axes[0]).set_title(
            'Insurance Premium Price for Patients with/without Any Known Allergies',
        )
        sns.kdeplot(data=df, x="PremiumPrice", hue="KnownAllergies", palette=palette, fill=True, ax=axes[1])
        plt.title('Density plot for Patients with/without Any Known Allergies', fontsize=12,
                  fontdict={"weight": "bold"})
        # Display the plots in Streamlit
        st.pyplot(fig)

        # --------------------------------------------------------------------------#

        # 7,Distribution of the Patients with/without Any History Of Cancer In Family
        st.subheader('7,Distribution of Patients with/without Any History Of Cancer In Family')

        fig, axes = plt.subplots(ncols=2, figsize=(14, 4))  # Create subplots with 2 columns

        sns.barplot(data=df, x="HistoryOfCancerInFamily", y="PremiumPrice", palette=palette, ax=axes[0]).set_title(
            'Insurance Premium Price for Patients with/without Any History Of Cancer In Family',
        )
        sns.kdeplot(data=df, x="PremiumPrice", hue="HistoryOfCancerInFamily", palette=palette, fill=True, ax=axes[1])
        plt.title('Density plot for Patients with/without Any History Of Cancer In Family', fontsize=12,
                  fontdict={"weight": "bold"})
        # Display the plots in Streamlit
        st.pyplot(fig)

        # --------------------------------------------------------------------------#

        palette2 = ["#007BFF", '#547CB0', '#A87D62', "#FD7E14"]
        # 8,Distribution of the Number Of Major Surgeries by Patients
        st.subheader('8,Distribution of Number Of Major Surgeries by Patients')

        fig, axes = plt.subplots(ncols=2, figsize=(14, 4))  # Create subplots with 2 columns

        sns.barplot(data=df, x="NumberOfMajorSurgeries", palette=palette2, y="PremiumPrice", ax=axes[0]).set_title(
            'Insurance Premium Price for Patients with/without Any History Of Cancer In Family',
        )
        sns.kdeplot(data=df, x="PremiumPrice", hue="NumberOfMajorSurgeries", palette=palette2,fill=True, ax=axes[1])
        plt.title('Density plot for Number Of Major Surgeries by Patients', fontsize=12,
                  fontdict={"weight": "bold"})
        # Display the plots in Streamlit
        st.pyplot(fig)

        # --------------------------------------------------------------------------#

        # 9,Distribution of the BMI status
        st.subheader('9,Distribution of the BMI status')

        fig, axes = plt.subplots(ncols=2, figsize=(14, 4))

        df_BMI = process_data(df)
        sns.countplot(x='PremiumLabel', hue='BMI_Status', data=df_BMI, ax=axes[0])
        sns.boxplot(data=df, x="PremiumPrice", y="BMI_Status", hue="BMI_Status", dodge=False, ax=axes[1]).set_title(
            'Insurance Premium Price for Various BMI Status')
        # Display the plot in Streamlit
        st.pyplot(fig)


if __name__ == '__main__':
    main()