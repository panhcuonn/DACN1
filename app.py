import pickle

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

# Load scikit-learn libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from joblib import dump, load
from sklearn.preprocessing import StandardScaler




# Load the data
@st.cache_data
def load_data():
    return pd.read_csv('Medicalpremium.csv')


# Load the model
# @st.cache(allow_output_mutation=True)
def load_model_Lrn(pkl=True):
    if pkl ==False:
        path ="D:\processingdataset\streamlit\lrmodel.joblib"
        return load(path)
    else:
        with open('D:\processingdataset\streamlit\lrmodel.pkl', 'rb') as file:
            return pickle.load(file)

def load_model_Poly(pkl=True):
    if pkl ==False:
        path = "D:\processingdataset\streamlit\polymodel.joblib"
        return load(path)
    else:
        with open('D:\processingdataset\streamlit\polymodel.pkl', 'rb') as file:
            return pickle.load(file)

def load_model_RF(pkl=True):
    if pkl ==False:
        path = "D:\processingdataset\streamlit/rfmodel.joblib"
        return load(path)
    else:
        with open('D:\processingdataset\streamlit/rfmodel.pkl', 'rb') as file:
            return pickle.load(file)

def load_model_GB(pkl=True):
    if pkl ==False:
        path = "D:\processingdataset\streamlit\gbmodel.joblib"
        return load(path)
    else:
        with open('D:\processingdataset\streamlit\gbmodel.pkl', 'rb') as file:
            return pickle.load(file)


# Make prediction
def predict(model, data):
    return model.predict(data)

def calculate_BMI(df):
    # Calculating BMI
    w = df['Weight'];
    h = df['Height'];

    # bmi = 10000*(weight/(height*height));

    df['BMI'] = 10000 * (w / (h * h))

    df['BMI_Status'] = np.select(
        [df['BMI'] < 18.499999,
         df['BMI'] >= 30,
         df['BMI'].between(18.5, 24.999999),
         df['BMI'].between(25, 29.9999999)],
        ['Underweight', 'Obesse', 'Normal', 'Overweight']
    )

    return df


def main():
    st.set_page_config(layout="wide")

    st.sidebar.title("Menu")
    selected = st.sidebar.radio("", ["Home", "Predict", "Visualizations"])

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
        model_Lrn = load_model_Lrn(pkl=False)
        model_Poly = load_model_Poly(pkl=False)
        model_RF = load_model_RF(pkl=False)
        model_GBM = load_model_GB(pkl=False)

        # --------------------------------------------------------------------------#

        # Collect user inputs
        Age = st.number_input("Age", min_value=18, max_value=70)
        Height = st.slider("Height(cm)", 140, 200)
        Weight = st.slider("Weight(kg)", 50, 140)
        NumberOfMajorSurgeries = st.number_input("Number Of Major Surgeries", min_value=0, max_value=10)
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

        # --------------------------------------------------------------------------#
        if st.button('Get Predict', ):
            # Make prediction
            prediction_Lrn = predict(model_Lrn, [
                [Age, NumberOfMajorSurgeries, AnyChronicDiseases, HistoryOfCancerInFamily, AnyTransplants, BMI,
                 BloodPressureProblems, Diabetes, KnownAllergies]])

            st.write(f"(Lrn)- Linear Regression model Your health insurance premium price is Rs. {prediction_Lrn[0]:.5f}")

            # prediction_Poly = predict(model_Poly, [
            #     [Age, NumberOfMajorSurgeries, AnyChronicDiseases, HistoryOfCancerInFamily, AnyTransplants, BMI,
            #      BloodPressureProblems, Diabetes, KnownAllergies]])
            # st.success(f"(Poly)- Polynomial model Your health insurance premium price is Rs. {prediction_Poly[0]:.5f}")

            prediction_RF = predict(model_RF, [
                [Age, NumberOfMajorSurgeries, AnyChronicDiseases, HistoryOfCancerInFamily, AnyTransplants, BMI,
                 BloodPressureProblems, Diabetes, KnownAllergies]])
            st.write(f"(RF)- Random Forest model Your health insurance premium price is Rs. {prediction_RF[0]:.5f}")

            prediction_GB = predict(model_GBM, [
                [Age, NumberOfMajorSurgeries, AnyChronicDiseases, HistoryOfCancerInFamily, AnyTransplants, BMI,
                 BloodPressureProblems, Diabetes, KnownAllergies]])
            st.write(f"(GBM)- Gradient Boosting model Your health insurance premium price is Rs. {prediction_GB[0]:.5f}")


        #
        # if prediction1[0] == prediction2[0]:
        #     st.success("The predictions are same")
        # else:
        #     st.success("The predictions are different")
        # if model_RF == model_GBM:
        #     print("The pickle files are identical")
        # else:
        #     print("The pickle files are different")




    # elif selected == "Contact":
    #     st.title("PYAR Insurance Company")
    #     st.subheader("Reach us at:")
    #     st.write("abc: 1111111111")
    #     st.write("def: 2222222222")
    #     st.write("pqr: 3333333333")
    #     st.write("lmn: 4444444444")
    #     st.write("pyarinsurance@pyar.com")
    #     st.write("insurance@pyar.com")

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

        df_BMI = calculate_BMI(df)
        sns.countplot(x='PremiumLabel', hue='BMI_Status', data=df_BMI, ax=axes[0])
        sns.boxplot(data=df, x="PremiumPrice", y="BMI_Status", hue="BMI_Status", dodge=False, ax=axes[1]).set_title(
            'Insurance Premium Price for Various BMI Status')
        # Display the plot in Streamlit
        st.pyplot(fig)


if __name__ == '__main__':
    main()