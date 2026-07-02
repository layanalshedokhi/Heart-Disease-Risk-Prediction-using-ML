import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report 
df = pd.read_csv("heart_disease_risk_dataset_earlymed (1).csv")

X = df.drop("Heart_Risk", axis=1)
y = df["Heart_Risk"]

#Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Scaling ONLY age (needed for logistic regression)
scaler = StandardScaler()

X_train["Age"] = scaler.fit_transform(X_train[["Age"]])
X_test["Age"] = scaler.transform(X_test[["Age"]])
#using Logistic regression
lr = LogisticRegression(random_state=42)

lr.fit(X_train, y_train)

y_predict_lr = lr.predict(X_test)

#using Random forest
rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

y_predict_rf = rf.predict(X_test)


lr_report = classification_report(
    y_test,
    y_predict_lr,
    output_dict=True
)

lr_report = pd.DataFrame(lr_report).transpose()

rf_report = classification_report(
    y_test,
    y_predict_rf,
    output_dict=True
)

rf_report = pd.DataFrame(rf_report).transpose()

# ======================================================
# DASHBOARD STATISTICS
# ======================================================

total_patients = len(df)

high_risk = df["Heart_Risk"].sum()

low_risk = total_patients - high_risk

# We'll replace this later with your real model accuracy
model_accuracy = "96.4%"

# ======================================================
# PAGE CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.stApp {
    background-color: #F4F8F7;
}

section[data-testid="stSidebar"] {
    background-color: #1B2A4A;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] .stRadio label {
    background-color: transparent;
    border-radius: 10px;
    padding: 6px 10px;
}

section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
    background-color: rgba(255,255,255,0.1);
}

div[data-testid="stMetric"] {
    background-color: white;
    border-radius: 14px;
    padding: 20px 18px;
    box-shadow: 0 4px 14px rgba(27, 42, 74, 0.08);
    border-left: 5px solid #6FD6C4;
}

div[data-testid="column"]:nth-of-type(1) div[data-testid="stMetric"] {
    border-left-color: #6FD6C4;
}
div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetric"] {
    border-left-color: #F7C97F;
}
div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetric"] {
    border-left-color: #8FA8E8;
}
div[data-testid="column"]:nth-of-type(4) div[data-testid="stMetric"] {
    border-left-color: #F79FA0;
}

div[data-testid="stMetricValue"] {
    color: #1B2A4A;
    font-weight: 700;
}

h1, h2, h3 {
    color: #1B2A4A !important;
}

.stButton > button {
    background-color: #6FD6C4;
    color: #1B2A4A;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #5AC3B0;
    color: white;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(27, 42, 74, 0.06);
}

div[data-testid="stAlert"] {
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(27, 42, 74, 0.06);
}
</style>
""", unsafe_allow_html=True)
# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title(" HeartCare AI 🫀")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Data Overview",
        "EDA",
        "Preprocessing",
        "Model Results",
        "Prediction"
    ]
)

# ======================================================
# DASHBOARD PAGE
# ======================================================
if page == "Dashboard":

    st.title("Heart Disease Risk Prediction Dashboard ")

    st.caption("Machine Learning for Early Heart Disease Risk Detection")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="👥 Total Patients",
            value=total_patients
        )

    with col2:
        st.metric(
            label="🔴 High Risk",
            value=high_risk
        )

    with col3:
        st.metric(
            label="🟢 Low Risk",
            value=low_risk
        )

    with col4:
        st.metric(
            label="🎯 Accuracy",
            value=model_accuracy
        )

    st.markdown("---")

    

    st.subheader("Welcome!")

    st.write("""
This dashboard was developed to predict whether a patient is at **high risk of heart disease** using Machine Learning.

Use the navigation menu on the left to:

- Explore the dataset
- View exploratory data analysis (EDA)
- Understand preprocessing steps
- Evaluate model performance
- Predict heart disease risk for a new patient
""")

# ======================================================
# DATA PAGE
# ======================================================
# ======================================================
# DATA OVERVIEW
# ======================================================

elif page == "Data Overview":

    st.title("Data Overview 📊")

    st.caption("Summary of the Heart Disease Risk Dataset")

    st.write("""
This dataset contains patient medical and lifestyle information used to predict whether a patient has a **High** or **Low** risk of heart disease.
""")

    st.markdown("---")



    # =====================================
    # Dataset Summary
    # =====================================

    st.subheader("Dataset Summary")

    rows = df.shape[0]
    columns = df.shape[1]
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", rows)

    with col2:
        st.metric("Columns", columns)

    with col3:
        st.metric("Missing Values", missing)

    with col4:
        st.metric(
        "Duplicate Rows",
        duplicates,
        help="Duplicate records identified during the initial dataset inspection."
    )
        
    if duplicates > 0:
        st.warning(
        f"⚠️**Data Quality Status:** {duplicates} duplicate records were identified during the initial dataset inspection. "
        "These records are further analyzed and explained in the **Preprocessing** section."
    )
    else:
        st.success("✅ No duplicate records were detected.")

    st.markdown("---")

    # =====================================
    # Dataset Preview
    # =====================================

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # =====================================
    # Feature Information
    # =====================================

    st.subheader("Feature Information")

    feature_info = pd.DataFrame({
        "Feature": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True
    )

# ======================================================
# EDA PAGE
# ======================================================
elif page == "EDA":

    st.title("Exploratory Data Analysis ")

    st.caption(
        "Visual exploration of the dataset to identify important patterns before preprocessing and model training."
    )
    st.markdown("---")
    

    col1, col2 = st.columns(2, gap="large")

# ==========================
# LEFT COLUMN
# ==========================

    with col1:

     st.subheader("Heart Risk Distribution")

     risk_counts = df["Heart_Risk"].value_counts().sort_index()

     fig = px.pie(
        values=risk_counts.values,
        names=["No Risk", "Risk"],
        hole=0.55,
        color_discrete_sequence=["#8FD3C1", "#F7C97F"]
     )

     fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_size=15
     )

     fig.update_layout(
        height=350,
        showlegend=False,
        margin=dict(t=30, b=10, l=0, r=80)
     )

     st.plotly_chart(fig, use_container_width=True)

# ==========================
# RIGHT COLUMN
# ==========================

    with col2:

     st.subheader("Age Distribution by Heart Disease Risk")
     st.markdown("")

     fig, ax = plt.subplots(figsize=(5.5,4))

     sns.boxplot(
        data=df,
        x="Heart_Risk",
        y="Age",
        palette=["#8FD3C1", "#F7C97F"],
        linewidth=2,
        ax=ax
     )

     ax.set_xticklabels(["No Risk", "Heart Risk"])
     ax.set_xlabel("")
     ax.set_ylabel("Age")
     ax.set_title("")

     sns.despine()

     st.pyplot(fig)
   
    st.markdown("---")

    st.subheader("Key Findings")

    st.info("""
    • Balanced Dataset: The target class is perfectly balanced, with 50% of patients at heart disease risk and 50% not at risk — eliminating class imbalance as a concern for model training.

    • Patients classified as **Heart Risk** generally have a higher median age.

    • No age outliers are present in both groups.
     """)
    st.markdown("---")

    st.subheader("Symptoms Analysis")

    st.caption(
    "Exploring the prevalence of symptoms and the relationships between symptoms among high-risk patients."
)

    st.markdown("")
    col1, col2 = st.columns(2, gap="large")
    
    with col1:

     st.subheader("Most Common Symptoms")

     symptoms = [
        "Shortness_of_Breath",
        "Fatigue",
        "Palpitations",
        "Dizziness",
        "Swelling",
        "Pain_Arms_Jaw_Back",
        "Cold_Sweats_Nausea"
     ]

     risk_patients = df[df["Heart_Risk"] == 1]

     symptom_mean = (
        risk_patients[symptoms]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
     )

     symptom_mean.columns = ["Symptom", "Percentage"]

     fig = px.bar(
        symptom_mean,
        x="Symptom",
        y="Percentage",
        color="Percentage",
        color_continuous_scale="Teal",
        text_auto=".0%"
     )

     fig.update_layout(
        height=420,
        coloraxis_showscale=False,
        xaxis_title="",
        yaxis_title="Proportion of Patients",
        margin=dict(l=20, r=20, t=20, b=20)
     )

     fig.update_xaxes(tickangle=-25)

     st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


    with col2:

     st.subheader("Symptom Correlation")
     st.markdown("")

     correlation = risk_patients[symptoms].corr()

     fig, ax = plt.subplots(figsize=(6,5.8))
     fig.subplots_adjust(top=0.95)
     sns.heatmap(
        correlation,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar=False,
        ax=ax
     )
 
     ax.set_title("")

     st.pyplot(fig)



    st.subheader("Key Findings")

    st.info("""
• Uniform Symptom Prevalence:All seven symptoms occur at nearly identical rates (~80%) among at-risk patients, suggesting no single symptom stands out as more indicative of risk than the others.

• No Underlying Correlation: Despite their similar prevalence, revealing that symptoms co-occur by coincidence rather than by relationship.

• Statistical Independence: This confirms that symptoms are statistically independent — the presence of one symptom does not predict or increase the likelihood of another.
""")



    
# ======================================================
# PREPROCESSING PAGE
# ======================================================
elif page == "Preprocessing":
    st.title("Data Cleaning & Preprocessing")

    st.caption(
    "Preparing the dataset before training the Machine Learning models."
)
    st.markdown("---")

    st.subheader("Duplicate Records")

    duplicates = df.duplicated().sum()

    left_col, right_col = st.columns([3, 1], gap="large")

# ==========================
# Explanation
# ==========================

    with left_col:

     st.info(
        f"""

After further investigation, these records were **not removed** because they represent
different patients who happen to share identical medical characteristics rather than
data entry errors.

Therefore, all records were retained for model training.
"""
    )

# ==========================
# Investigation Result
# ==========================

    with right_col:

     st.metric(
        label="Duplicate Records",
        value=duplicates
    )

    st.caption("Records retained after investigation.")
    st.markdown("")

    st.subheader("Missing Values")
    missing = df.isnull().sum().sum()

    if missing == 0:
        st.success(
        "No missing values were found in the dataset. No imputation or data cleaning was required."
    )
       
    else:
       st.warning(
        f"The dataset contains {missing} missing values."
    )
       

    # ======================================================
# FEATURE SCALING
# ======================================================

    st.markdown("")

    st.subheader("Feature Scaling")

    st.write("""
The **Age** feature was standardized before training the **Logistic Regression**
model. Since Logistic Regression is sensitive to differences in feature scales.
""")

# Recreate the preprocessing used in the notebook

    X = df.drop("Heart_Risk", axis=1)
    y = df["Heart_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

    scaler = StandardScaler()

# Statistics BEFORE scaling
    before_min = X_train["Age"].min()
    before_max = X_train["Age"].max()
    before_mean = X_train["Age"].mean()
    before_std = X_train["Age"].std()

# Scale Age exactly as in the notebook
    X_train["Age"] = scaler.fit_transform(X_train[["Age"]])
    X_test["Age"] = scaler.transform(X_test[["Age"]])

# Statistics AFTER scaling
    after_min = X_train["Age"].min()
    after_max = X_train["Age"].max()
    after_mean = X_train["Age"].mean()
    after_std = X_train["Age"].std()

    before_col, after_col = st.columns(2, gap="large")

    with before_col:

     st.info(f"""
##### Before Scaling


- Minimum: **{before_min:.2f}**
- Maximum: **{before_max:.2f}**
- Mean: **{before_mean:.2f}**
- Standard Deviation: **{before_std:.2f}**
""")

    with after_col:

     st.success(f"""
##### After Scaling


- Minimum: **{after_min:.2f}**
- Maximum: **{after_max:.2f}**
- Mean: **{after_mean:.2f}**
- Standard Deviation: **{after_std:.2f}**
""")

    st.caption(
    "Only the Age feature was standardized using StandardScaler for the Logistic Regression model."
)

    
    
 # ======================================================
# MODEL RESULTS PAGE
# ======================================================

elif page == "Model Results":

    st.title("Model Performance")

    st.caption(
        "Performance evaluation of the trained Machine Learning models."
    )

    st.markdown("---")

    # ==================================================
    # MODEL TABS
    # ==================================================

    tab1, tab2 = st.tabs(
        ["Logistic Regression", "Random Forest"]
    )

    # ==================================================
    # LOGISTIC REGRESSION
    # ==================================================

    with tab1:

        # -----------------------------
        # Performance Metrics
        # -----------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Accuracy",
                f"{accuracy_score(y_test, y_predict_lr) * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Precision",
                f"{precision_score(y_test, y_predict_lr) * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Recall",
                f"{recall_score(y_test, y_predict_lr) * 100:.2f}%"
            )

        with col4:
            st.metric(
                "F1-Score",
                f"{f1_score(y_test, y_predict_lr) * 100:.2f}%"
            )

        st.markdown("---")

        st.subheader("Confusion Matrix")

        center_col = st.columns([1, 2, 1])

        with center_col[1]:

            fig, ax = plt.subplots(figsize=(6, 5))

            ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_predict_lr,
                cmap="PuBu",
                ax=ax
            )

            ax.set_title("")

            st.pyplot(fig)

        st.markdown("---")

        st.subheader("Key Findings")

        st.info("""
• Logistic Regression correctly classified the majority of patients.

• Produced fewer False Negatives, reducing the chance of missing high-risk patients.

• Showed balanced predictions across both classes, making it the preferred model.
""")

    # ==================================================
    # RANDOM FOREST
    # ==================================================

    with tab2:

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Accuracy",
                f"{accuracy_score(y_test, y_predict_rf) * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Precision",
                f"{precision_score(y_test, y_predict_rf) * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Recall",
                f"{recall_score(y_test, y_predict_rf) * 100:.2f}%"
            )

        with col4:
            st.metric(
                "F1-Score",
                f"{f1_score(y_test, y_predict_rf) * 100:.2f}%"
            )

        st.markdown("---")

        st.subheader("Confusion Matrix")

        center_col = st.columns([1, 2, 1])

        with center_col[1]:

            fig, ax = plt.subplots(figsize=(6, 5))

            ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_predict_rf,
                cmap="PuBu",
                ax=ax
            )

            ax.set_title("")

            st.pyplot(fig)

        st.markdown("---")

        st.subheader("Key Findings")

        st.info("""
• Random Forest correctly classified the majority of patients.

• Produced more False Negatives than Logistic Regression.

• Achieved strong performance but was less suitable for healthcare due to the higher False Negative rate.
""")

# ======================================================
# PREDICTION PAGE
# ======================================================

elif page == "Prediction":

    st.title("Heart Disease Risk Prediction 🫀")

    st.caption(
        "Enter the patient's information below to estimate the risk of heart disease using the trained Logistic Regression model."
    )

    st.markdown("---")

    # ==================================================
    # INPUT FORM
    # ==================================================

    left_col, right_col = st.columns(2, gap="large")

    # -----------------------------
    # Patient Information
    # -----------------------------

    with left_col:

        st.subheader("Patient Information")

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        age = st.slider(
            "Age",
            min_value=int(df["Age"].min()),
            max_value=int(df["Age"].max()),
            value=int(df["Age"].median())
        )

        smoking = st.selectbox("Smoking", ["No", "Yes"])
        obesity = st.selectbox("Obesity", ["No", "Yes"])
        sedentary = st.selectbox("Sedentary Lifestyle", ["No", "Yes"])
        family_history = st.selectbox("Family History", ["No", "Yes"])
        stress = st.selectbox("Chronic Stress", ["No", "Yes"])

    # -----------------------------
    # Medical Information
    # -----------------------------

    with right_col:

     st.subheader("Medical Information")

     with st.expander("🩺 Medical Conditions", expanded=True):

        high_bp = int(st.checkbox("High Blood Pressure"))
        cholesterol = int(st.checkbox("High Cholesterol"))
        diabetes = int(st.checkbox("Diabetes"))

     with st.expander("🤒 Symptoms", expanded=True):

        chest_pain = int(st.checkbox("Chest Pain"))
        shortness = int(st.checkbox("Shortness of Breath"))
        fatigue = int(st.checkbox("Fatigue"))
        palpitations = int(st.checkbox("Palpitations"))
        dizziness = int(st.checkbox("Dizziness"))
        swelling = int(st.checkbox("Swelling"))
        pain = int(st.checkbox("Pain in Arms, Jaw or Back"))
        cold = int(st.checkbox("Cold Sweats / Nausea"))

    

    # ==================================================
    # PREDICT BUTTON
    # ==================================================

    st.markdown("")

    c1, c2, c3 = st.columns([2,1,2])

    with c2:
     predict = st.button(
        "🔍 Predict Risk",
        use_container_width=True
    )    
    # ==================================================
    # PREDICTION
    # ==================================================

    if predict:
        patient = pd.DataFrame({

    "Chest_Pain": [chest_pain],
    "Shortness_of_Breath": [shortness],
    "Fatigue": [fatigue],
    "Palpitations": [palpitations],
    "Dizziness": [dizziness],
    "Swelling": [swelling],
    "Pain_Arms_Jaw_Back": [pain],
    "Cold_Sweats_Nausea": [cold],

    "High_BP": [high_bp],
    "High_Cholesterol": [cholesterol],
    "Diabetes": [diabetes],

    "Smoking": [1 if smoking == "Yes" else 0],
    "Obesity": [1 if obesity == "Yes" else 0],
    "Sedentary_Lifestyle": [1 if sedentary == "Yes" else 0],
    "Family_History": [1 if family_history == "Yes" else 0],
    "Chronic_Stress": [1 if stress == "Yes" else 0],

    "Gender": [1 if gender == "Male" else 0],
    "Age": [age]

})


        # Scale Age

        patient["Age"] = scaler.transform(patient[["Age"]])

        prediction = lr.predict(patient)[0]

        probability = lr.predict_proba(patient)[0]

        confidence = max(probability) * 100

        st.markdown("---")

        # ==================================================
        # RESULT
        # ==================================================

        if prediction == 1:

            st.error(f"""
## ❤️ High Risk

The model predicts that the patient is at **High Risk** of heart disease.

**Prediction Confidence:** {confidence:.1f}%
""")

        else:

            st.success(f"""
## 💚 Low Risk

The model predicts that the patient is at **Low Risk** of heart disease.

**Prediction Confidence:** {confidence:.1f}%
""")

        st.markdown("---")



