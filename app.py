import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Wine Quality Analytics Dashboard",
    page_icon="🍷",
    layout="wide"
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("winequality.csv")

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    return df


try:
    df = load_data()
except Exception as e:
    st.error("Could not load winequality.csv")
    st.write("Make sure winequality.csv is inside the same folder as app.py.")
    st.exception(e)
    st.stop()


# ============================================================
# BASIC DATA CLEANING
# ============================================================

# Remove completely empty rows
df = df.dropna(how="all")

# Remove duplicate rows
df = df.drop_duplicates()

# Convert possible numeric columns to numeric where possible
for column in df.columns:
    if column != "color":
        try:
            df[column] = pd.to_numeric(df[column])
        except:
            pass


# Check target column
if "quality" not in df.columns:
    st.error("The column 'quality' was not found in the dataset.")
    st.write("Columns found:")
    st.write(df.columns.tolist())
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🍷 Wine Quality Analytics Dashboard")
st.write(
    "Analysis and machine learning of the Wine Quality Dataset "
    "from the Alcoholic Beverage Sector project."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dataset Overview",
        "Sprint 1 - EDA",
        "Sprint 2 - Dashboard",
        "Sprint 3 - Machine Learning"
    ]
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

if page == "Dataset Overview":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    with col4:
        st.metric("Duplicate Rows Removed", "Yes")

    st.subheader("Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Column Names")

    st.write(df.columns.tolist())

    st.subheader("Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(
        dtype_df,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )


# ============================================================
# SPRINT 1 - EDA
# ============================================================

elif page == "Sprint 1 - EDA":

    st.header("Sprint 1 - Exploratory Data Analysis")

    st.write(
        "The purpose of EDA is to understand the Wine Quality Dataset, "
        "study the distribution of variables, identify important "
        "variables related to wine quality, and obtain useful insights."
    )

    # --------------------------------------------------------
    # 1. DATASET DIMENSIONS
    # --------------------------------------------------------

    st.subheader("1. Dataset Dimensions")

    st.write(
        f"The dataset contains **{df.shape[0]} rows** and "
        f"**{df.shape[1]} columns**."
    )

    # --------------------------------------------------------
    # 2. QUALITY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("2. Distribution of Wine Quality")

    quality_counts = df["quality"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    quality_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Number of Wines")
    ax.set_title("Distribution of Wine Quality")

    st.pyplot(fig)

    # --------------------------------------------------------
    # 3. DISTRIBUTION OF NUMERIC COLUMNS
    # --------------------------------------------------------

    st.subheader("3. Distribution of Variables")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    selected_column = st.selectbox(
        "Select a column to view its distribution:",
        numeric_columns
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        df[selected_column].dropna(),
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {selected_column.title()}"
    )

    st.pyplot(fig)

    # --------------------------------------------------------
    # 4. STATISTICAL SUMMARY
    # --------------------------------------------------------

    st.subheader("4. Statistical Summary")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    # --------------------------------------------------------
    # 5. CORRELATION
    # --------------------------------------------------------

    st.subheader("5. Correlation Analysis")

    numeric_df = df.select_dtypes(include=np.number)

    correlation = numeric_df.corr()

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)

    # --------------------------------------------------------
    # 6. VARIABLES RELATED TO QUALITY
    # --------------------------------------------------------

    st.subheader("6. Significant Variables with Respect to Quality")

    quality_corr = (
        numeric_df.corr()["quality"]
        .drop("quality")
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    correlation_table = pd.DataFrame({
        "Variable": quality_corr.index,
        "Correlation with Quality": quality_corr.values
    })

    st.dataframe(
        correlation_table,
        use_container_width=True
    )

    # --------------------------------------------------------
    # 7. ALCOHOL VS QUALITY
    # --------------------------------------------------------

    if "alcohol" in df.columns:

        st.subheader("7. Alcohol vs Quality")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.boxplot(
            x="quality",
            y="alcohol",
            data=df,
            ax=ax
        )

        ax.set_title("Alcohol Content vs Wine Quality")

        st.pyplot(fig)

    # --------------------------------------------------------
    # 8. VOLATILE ACIDITY VS QUALITY
    # --------------------------------------------------------

    if "volatile acidity" in df.columns:

        st.subheader("8. Volatile Acidity vs Quality")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.boxplot(
            x="quality",
            y="volatile acidity",
            data=df,
            ax=ax
        )

        ax.set_title(
            "Volatile Acidity vs Wine Quality"
        )

        st.pyplot(fig)

    # --------------------------------------------------------
    # 9. SULPHATES VS QUALITY
    # --------------------------------------------------------

    if "sulphates" in df.columns:

        st.subheader("9. Sulphates vs Quality")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.boxplot(
            x="quality",
            y="sulphates",
            data=df,
            ax=ax
        )

        ax.set_title(
            "Sulphates vs Wine Quality"
        )

        st.pyplot(fig)

    # --------------------------------------------------------
    # 10. COLOR DISTRIBUTION
    # --------------------------------------------------------

    if "color" in df.columns:

        st.subheader("10. Wine Color Distribution")

        color_counts = df["color"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))

        color_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Wine Color")
        ax.set_ylabel("Number of Wines")
        ax.set_title("Red Wine vs White Wine")

        st.pyplot(fig)

    # --------------------------------------------------------
    # INSIGHTS
    # --------------------------------------------------------

    st.subheader("EDA Insights")

    st.markdown(
        """
        **Important observations:**

        1. Wine quality scores are distributed across different quality levels.
        2. Chemical properties such as alcohol, volatile acidity,
           sulphates and other physicochemical variables can be
           studied in relation to quality.
        3. Correlation analysis helps identify variables that have
           stronger relationships with the quality score.
        4. Alcohol content can be compared across different quality
           levels to understand its relationship with wine quality.
        5. Volatile acidity and sulphates can also be compared with
           quality scores.
        6. Wine color allows comparison between red and white wines.

        **Recommendations:**

        - Wine producers can monitor important chemical properties
          during production.
        - Variables strongly associated with quality can be given
          greater attention during quality control.
        - Machine learning can be used to predict wine quality
          from the available chemical properties.
        """
    )


# ============================================================
# SPRINT 2 - STREAMLIT DASHBOARD
# ============================================================

elif page == "Sprint 2 - Dashboard":

    st.header("Sprint 2 - Interactive Streamlit Dashboard")

    st.write(
        "Use the filters below to interactively explore the dataset."
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.sidebar.subheader("Filters")

    filtered_df = df.copy()

    # Wine color filter
    if "color" in df.columns:

        color_values = (
            df["color"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_colors = st.sidebar.multiselect(
            "Wine Color",
            options=color_values,
            default=color_values
        )

        filtered_df = filtered_df[
            filtered_df["color"].isin(selected_colors)
        ]

    # Quality filter
    quality_values = sorted(
        df["quality"].dropna().unique()
    )

    selected_quality = st.sidebar.multiselect(
        "Quality Score",
        options=quality_values,
        default=quality_values
    )

    filtered_df = filtered_df[
        filtered_df["quality"].isin(selected_quality)
    ]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Filtered Wines",
            len(filtered_df)
        )

    with col2:
        st.metric(
            "Average Quality",
            round(filtered_df["quality"].mean(), 2)
            if len(filtered_df) > 0 else 0
        )

    with col3:
        st.metric(
            "Minimum Quality",
            filtered_df["quality"].min()
            if len(filtered_df) > 0 else 0
        )

    with col4:
        st.metric(
            "Maximum Quality",
            filtered_df["quality"].max()
            if len(filtered_df) > 0 else 0
        )

    # --------------------------------------------------------
    # FILTERED DATA
    # --------------------------------------------------------

    st.subheader("Filtered Dataset")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # QUALITY CHART
    # --------------------------------------------------------

    st.subheader("Quality Distribution")

    if len(filtered_df) > 0:

        quality_chart = (
            filtered_df["quality"]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        quality_chart.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Quality Score")
        ax.set_ylabel("Number of Wines")
        ax.set_title("Filtered Wine Quality Distribution")

        st.pyplot(fig)

    # --------------------------------------------------------
    # ALCOHOL ANALYSIS
    # --------------------------------------------------------

    if "alcohol" in filtered_df.columns:

        st.subheader("Alcohol Content Analysis")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.boxplot(
            x="quality",
            y="alcohol",
            data=filtered_df,
            ax=ax
        )

        ax.set_title(
            "Alcohol Content by Quality Score"
        )

        st.pyplot(fig)

    # --------------------------------------------------------
    # VOLATILE ACIDITY
    # --------------------------------------------------------

    if "volatile acidity" in filtered_df.columns:

        st.subheader("Volatile Acidity Analysis")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.boxplot(
            x="quality",
            y="volatile acidity",
            data=filtered_df,
            ax=ax
        )

        ax.set_title(
            "Volatile Acidity by Quality Score"
        )

        st.pyplot(fig)

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.subheader("Correlation Heatmap")

    if len(filtered_df) > 1:

        filtered_numeric = filtered_df.select_dtypes(
            include=np.number
        )

        if filtered_numeric.shape[1] > 1:

            fig, ax = plt.subplots(
                figsize=(12, 8)
            )

            sns.heatmap(
                filtered_numeric.corr(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                ax=ax
            )

            ax.set_title(
                "Correlation Heatmap of Filtered Data"
            )

            st.pyplot(fig)


# ============================================================
# SPRINT 3 - MACHINE LEARNING
# ============================================================

elif page == "Sprint 3 - Machine Learning":

    st.header("Sprint 3 - Wine Quality Prediction")

    st.write(
        "Machine learning models are trained to predict the wine "
        "quality score using the input variables specified in the project."
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target = "quality"

    # --------------------------------------------------------
    # INPUT FEATURES
    # --------------------------------------------------------

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Identify categorical columns
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Identify numerical columns
    numerical_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    st.subheader("Machine Learning Setup")

    st.write(
        "**Target variable:** quality"
    )

    st.write(
        "**Input variables:**"
    )

    st.write(
        X.columns.tolist()
    )

    st.write(
        "**Machine Learning Task:** Classification"
    )

    st.write(
        "**Train/Test Split:** 75% training and 25% testing"
    )

    st.write(
        "**Evaluation Metric:** Accuracy Score"
    )

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    transformers = []

    if numerical_columns:

        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numerical_columns
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = {

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000
        ),

        "SVM": SVC(
            kernel="rbf"
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    results = []

    trained_models = {}

    progress = st.progress(0)

    for index, (model_name, model) in enumerate(
        models.items()
    ):

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results.append({
            "Algorithm": model_name,
            "Accuracy": accuracy
        })

        trained_models[model_name] = pipeline

        progress.progress(
            (index + 1) / len(models)
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df["Accuracy (%)"] = (
        results_df["Accuracy"] * 100
    ).round(2)

    st.subheader("Model Accuracy Results")

    st.dataframe(
        results_df[
            ["Algorithm", "Accuracy (%)"]
        ],
        use_container_width=True
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("Algorithm vs Accuracy")

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        results_df["Algorithm"],
        results_df["Accuracy (%)"]
    )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Comparison of Machine Learning Algorithms"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    st.pyplot(fig)

    # --------------------------------------------------------
    # BEST ACCURACY FOR PROJECT CONCLUSION
    # --------------------------------------------------------

    best_index = results_df["Accuracy"].idxmax()

    best_algorithm = results_df.loc[
        best_index,
        "Algorithm"
    ]

    best_accuracy = results_df.loc[
        best_index,
        "Accuracy (%)"
    ]

    st.subheader("Model Evaluation Summary")

    st.write(
        f"Among the five algorithms tested, "
        f"**{best_algorithm}** produced the highest test accuracy "
        f"in this run, with an accuracy of **{best_accuracy}%**."
    )

    st.write(
        "The comparison is based on the 75:25 train-test split "
        "and accuracy score required by the project."
    )

    # --------------------------------------------------------
    # PREDICTION DEMO
    # --------------------------------------------------------

    st.subheader("Wine Quality Prediction Demo")

    st.write(
        "Enter the values below to predict the wine quality "
        "using the Random Forest model."
    )

    prediction_model = trained_models["Random Forest"]

    input_data = {}

    input_columns = X.columns.tolist()

    for column in input_columns:

        if column in categorical_columns:

            options = (
                df[column]
                .dropna()
                .unique()
                .tolist()
            )

            input_data[column] = st.selectbox(
                column.title(),
                options
            )

        else:

            min_value = float(
                df[column].min()
            )

            max_value = float(
                df[column].max()
            )

            mean_value = float(
                df[column].mean()
            )

            input_data[column] = st.number_input(
                column.title(),
                min_value=min_value,
                max_value=max_value,
                value=mean_value
            )

    if st.button("Predict Wine Quality"):

        input_df = pd.DataFrame(
            [input_data]
        )

        prediction = prediction_model.predict(
            input_df
        )

        st.success(
            f"Predicted Wine Quality Score: {prediction[0]}"
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")
st.sidebar.write(
    "Wine Quality Project | Alcoholic Beverage Sector"
)