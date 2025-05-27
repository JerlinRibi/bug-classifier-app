import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# Streamlit UI setup
st.set_page_config(page_title="Bug Classifier", layout="wide")
st.title("🪲 Bug Classification & Logic-Level Issue Analyzer")

uploaded_file = st.file_uploader("📂 Upload Bug Report (.xlsx or .csv)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read uploaded file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        df.columns = df.columns.str.strip()

        # Try again if 'Details' not found
        if "Details" not in df.columns:
            df = pd.read_excel(uploaded_file, skiprows=2, engine="openpyxl")
            df.columns = df.columns.str.strip()

        st.write("📋 Columns found in your file:", df.columns.tolist())

        if "Details" not in df.columns:
            st.error("❌ The uploaded file must contain a 'Details' column.")
        else:
            df = df.dropna(subset=["Details"])
            df["Details"] = df["Details"].astype(str)

            # Add Bug ID if missing
            if "S.No" in df.columns:
                df.rename(columns={"S.No": "Bug ID"}, inplace=True)
            elif "Bug ID" not in df.columns:
                df["Bug ID"] = range(1, len(df) + 1)

            # Load model and vectorizer
            model, vectorizer = load_model()
            X = vectorizer.transform(df["Details"])
            preds = model.predict(X)
            probs = model.predict_proba(X).max(axis=1)

            # Map numeric categories to labels if necessary
            category_map = {0: "UI", 1: "API", 2: "DB", 3: "Server", 4: "Other"}
            if pd.api.types.is_numeric_dtype(preds):
                df["Predicted Category"] = [category_map.get(int(p), str(p)) for p in preds]
            else:
                df["Predicted Category"] = preds

            df["Confidence Score"] = [round(p, 2) for p in probs]

            # Use existing logic-level issue column or fallback
            if "Logical_Issue" in df.columns:
                df["Logic-level Issue"] = df["Logical_Issue"]
            else:
                df["Logic-level Issue"] = "N/A"

            # Pie chart – bug origin by type with better label spacing and legend
            st.markdown("### 🥧 Pie Chart – Bug Origin by Type")
            bug_counts = df['bug_type'].value_counts()

            fig1, ax1 = plt.subplots(figsize=(8,8))
            wedges, texts, autotexts = ax1.pie(
               bug_counts,
               autopct='%1.1f%%',
               startangle=90,
               pctdistance=0.7,
               labeldistance=1.2,
               colors=sns.color_palette("pastel"),
            )

            # Percentage texts (autotexts) 
            for autotext in autotexts:
                autotext.set_fontsize(8)   

            ax1.axis('equal')
            ax1.legend(wedges, bug_counts.index, title="Bug Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            st.pyplot(fig1)

            # Bar chart – logic-level bugs by type
            st.markdown("### 📊 Bar Chart – Logic-Level Bugs by Category")
            logic_bugs = df[df['Logical_Issue'].astype(str).str.lower() == 'yes']
            logic_counts = logic_bugs['bug_type'].value_counts()
            st.bar_chart(logic_counts)

            # 📑 TABLE OUTPUT
            st.subheader("📑 Table – Bug Classification Results")
            table_cols = ["Bug ID", "Details", "Predicted Category", "Logic-level Issue", "Confidence Score"]
            st.dataframe(df[table_cols].reset_index(drop=True))

    except Exception as e:
        st.error(f"❌ Error: {e}")
