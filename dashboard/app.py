import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Loading saved .pkl model files
import json # Loading feature_columns.json file
import os   # Finding file paths on your computer

# 1. PATH SETUP (always relative to app.py location)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__)) # folder where app.py  lives
DATA_PATH  = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 2. PAGE CONFIGURATION
st.set_page_config(
    page_title="Automotive Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# 3. LOAD DATA AND MODELS WITH CACHING
@st.cache_data
def load_data():
    """Load and return cleaned dataset with engineered features."""
    df = pd.read_csv(DATA_PATH)
    if 'hp_per_cylinder' not in df.columns:
        df['hp_per_cylinder'] = df['Engine HP'] / df['Engine Cylinders']
        df['fuel_economy']    = (df['highway MPG'] + df['city mpg']) / 2
        df['car_age']         = 2017 - df['Year']
    return df

@st.cache_resource
def load_models():
    """Load all trained models from disk — not retrained inside app."""
    reg_model = joblib.load(os.path.join(MODELS_DIR, 'regression_model.pkl'))
    clf_model = joblib.load(os.path.join(MODELS_DIR, 'classification_model.pkl'))
    scaler    = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    kmeans    = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    pca       = joblib.load(os.path.join(MODELS_DIR, 'pca_model.pkl'))
    with open(os.path.join(MODELS_DIR, 'feature_columns.json'), 'r') as f:
        feature_cols = json.load(f)
    return reg_model, clf_model, scaler, kmeans, pca, feature_cols

# Calling the function
df           = load_data()
reg_model, clf_model, scaler, kmeans, pca, feature_cols = load_models()
num_cols_all = df.select_dtypes(include='number').columns.tolist()

# 4. SIDEBAR BRANDING
st.sidebar.title("🚗 Automotive Price Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Car Features and MSRP (Kaggle)")
st.sidebar.markdown("**Regression Target:** MSRP ($)")
st.sidebar.markdown("**Classification Target:** Vehicle Size")
st.sidebar.markdown("---")
st.sidebar.markdown("**Authors:** Husnain Maroof | Huzaifa Shaheed")
st.sidebar.markdown("**Course:** MCT-341L — Data Science & Analytics")
st.sidebar.markdown("**Instructor:** Dr. Muhammad Ahsan Naeem")
st.sidebar.markdown("**Department:** Mechatronics & Control Engineering")
st.sidebar.markdown("**University:** UET Lahore")
st.sidebar.markdown("---")
st.sidebar.markdown("Use the tabs above to navigate between sections.")

# TABS
tab_a, tab_b, tab_c, tab_d = st.tabs([
    "📊 Tab A — Dataset Overview",
    "🔍 Tab B — Exploratory Analysis",
    "🎯 Tab C — Model Prediction",
    "🔵 Tab D — Cluster Explorer"
])

# TAB A — DATASET OVERVIEW
with tab_a:
    st.header("📊 Dataset Overview")
    st.markdown("Explore the cleaned automotive dataset used for modelling.")

    # 5. Metrics Row 
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows",     f"{df.shape[0]:,}")
    col2.metric("Total Features", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("---")

    # 6. Dataframe 
    st.subheader("Cleaned Dataset")
    st.caption("Fully cleaned dataset after preprocessing, encoding and feature engineering. Click any column header to sort.")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")

    # 7. Column Distribution 
    st.subheader("Column Distribution")
    selected_col = st.selectbox(
        "Select a column to visualise its distribution:",
        df.columns.tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    if df[selected_col].dtype in ['int64', 'float64']:
        sns.histplot(df[selected_col].dropna(), kde=True,
                     color='steelblue', ax=ax)
        ax.set_xlabel(f"{selected_col}")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {selected_col}")
    else:
        counts = df[selected_col].value_counts().head(15)
        ax.bar(counts.index.astype(str), counts.values,
               color='steelblue', edgecolor='white')
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Count")
        ax.set_title(f"Top Categories in {selected_col}")
        plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(f"Distribution of **{selected_col}** — inspect the shape, spread and frequency of values in this column.")

    st.markdown("---")

    # 8. Correlation Heatmap 
    st.subheader("Pearson Correlation Matrix")
    corr = df[num_cols_all].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", square=True, ax=ax2,
                linewidths=0.5, vmin=-1, vmax=1)
    ax2.set_title("Pearson Correlation Matrix — All Numerical Features", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig2)
    st.caption("Red = strong positive correlation, Blue = strong negative. Upper triangle masked to avoid redundancy.")

# TAB B — EXPLORATORY ANALYSIS 
with tab_b:
    st.header("🔍 Exploratory Analysis")
    st.markdown("Interactively explore relationships between any two features in the dataset.")

    # 9. Feature Selection
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select X axis feature:", num_cols_all, index=3)
    with col2:
        y_default = num_cols_all.index('MSRP') if 'MSRP' in num_cols_all else 1
        y_feature = st.selectbox("Select Y axis feature:", num_cols_all, index=y_default)

    # 10. Guard against same X and Y ───────────────────────
    if x_feature == y_feature:
        st.warning("⚠️ Please select two different features for X and Y axes.")
    else:
        col3, col4 = st.columns(2)
        with col3:
            
            # 11. Trend line checkbox 
            show_trend = st.checkbox("Show trend line", value=False)
        with col4:
            
            # 12. Colour by 
            color_options = ['None'] + num_cols_all
            color_by = st.selectbox("Colour points by:", color_options)

        # 13. Filter slider 
        filter_col = st.selectbox(
            "Filter data by feature:",
            num_cols_all,
            index=num_cols_all.index('MSRP') if 'MSRP' in num_cols_all else 0
        )
        min_val = float(df[filter_col].min())
        max_val = float(df[filter_col].max())
        filter_range = st.slider(
            f"Filter range for {filter_col}:",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )

        df_filtered = df[
            (df[filter_col] >= filter_range[0]) &
            (df[filter_col] <= filter_range[1])
        ]
        st.caption(f"Showing **{len(df_filtered):,}** of {len(df):,} rows after filtering on {filter_col}.")

        # 14. Scatter Plot 
        fig, ax = plt.subplots(figsize=(10, 6))
        if color_by != 'None':
            sc = ax.scatter(
                df_filtered[x_feature], df_filtered[y_feature],
                c=df_filtered[color_by], cmap='viridis',
                alpha=0.5, s=10
            )
            plt.colorbar(sc, ax=ax, label=color_by)
        else:
            ax.scatter(
                df_filtered[x_feature], df_filtered[y_feature],
                color='steelblue', alpha=0.5, s=10
            )

        if show_trend and len(df_filtered) > 1:
            try:
                coeffs   = np.polyfit(df_filtered[x_feature], df_filtered[y_feature], 1)
                trend_fn = np.poly1d(coeffs)
                x_sorted = np.sort(df_filtered[x_feature])
                ax.plot(x_sorted, trend_fn(x_sorted),
                        color='red', linewidth=2, label='Trend Line')
                ax.legend()
            except Exception:
                pass

        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f"{x_feature} vs {y_feature}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(f"Scatter plot of **{x_feature}** vs **{y_feature}** — use controls above to filter, colour and add trend line.")

# TAB C — MODEL PREDICTION 
with tab_c:
    st.header("🎯 Model Prediction")
    st.markdown("Adjust the car specifications below and click **Predict** to get MSRP price and Vehicle Size predictions from trained models.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔧 Engine & Performance")
        engine_hp   = st.slider("Engine HP (hp)",              55,   495,  200)
        engine_cyl  = st.slider("Engine Cylinders (count)",     3,    16,    6)
        highway_mpg = st.slider("highway MPG (miles/gallon)",  12,    42,   26)
        city_mpg    = st.slider("city mpg (miles/gallon)",      7,    32,   18)

    with col2:
        st.subheader("📅 Vehicle Details")
        year      = st.slider("Model Year",              1990, 2017, 2015)
        num_doors = st.slider("Number of Doors (count)",    2,    4,    4)
        popularity = st.slider("Popularity Score",           2, 4199, 1000)
        transmission = st.selectbox(
            "Transmission Type",
            ["AUTOMATIC", "MANUAL", "AUTOMATED_MANUAL"]
        )

    with col3:
        st.subheader("🚙 Drive & Brand")
        driven_wheels = st.selectbox(
            "Driven Wheels",
            ["front wheel drive", "rear wheel drive",
             "all wheel drive",   "four wheel drive"]
        )
        make_freq          = st.slider("Make Frequency (brand size in dataset)",   3, 1123,  400)
        fuel_type_freq     = st.slider("Engine Fuel Type Frequency",               2, 6959, 4000)
        vehicle_style_freq = st.slider("Vehicle Style Frequency",                 29, 2914, 1000)

    # 15. Compute Engineered Features automatically 
    hp_per_cyl   = engine_hp / engine_cyl if engine_cyl > 0 else 0
    fuel_economy = (highway_mpg + city_mpg) / 2
    car_age      = 2017 - year

    # 16. One-Hot Encoding
    trans_auto   = 1 if transmission    == "AUTOMATIC"         else 0
    trans_manual = 1 if transmission    == "MANUAL"            else 0
    dw_four      = 1 if driven_wheels   == "four wheel drive"  else 0
    dw_front     = 1 if driven_wheels   == "front wheel drive" else 0
    dw_rear      = 1 if driven_wheels   == "rear wheel drive"  else 0

    # 17. Assemble Feature Vector (must match training column order) 
    input_values = {
        'Make':                           make_freq,
        'Year':                           year,
        'Engine Fuel Type':               fuel_type_freq,
        'Engine HP':                      engine_hp,
        'Engine Cylinders':               engine_cyl,
        'Number of Doors':                num_doors,
        'Vehicle Style':                  vehicle_style_freq,
        'highway MPG':                    highway_mpg,
        'city mpg':                       city_mpg,
        'Popularity':                     popularity,
        'Transmission Type_AUTOMATIC':    trans_auto,
        'Transmission Type_MANUAL':       trans_manual,
        'Driven_Wheels_four wheel drive': dw_four,
        'Driven_Wheels_front wheel drive':dw_front,
        'Driven_Wheels_rear wheel drive': dw_rear,
        'hp_per_cylinder':                hp_per_cyl,
        'fuel_economy':                   fuel_economy,
        'car_age':                        car_age,
    }

    input_df = pd.DataFrame([input_values])[feature_cols]

    st.markdown("---")

    # 18. Predict Button 
    if st.button("🚗 Predict MSRP and Vehicle Size", type="primary"):
        try:
            input_scaled = scaler.transform(input_df)

            # Regression prediction
            reg_pred = reg_model.predict(input_scaled)[0]

            # Classification prediction
            clf_pred    = clf_model.predict(input_scaled)[0]
            clf_prob    = clf_model.predict_proba(input_scaled)[0]
            size_map    = {0: 'Compact', 1: 'Midsize', 2: 'Large'}
            pred_size   = size_map[int(clf_pred)]

            col1, col2 = st.columns(2)

            # 19. Regression result with colour coding 
            with col1:
                st.subheader("💰 Predicted MSRP")
                if reg_pred < 25000:
                    st.success(f"**${reg_pred:,.2f}**")
                    st.caption("Economy range — affordable vehicle (under $25,000)")
                elif reg_pred < 50000:
                    st.warning(f"**${reg_pred:,.2f}**")
                    st.caption("Mid-range vehicle ($25,000 – $50,000)")
                else:
                    st.error(f"**${reg_pred:,.2f}**")
                    st.caption("Premium range — expensive vehicle (above $50,000)")

            # 20. Classification result with colour coding
            with col2:
                st.subheader("📏 Predicted Vehicle Size")
                if pred_size == 'Compact':
                    st.success(f"**{pred_size}** 🟢")
                    st.caption("Small, fuel-efficient economy vehicle")
                elif pred_size == 'Midsize':
                    st.warning(f"**{pred_size}** 🟡")
                    st.caption("Medium-sized balanced vehicle")
                else:
                    st.error(f"**{pred_size}** 🔴")
                    st.caption("Large, powerful vehicle")

            st.markdown("---")

            # 21. Classification probabilities as bar chart 
            st.subheader("📊 Classification Confidence")
            prob_df = pd.DataFrame({
                'Vehicle Size': ['Compact', 'Midsize', 'Large'],
                'Probability':  np.round(clf_prob, 4)
            }).set_index('Vehicle Size')
            st.bar_chart(prob_df)
            st.caption("Model confidence for each Vehicle Size class — the tallest bar is the predicted class.")

            st.markdown("---")

            # 22. Computed engineered features display
            st.subheader("⚙️ Automatically Computed Features")
            c1, c2, c3 = st.columns(3)
            c1.metric("HP per Cylinder",    f"{hp_per_cyl:.2f} hp/cyl")
            c2.metric("Fuel Economy Score", f"{fuel_economy:.1f} mpg")
            c3.metric("Car Age",            f"{car_age} years")
            st.caption("These features are derived automatically from your inputs and fed into the model.")

        except Exception as e:
            st.error(f"Prediction error — please check inputs: {e}")

# TAB D — CLUSTER EXPLORER
with tab_d:
    st.header("🔵 Cluster Explorer")
    st.markdown("Explore the three K-Means clusters discovered in the automotive dataset — without using Vehicle Size labels.")

    # 23. Compute clusters for full dataset 
    X_full        = df[feature_cols].copy()
    X_full_scaled = scaler.transform(X_full)
    cluster_labels = kmeans.predict(X_full_scaled)
    X_pca          = pca.transform(X_full_scaled)
    centroids_pca  = pca.transform(kmeans.cluster_centers_)

    # 24. Cluster selection
    selected_cluster = st.selectbox(
        "Select a cluster to highlight:",
        options=[0, 1, 2],
        format_func=lambda x: f"Cluster {x}"
    )

    # 25. PCA Scatter with centroids 
    cluster_colors = ['steelblue', 'orange', 'green']
    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(3):
        mask = cluster_labels == i
        if i == selected_cluster:
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=cluster_colors[i], alpha=0.7, s=20,
                       label=f'Cluster {i} (selected)')
        else:
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c='lightgray', alpha=0.2, s=10,
                       label=f'Cluster {i}')

    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c='red', marker='*', s=400,
               edgecolors='black', linewidth=1.5,
               label='Centroids', zorder=5)

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title(f"K-Means Clustering (k=3) — Cluster {selected_cluster} Highlighted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(f"Cluster **{selected_cluster}** shown in colour. Other clusters greyed out. Red ★ = cluster centroids (PCA projected).")

    st.markdown("---")

    # 26. Comparison Table 
    st.subheader(f"Cluster {selected_cluster} vs Overall Dataset")

    cluster_mask       = cluster_labels == selected_cluster
    overall_means      = df[num_cols_all].mean()
    cluster_means_vals = df[num_cols_all][cluster_mask].mean()
    delta              = cluster_means_vals - overall_means

    comparison_df = pd.DataFrame({
        'Overall Mean':                   overall_means.round(3),
        f'Cluster {selected_cluster} Mean': cluster_means_vals.round(3),
        'Delta (Cluster − Overall)':      delta.round(3)
    })
    st.dataframe(comparison_df, use_container_width=True)
    st.caption(f"Positive delta = Cluster {selected_cluster} is above dataset average. Negative = below average.")

    st.markdown("---")

    # 27. Delta Bar Chart 
    st.subheader(f"What Makes Cluster {selected_cluster} Distinctive")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    bar_colors = ['green' if x > 0 else 'red' for x in delta.values]
    ax2.barh(delta.index, delta.values, color=bar_colors)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel("Delta (Cluster Mean − Overall Mean)")
    ax2.set_title(f"Feature Delta — Cluster {selected_cluster} vs Overall Dataset")
    plt.tight_layout()
    st.pyplot(fig2)
    st.caption(f"**Green** bars = Cluster {selected_cluster} is above dataset average for that feature. **Red** = below average. Longer bar = more distinctive.")