import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import xgboost as xgb
import json
import os
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Airbnb Recommender Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Futuristic Theme (keeping the existing styling)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Color Variables */
    :root {
        --bg-primary: #2C003E;
        --bg-secondary: #1a0025;
        --accent-violet: #5A189A;
        --accent-pink: #FF006E;
        --accent-purple: #8338EC;
        --accent-blue: #3A86FF;
        --accent-yellow: #FFBE0B;
        --text-primary: #FFFFFF;
        --text-secondary: #E0E0E0;
        --text-muted: #9D8BA7;
    }

    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #2C003E 0%, #1a0025 50%, #0d0015 100%);
    }

    /* Main header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF006E 0%, #8338EC 50%, #3A86FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-shadow: 0 0 30px rgba(255, 0, 110, 0.3);
    }

    .sub-header {
        color: #E0E0E0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #FFFFFF;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, #FF006E, #8338EC, #3A86FF) 1;
        display: inline-block;
    }

    /* Card styling */
    .listing-card {
        background: linear-gradient(145deg, rgba(90, 24, 154, 0.3) 0%, rgba(44, 0, 62, 0.8) 100%);
        border: 1px solid rgba(131, 56, 236, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 30px rgba(131, 56, 236, 0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }

    .listing-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 40px rgba(255, 0, 110, 0.3);
        border-color: rgba(255, 0, 110, 0.5);
    }

    /* Rating badge */
    .rating-badge {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FFBE0B;
        background: linear-gradient(135deg, rgba(255, 190, 11, 0.2) 0%, rgba(255, 0, 110, 0.2) 100%);
        padding: 0.4rem 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 190, 11, 0.3);
        display: inline-block;
        text-shadow: 0 0 10px rgba(255, 190, 11, 0.5);
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2.4rem;
        font-weight: 700;
        color: #FFFFFF !important;
        text-shadow: 0 0 20px rgba(58, 134, 255, 0.5);
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
        color: #8338EC !important;
    }

    [data-testid="stMetricLabel"] {
        color: #E0E0E0 !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2C003E 0%, #1a0025 100%);
        border-right: 1px solid rgba(131, 56, 236, 0.3);
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #E0E0E0;
    }

    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: rgba(90, 24, 154, 0.3) !important;
        border: 1px solid rgba(131, 56, 236, 0.5) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(90deg, rgba(90, 24, 154, 0.3) 0%, rgba(44, 0, 62, 0.5) 100%);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid rgba(131, 56, 236, 0.3);
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        padding: 0 24px;
        font-weight: 500;
        color: #E0E0E0 !important;
        background: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF006E 0%, #8338EC 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 20px rgba(255, 0, 110, 0.4);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #5A189A 0%, #8338EC 100%);
        color: white;
        border: 1px solid rgba(131, 56, 236, 0.5);
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #FF006E 0%, #8338EC 100%);
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(255, 0, 110, 0.5);
        border-color: #FF006E;
    }

    /* Info/Success boxes */
    .stAlert {
        background: linear-gradient(135deg, rgba(90, 24, 154, 0.4) 0%, rgba(44, 0, 62, 0.6) 100%) !important;
        border: 1px solid rgba(131, 56, 236, 0.4) !important;
        border-radius: 12px !important;
        color: #E0E0E0 !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(131, 56, 236, 0.3);
    }

    /* Text colors */
    .stMarkdown, .stText, p, span {
        color: #E0E0E0;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1a0025;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #5A189A, #8338EC);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #8338EC, #FF006E);
    }
    </style>
""", unsafe_allow_html=True)


# Data loading functions
@st.cache_data
def load_model_artifacts():
    """Load trained model artifacts"""
    try:
        model_path = 'data/xgboost_model_optimized/'

        if model_path is None:
            st.error("Could not find trained model artifacts. Please run train_clean_model.py first.")
            return None, None, None

        # Load model info
        with open(f"{model_path}model_info.json", 'r') as f:
            model_info = json.load(f)

        # Load feature importance
        feature_importance = pd.read_csv(f"{model_path}feature_importance.csv")

        # Load the actual XGBoost model
        model = xgb.XGBRegressor()
        model.load_model(f"{model_path}xgboost_model.json")

        return model, model_info, feature_importance

    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None


@st.cache_data
def load_listings_data():
    """Load listings data from parquet file"""
    try:
        df = pd.read_parquet('data/listings.parquet')
        # Clean up the data
        df = df.dropna(subset=['name'])
        return df

    except Exception as e:
        st.error(f"Error loading listings data: {e}")
        return pd.DataFrame()


@st.cache_data
def load_train_data():
    """Load training data to get user information"""
    try:
        df = pd.read_parquet('data/train.parquet')
        return df

    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return pd.DataFrame()


@st.cache_data
def load_user_mapping():
    """Load user ID mapping"""
    try:
        df = pd.read_csv('data/user_id_mapping.csv')
        return df

    except Exception as e:
        st.warning(f"Could not load user mapping: {e}")
        return pd.DataFrame()


def prepare_features_for_prediction(listings_df, train_df, user_id, item_ids):
    """Prepare features for model prediction"""
    try:
        # Get user stats from training data
        user_stats = train_df[train_df['user_id'] == user_id].agg({
            'item_id': 'count'
        }).rename({'item_id': 'user_review_count'})
        user_review_count = user_stats['user_review_count'] if not pd.isna(user_stats['user_review_count']) else 0

        # Get item stats for each listing
        item_stats = train_df.groupby('item_id').agg({
            'user_id': 'count'
        }).rename(columns={'user_id': 'item_review_count'})

        # Prepare features for each listing
        features_list = []
        for item_id in item_ids:
            listing = listings_df[listings_df['listing_id'] == item_id].iloc[0] if len(
                listings_df[listings_df['listing_id'] == item_id]) > 0 else None

            if listing is None:
                continue

            # Get item review count
            item_review_count = item_stats.loc[item_id, 'item_review_count'] if item_id in item_stats.index else 0

            # Create feature vector
            features = {
                'user_id': user_id,
                'item_id': item_id,
                'user_review_count': user_review_count,
                'item_review_count': item_review_count,
                'price': listing.get('price', 0),
                'accommodates': listing.get('accommodates', 1),
                'bedrooms': listing.get('bedrooms', 0),
                'beds': listing.get('beds', 1),
                'minimum_nights': listing.get('minimum_nights', 1),
                'number_of_reviews': listing.get('number_of_reviews', 0),
                'review_scores_rating': listing.get('review_scores_rating', 0),
                'review_scores_location': listing.get('review_scores_location', 0),
                'review_scores_value': listing.get('review_scores_value', 0),
                'latitude': listing.get('latitude', 0),
                'longitude': listing.get('longitude', 0),
                'host_is_superhost': 1 if listing.get('host_is_superhost') else 0,
                'instant_bookable': 1 if listing.get('instant_bookable') else 0,
            }

            # Add encoded categorical features (simplified encoding)
            property_type_map = {'Entire home/apt': 0, 'Private room': 1, 'Shared room': 2, 'Hotel room': 3}
            room_type_map = {'Entire home/apt': 0, 'Private room': 1, 'Shared room': 2, 'Hotel room': 3}

            features['property_type_encoded'] = property_type_map.get(listing.get('property_type', ''), 0)
            features['room_type_encoded'] = room_type_map.get(listing.get('room_type', ''), 0)
            features['neighbourhood_cleansed_encoded'] = hash(str(listing.get('neighbourhood_cleansed', ''))) % 100

            # Add derived features
            features['price_per_person'] = features['price'] / (features['accommodates'] + 1e-6)
            features['bedroom_ratio'] = features['bedrooms'] / (features['accommodates'] + 1e-6)
            features['bed_ratio'] = features['beds'] / (features['accommodates'] + 1e-6)
            features['review_score_composite'] = (
                    features['review_scores_rating'] * 0.5 +
                    features['review_scores_location'] * 0.3 +
                    features['review_scores_value'] * 0.2
            )

            features_list.append(features)

        return pd.DataFrame(features_list)

    except Exception as e:
        st.error(f"Error preparing features: {e}")
        return pd.DataFrame()


def generate_recommendations_with_model(model, model_info, user_id, listings_df, train_df, n_recommendations=10):
    """Generate recommendations using the trained model"""
    try:
        if model is None or listings_df.empty:
            return []

        # Sample listings for prediction (you could use all listings in production)
        sample_size = min(100, len(listings_df))  # Sample 100 listings for efficiency
        np.random.seed(hash(user_id) % 10000)  # Deterministic sampling per user
        sample_listings = listings_df.sample(n=sample_size)

        # Get item IDs (assuming listing_id maps to item_id)
        item_ids = sample_listings['listing_id'].tolist()

        # Prepare features
        features_df = prepare_features_for_prediction(listings_df, train_df, user_id, item_ids)

        if features_df.empty:
            return []

        # Ensure features match model expectations
        expected_features = model_info['feature_names']
        available_features = [f for f in expected_features if f in features_df.columns]

        # Fill missing features with 0
        for feature in expected_features:
            if feature not in features_df.columns:
                features_df[feature] = 0

        # Reorder columns to match model
        X = features_df[expected_features].fillna(0)

        # Make predictions
        predictions = model.predict(X)
        predictions = np.clip(predictions, 1.0, 5.0)  # Clip to valid rating range

        # Create recommendations
        recommendations = []
        for idx, (_, listing) in enumerate(sample_listings.iterrows()):
            if idx >= len(predictions):
                break

            name = str(listing['name'])
            if len(name) > 60:
                name = name[:57] + '...'

            price = listing['price'] if pd.notna(listing['price']) else 0
            beds = int(listing['beds']) if pd.notna(listing['beds']) and listing['beds'] > 0 else 1
            bedrooms = int(listing['bedrooms']) if pd.notna(listing['bedrooms']) and listing['bedrooms'] > 0 else 0
            review_score = listing['review_scores_rating'] if pd.notna(listing['review_scores_rating']) else 0

            rec = {
                'Listing ID': listing['listing_id'],
                'Listing Name': name,
                'Neighborhood': listing['neighbourhood_cleansed'] if pd.notna(
                    listing['neighbourhood_cleansed']) else 'Unknown',
                'Price': f"${price:.0f}" if price > 0 else 'Contact Host',
                'Predicted Rating': round(float(predictions[idx]), 2),
                'Room Type': listing['room_type'] if pd.notna(listing['room_type']) else 'Unknown',
                'Beds': beds,
                'Bedrooms': bedrooms,
                'Image URL': listing.get('picture_url', ''),
                'Superhost': listing['host_is_superhost'] if pd.notna(listing['host_is_superhost']) else False,
                'Airbnb Rating': f"{review_score:.2f}" if review_score > 0 else 'New',
                'Reviews': int(listing['number_of_reviews']) if pd.notna(listing['number_of_reviews']) else 0
            }
            recommendations.append(rec)

        # Sort by predicted rating (descending)
        recommendations.sort(key=lambda x: x['Predicted Rating'], reverse=True)

        return recommendations[:n_recommendations]

    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return []


def get_user_metadata_from_data(train_df, user_mapping_df):
    """Get user metadata from actual data"""
    try:
        user_stats = train_df.groupby('user_id').agg({
            'rating': ['count', 'mean'],
            'item_id': 'nunique'
        }).round(2)

        user_stats.columns = ['booking_count', 'avg_rating', 'unique_listings']
        user_metadata = {}

        for user_id in user_stats.index:
            stats = user_stats.loc[user_id]

            # Try to get name from mapping, otherwise generate
            if not user_mapping_df.empty and user_id in user_mapping_df['user_id'].values:
                reviewer_id = user_mapping_df[user_mapping_df['user_id'] == user_id]['reviewer_id'].iloc[0]
                name = f"User {reviewer_id}"
            else:
                name = f"User {user_id}"

            user_metadata[user_id] = {
                'name': name,
                'history': int(stats['booking_count']),
                'avg_rating': float(stats['avg_rating']),
                'unique_listings': int(stats['unique_listings'])
            }

        return user_metadata

    except Exception as e:
        st.error(f"Error creating user metadata: {e}")
        return {}


# Load all data
model, model_info, feature_importance = load_model_artifacts()
listings_df = load_listings_data()
train_df = load_train_data()
user_mapping_df = load_user_mapping()

# Get user metadata from actual data
if not train_df.empty:
    user_metadata = get_user_metadata_from_data(train_df, user_mapping_df)
else:
    user_metadata = {}

# Sidebar
st.sidebar.markdown("## Airbnb Recommender")
st.sidebar.markdown("---")

# User selection
if user_metadata:
    user_ids = list(user_metadata.keys())
    st.sidebar.markdown("### Select User")
    selected_user = st.sidebar.selectbox(
        "User ID",
        options=user_ids,
        format_func=lambda x: f"{user_metadata[x]['name']} ({x})",
        label_visibility="collapsed"
    )

    # Display user metadata
    st.sidebar.markdown("---")
    st.sidebar.markdown("### User Profile")
    user_info = user_metadata[selected_user]
    st.sidebar.markdown(f"**Name:** {user_info['name']}")
    st.sidebar.markdown(f"**Booking History:** {user_info['history']} ratings")
    st.sidebar.markdown(f"**Average Rating:** {user_info['avg_rating']:.1f} / 5.0")
    st.sidebar.markdown(f"**Unique Listings:** {user_info['unique_listings']}")
else:
    st.sidebar.error("No user data available. Please check data files.")
    selected_user = None

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Info")
if model_info:
    st.sidebar.success(
        f"WIP"
    )
else:
    st.sidebar.error("❌ **Model Not Loaded**\n\nPlease run XGBoost.ipynb first")

# Main content
st.markdown('<h1 class="main-header">Airbnb Recommendation Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Personalized listing recommendations powered by trained XGBoost model</p>',
            unsafe_allow_html=True)

if model is None or selected_user is None:
    st.error("Cannot generate recommendations. Please ensure the model is trained and user data is available.")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["User Recommendations", "Model Performance"])

# Tab 1: User Recommendations
with tab1:
    st.markdown(f'<p class="section-header">Top 10 Picks for {user_metadata[selected_user]["name"]}</p>',
                unsafe_allow_html=True)
    st.markdown("*Personalized recommendations based on your preferences and booking history*")

    # Generate recommendations using trained model
    with st.spinner("Generating personalized recommendations..."):
        recommendations = generate_recommendations_with_model(
            model, model_info, selected_user, listings_df, train_df, n_recommendations=10
        )

    if not recommendations:
        st.error("Could not generate recommendations. Please check if all data files are available.")
    else:
        # Display all 10 recommendations
        for idx, rec in enumerate(recommendations):
            col1, col2 = st.columns([1, 2])

            with col1:
                # Display image if available
                if rec['Image URL'] and rec['Image URL'].startswith('http'):
                    try:
                        st.image(
                            rec['Image URL'],
                            use_container_width=True,
                            caption=f"Rank #{idx + 1}"
                        )
                    except Exception:
                        st.image(
                            "https://via.placeholder.com/300x200/2C003E/FFFFFF?text=Image+Not+Available",
                            use_container_width=True,
                            caption=f"Rank #{idx + 1}"
                        )
                else:
                    st.image(
                        "https://via.placeholder.com/300x200/2C003E/FFFFFF?text=No+Image",
                        use_container_width=True,
                        caption=f"Rank #{idx + 1}"
                    )

            with col2:
                # Listing name
                st.markdown(f"### {rec['Listing Name']}")

                # Superhost badge
                if rec['Superhost']:
                    st.markdown("**⭐ SUPERHOST**")

                # Rating
                st.markdown(f"**Model Prediction:** <span class='rating-badge'>{rec['Predicted Rating']}/5.0</span>",
                            unsafe_allow_html=True)

                # Details in columns
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Price/Night", rec['Price'])
                with col_b:
                    st.metric("Beds", rec['Beds'])
                with col_c:
                    st.metric("Reviews", rec['Reviews'])

                # Additional info
                st.markdown(f"**Location:** {rec['Neighborhood']}")
                st.markdown(f"**Type:** {rec['Room Type']} | **Airbnb Rating:** {rec['Airbnb Rating']}")

                # View button
                airbnb_url = f"https://www.airbnb.com/rooms/{rec['Listing ID']}"
                st.link_button("🔗 View on Airbnb", airbnb_url, use_container_width=True)

            st.markdown("---")

# Tab 2: Model Performance
with tab2:
    st.markdown('<p class="section-header">Trained XGBoost Model Performance</p>', unsafe_allow_html=True)

    if model_info:
        st.success(f"""
        WIP
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Test RMSE",
                value=f"{model_info['test_rmse']:.3f}",
                delta=f"CV: {model_info.get('cv_rmse', 0):.3f}",
                delta_color="normal",
                help="Root Mean Squared Error on test set"
            )

        with col2:
            st.metric(
                label="Test MAE",
                value=f"{model_info['test_mae']:.3f}",
                delta="No leakage",
                delta_color="normal",
                help="Mean Absolute Error (clean features only)"
            )

        with col3:
            st.metric(
                label="Features",
                value=str(model_info['n_features']),
                delta="100% clean",
                delta_color="normal",
                help=f"{model_info['n_features']} features with no target-derived leakage"
            )

        st.markdown("---")

        # Display hyperparameters
        st.markdown('<p class="section-header">Optimal Hyperparameters</p>', unsafe_allow_html=True)

        if 'optimal_hyperparameters' in model_info:
            col1, col2 = st.columns(2)

            params = model_info['optimal_hyperparameters']
            param_items = list(params.items())
            mid = len(param_items) // 2

            with col1:
                for key, value in param_items[:mid]:
                    st.markdown(f"**{key}:** {value}")

            with col2:
                for key, value in param_items[mid:]:
                    st.markdown(f"**{key}:** {value}")

        st.markdown("---")

        # Feature importance from trained model
        if feature_importance is not None and not feature_importance.empty:
            st.markdown('<p class="section-header">Feature Importance Analysis</p>', unsafe_allow_html=True)
            st.markdown("*Features ranked by importance in the trained XGBoost model*")
            st.info(
                "WIP")

            # Convert importance to percentage
            feature_importance['importance_pct'] = (feature_importance['importance'] / feature_importance[
                'importance'].sum()) * 100
            top_features = feature_importance.head(20)

            # Create horizontal bar chart
            fig = go.Figure()

            # Custom color scale
            colors = []
            for val in top_features['importance_pct']:
                if val > 20:
                    colors.append('#FF006E')  # Bright Pink for top
                elif val > 5:
                    colors.append('#8338EC')  # Neon Purple
                elif val > 2:
                    colors.append('#3A86FF')  # Cyan/Blue
                else:
                    colors.append('#5A189A')  # Deep Violet

            fig.add_trace(go.Bar(
                x=top_features['importance_pct'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
                ),
                text=[f"{val:.1f}%" for val in top_features['importance_pct']],
                textposition='outside',
                textfont=dict(size=11, color='#E0E0E0'),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>'
            ))

            fig.update_layout(
                title=dict(
                    text="Feature Importance (Top 20)",
                    font=dict(size=18, color='#FFFFFF')
                ),
                xaxis=dict(
                    title=dict(text="Importance (%)", font=dict(color='#E0E0E0')),
                    tickfont=dict(color='#E0E0E0'),
                    gridcolor='rgba(131, 56, 236, 0.2)',
                    zerolinecolor='rgba(131, 56, 236, 0.3)'
                ),
                yaxis=dict(
                    title=dict(text="", font=dict(color='#E0E0E0')),
                    tickfont=dict(color='#E0E0E0'),
                    autorange="reversed"
                ),
                height=600,
                template="plotly_dark",
                paper_bgcolor='rgba(44, 0, 62, 0.5)',
                plot_bgcolor='rgba(26, 0, 37, 0.8)',
                hovermode='closest',
                margin=dict(l=20, r=100, t=60, b=40),
                font=dict(family="Outfit, sans-serif")
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown('<p class="section-header">Complete Feature Importance Table</p>', unsafe_allow_html=True)

            display_importance = feature_importance.copy()
            display_importance['Importance (%)'] = display_importance['importance_pct'].round(2)
            display_importance = display_importance[['feature', 'Importance (%)']].rename(
                columns={'feature': 'Feature'})

            st.dataframe(
                display_importance,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.error("Model information not available. Please run train_clean_model.py first.")

# Footer
st.markdown(
    '<div class="footer">'
    f'Powered by Trained XGBoost Model | {len(listings_df):,} Real Listings | Last Updated: ' +
    datetime.now().strftime("%B %Y") +
    '</div>',
    unsafe_allow_html=True
)
