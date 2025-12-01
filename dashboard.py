import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Airbnb Recommender Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF5A5F;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .listing-card {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #ffffff;
    }
    .rating-badge {
        font-size: 1.5rem;
        font-weight: bold;
        color: #FF5A5F;
    }
    </style>
""", unsafe_allow_html=True)

# Data generation functions
@st.cache_data
def load_data():
    """Generate realistic dummy data for the dashboard"""
    
    # Mock user metadata
    user_metadata = {
        'user_1': {'history': 12, 'avg_rating': 4.8, 'name': 'Sarah M.'},
        'user_2': {'history': 8, 'avg_rating': 4.5, 'name': 'John D.'},
        'user_3': {'history': 25, 'avg_rating': 4.9, 'name': 'Emily R.'},
        'user_4': {'history': 5, 'avg_rating': 4.2, 'name': 'Michael T.'},
        'user_5': {'history': 18, 'avg_rating': 4.7, 'name': 'Lisa K.'}
    }
    
    # Mock neighborhoods
    neighborhoods = [
        'Downtown', 'Midtown', 'Brooklyn Heights', 'SoHo', 'Greenwich Village',
        'Upper East Side', 'Chelsea', 'Williamsburg', 'East Village', 'Tribeca'
    ]
    
    # Mock amenities
    amenities_list = [
        'WiFi, Kitchen, Washer',
        'WiFi, Air Conditioning, Pool',
        'WiFi, Parking, Gym',
        'WiFi, Kitchen, Balcony',
        'WiFi, Hot Tub, Fireplace',
        'WiFi, Kitchen, Workspace',
        'WiFi, Pool, BBQ Grill',
        'WiFi, Kitchen, Garden',
        'WiFi, Parking, Pet Friendly',
        'WiFi, Kitchen, Ocean View'
    ]
    
    # Mock listing names
    listing_names = [
        'Cozy Downtown Loft',
        'Modern Midtown Apartment',
        'Charming Brooklyn Studio',
        'Luxury SoHo Penthouse',
        'Quaint Village Townhouse',
        'Spacious Upper East Side',
        'Stylish Chelsea Condo',
        'Trendy Williamsburg Loft',
        'Historic East Village',
        'Elegant Tribeca Duplex'
    ]
    
    return user_metadata, neighborhoods, amenities_list, listing_names

def generate_recommendations(user_id, n_recommendations=10):
    """Generate mock recommendations for a user"""
    _, neighborhoods, amenities_list, listing_names = load_data()
    
    # Generate recommendations with some randomness
    np.random.seed(hash(user_id) % 1000)  # Consistent per user
    
    recommendations = []
    for i in range(n_recommendations):
        # Generate realistic predicted ratings (4.0 - 5.0 range)
        base_rating = 4.3 + np.random.random() * 0.7
        
        # Generate realistic prices ($50 - $500)
        price = np.random.randint(50, 500)
        
        rec = {
            'Listing Name': listing_names[i % len(listing_names)],
            'Neighborhood': neighborhoods[i % len(neighborhoods)],
            'Price': f'${price}',
            'Predicted Rating': round(base_rating, 1),
            'Amenities': amenities_list[i % len(amenities_list)]
        }
        recommendations.append(rec)
    
    # Sort by predicted rating (descending)
    recommendations.sort(key=lambda x: x['Predicted Rating'], reverse=True)
    
    return pd.DataFrame(recommendations)

def get_feature_importance_data():
    """Get feature importance data (based on actual model)"""
    feature_importance = {
        'user_avg_x_item_avg': 72.3,
        'user_avg_rating': 9.6,
        'user_max_rating': 2.5,
        'item_max_rating': 2.5,
        'item_avg_rating': 2.0,
        'user_min_rating': 1.9,
        'item_min_rating': 1.0,
        'item_avg_x_review_score': 1.0,
        'user_rating_std': 0.7,
        'item_rating_std': 0.6,
        'item_id': 0.6,
        'item_review_count': 0.5,
        'host_is_superhost': 0.4,
        'minimum_nights': 0.3,
        'room_type_encoded': 0.3,
        'latitude': 0.3,
        'review_score_composite': 0.2,
        'number_of_reviews': 0.2,
        'price_per_person': 0.2,
        'review_scores_value': 0.2
    }
    
    return pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance (%)': list(feature_importance.values())
    }).sort_values('Importance (%)', ascending=True)

# Load data
user_metadata, neighborhoods, amenities_list, listing_names = load_data()

# Sidebar
st.sidebar.title("🏠 Airbnb Recommender")
st.sidebar.markdown("---")

# User selection
user_ids = list(user_metadata.keys())
selected_user = st.sidebar.selectbox(
    "Select User ID",
    options=user_ids,
    format_func=lambda x: f"{user_metadata[x]['name']} ({x})"
)

# Display user metadata
st.sidebar.markdown("### User Profile")
user_info = user_metadata[selected_user]
st.sidebar.markdown(f"**Name:** {user_info['name']}")
st.sidebar.markdown(f"**History:** {user_info['history']} stays")
st.sidebar.markdown(f"**Avg Rating Given:** {user_info['avg_rating']:.1f} ⭐")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard showcases recommendations powered by XGBoost machine learning model. "
    "The model predicts user ratings for Airbnb listings based on 35+ engineered features."
)

# Main content
st.markdown('<h1 class="main-header">🏠 Airbnb Recommendation Dashboard</h1>', unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📋 User Recommendations", "📊 Model Performance"])

# Tab 1: User Recommendations
with tab1:
    st.header(f"Top 10 Picks for {user_metadata[selected_user]['name']}")
    st.markdown(f"*Personalized recommendations based on your booking history and preferences*")
    
    # Generate recommendations
    recommendations_df = generate_recommendations(selected_user, n_recommendations=10)
    
    # Display top 5 as cards
    st.subheader("🌟 Top 5 Recommendations")
    
    for idx in range(min(5, len(recommendations_df))):
        rec = recommendations_df.iloc[idx]
        
        # Create card layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Image placeholder
            st.image(
                "https://via.placeholder.com/200x150/FF5A5F/FFFFFF?text=Listing",
                use_container_width=True,
                caption=f"#{idx + 1}"
            )
        
        with col2:
            # Listing details
            st.markdown(f"### {rec['Listing Name']}")
            
            # Rating with stars
            rating = rec['Predicted Rating']
            stars = "⭐" * int(rating)
            st.markdown(f"**Predicted Rating:** <span class='rating-badge'>{rating}/5.0</span> {stars}", unsafe_allow_html=True)
            
            # Other details
            col_price, col_location = st.columns(2)
            with col_price:
                st.metric("Price per Night", rec['Price'])
            with col_location:
                st.markdown(f"**📍 Location:** {rec['Neighborhood']}")
            
            st.markdown(f"**🏡 Amenities:** {rec['Amenities']}")
            
            # Book button (mock)
            st.button(f"View Details", key=f"book_{idx}", use_container_width=True)
        
        st.markdown("---")
    
    # Display full table (top 10)
    st.subheader("📊 Complete Recommendation List")
    st.dataframe(
        recommendations_df,
        use_container_width=True,
        hide_index=True
    )

# Tab 2: Model Performance
with tab2:
    st.header("XGBoost Model Performance Metrics")
    
    # Row 1: Key Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="RMSE",
            value="0.905",
            delta="-0.095 from target",
            delta_color="normal",
            help="Root Mean Squared Error on 5-star scale"
        )
    
    with col2:
        st.metric(
            label="MAE",
            value="0.685",
            delta="Excellent",
            delta_color="normal",
            help="Mean Absolute Error"
        )
    
    with col3:
        st.metric(
            label="Training Time",
            value="2.3 min",
            delta="Optimized",
            delta_color="normal",
            help="Time to train 200 boosting rounds"
        )
    
    st.markdown("---")
    
    # Row 2: Feature Importance Chart
    st.subheader("Feature Importance Analysis")
    st.markdown("*Top 20 features ranked by importance in the XGBoost model*")
    
    feature_importance_df = get_feature_importance_data()
    top_features = feature_importance_df.tail(20)  # Top 20
    
    # Create horizontal bar chart using Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_features['Importance (%)'],
        y=top_features['Feature'],
        orientation='h',
        marker=dict(
            color=top_features['Importance (%)'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Importance (%)")
        ),
        text=[f"{val:.1f}%" for val in top_features['Importance (%)']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Feature Importance (Top 20)",
        xaxis_title="Importance (%)",
        yaxis_title="Feature",
        height=600,
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional model info
    st.markdown("---")
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Hyperparameters:**
        - N Estimators: 200
        - Max Depth: 5
        - Learning Rate: 0.05
        - Subsample: 0.85
        - Col Sample by Tree: 0.85
        """)
    
    with col2:
        st.markdown("""
        **Model Details:**
        - Total Features: 35
        - Algorithm: XGBoost Regressor
        - Objective: Squared Error
        - Regularization: L1 (0.1) + L2 (1.0)
        """)
    
    # Feature importance table
    st.markdown("---")
    st.subheader("Complete Feature Importance Table")
    st.dataframe(
        feature_importance_df.sort_values('Importance (%)', ascending=False),
        use_container_width=True,
        hide_index=True
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Powered by XGBoost | Model trained on Airbnb dataset | Last updated: " + 
    datetime.now().strftime("%B %Y") +
    "</div>",
    unsafe_allow_html=True
)

