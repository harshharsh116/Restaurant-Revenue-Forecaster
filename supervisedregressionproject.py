import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import streamlit as st
import io


st.markdown("""
    <style>


    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e9fd 100%);
    }

    h1 {
        color: #2c3e50 !important;
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }

    h2, h3 {
        color: #34495e !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.4rem;
        margin-top: 2rem !important;
    }

    /* ── SIDEBAR FIX: Make text visible on dark background ── */
    section[data-testid="stSidebar"] {
        background-color: #2c3e50 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ecf0f1 !important;          /* Light gray/white text */
    }

    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {
        color: #f0f4f8 !important;          
    }

    section[data-testid="stSidebar"] .stButton > button {
        color: white !important;
        background-color: #3498db !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: bold !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #2980b9 !important;
        transform: translateY(-2px) !important;
    }

    .stButton > button {
        background-color: #3498db !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: bold !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }

    .stButton > button:hover {
        background-color: #2980b9 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
    }

    .stPlotlyChart, .element-container[data-testid="stDataFrame"] {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        padding: 1rem;
        margin: 1rem 0;
    }

    .dataframe {
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    hr {
        border-top: 1px solid #bdc3c7 !important;
        margin: 2.5rem 0 !important;
    }

    .prediction-box {
        background: linear-gradient(90deg, #27ae60, #2ecc71);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 6px 20px rgba(39,174,96,0.3);
    }
    </style>
""", unsafe_allow_html=True)



st.title("🍽️ Restaurant Revenue Forecaster")
st.markdown(
    "<h4 style='text-align:center; color:#7f8c8d;'>Predict revenue based on location, cuisine, ratings, marketing & more</h4>",
    unsafe_allow_html=True)



@st.cache_data
def load_data():
    df = pd.read_csv("restaurant_data.csv")
    return df


df = load_data()

with st.sidebar:
    st.markdown("<h3 style='color:#ecf0f1; text-align:center;'>Control Panel</h3>", unsafe_allow_html=True)
    st.markdown("---")

    c1 = st.checkbox("Show Raw Original Data")
    c2 = st.checkbox("Show EDA Report")
    c5 = st.checkbox("Prediction Mode", value=False)

    st.markdown("---")
    st.markdown("<h4 style='color:#bdc3c7;'>Model Selection</h4>", unsafe_allow_html=True)
    model_name = st.selectbox("Select model",
                              ['Linear', 'Ridge', 'Lasso', 'Polynomial', 'Xgboost', 'SVR', 'Elastic net', 'GridSearch'],
                              index=0)

    # Model hyperparameters
    if model_name in ['Ridge', 'Lasso', 'Elastic net']:
        a = st.number_input("Alpha (regularization)", 0.01, 2.0, 0.2, step=0.05)

    if model_name == 'Polynomial':
        d = st.number_input("Polynomial Degree", 1, 10, 2)

    st.markdown("---")
    b1 = st.button("📊 Show Model Results")


if c1:
    st.markdown("<h2 style='text-align:center;'>📋 Original Dataset</h2>", unsafe_allow_html=True)
    st.dataframe(df)

if c2:
    st.markdown("<h2 style='text-align:center; color:#2c3e50;'>📊 Exploratory Data Analysis</h2>",
                unsafe_allow_html=True)
    st.markdown("---")

    # Top / Bottom rows
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align:center;'>First 5 Rows</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(5))
    with col2:
        st.markdown("<h3 style='text-align:center;'>Last 5 Rows</h3>", unsafe_allow_html=True)
        st.dataframe(df.tail(5))

  
    st.markdown("<h3 style='text-align:center;'>Data Information</h3>", unsafe_allow_html=True)
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.markdown("<h3 style='text-align:center;'>Numeric Summary</h3>", unsafe_allow_html=True)
    st.dataframe(df.describe())

    st.markdown("<h3 style='text-align:center;'>Missing Values</h3>", unsafe_allow_html=True)
    st.write(df.isnull().sum())

  
    plots = [
        ("CapacityWise count of restaurants", px.histogram(df, x="Seating Capacity", nbins=10), "pink"),
        ("LocationWise Count of Restaurants",
         px.bar(df.groupby('Location')['Name'].count().reset_index(), x="Location", y="Name"), "pink"),
        ("Cuisine TypeWise Count of Restaurants",
         px.bar(df.groupby('Cuisine')['Name'].count().reset_index(), x="Cuisine", y="Name"), "pink"),
        ("Count of Restaurants By Rating", px.histogram(df, x="Rating", nbins=5), "pink"),
        ("Count of CustomerReviews By Cuisine",
         px.bar(df.groupby('Cuisine')['Number of Reviews'].count().reset_index(), x="Cuisine", y="Number of Reviews"),
         "darkblue"),
        ("Count of Restaurants By Parking Availability In Specific Region",
         px.bar(df.groupby(['Parking Availability', 'Location'])['Name'].count().reset_index(), x="Location", y="Name",
                color="Parking Availability", barmode='stack'), None),
        ("CuisineWise Average Meal Price",
         px.bar(df.groupby('Cuisine')['Average Meal Price'].mean().reset_index(), x="Cuisine", y="Average Meal Price"),
         "brown"),
        ("CuisineWise Average Social Media Followers In Specific Region",
         px.bar(df.groupby(['Cuisine', 'Location'])['Social Media Followers'].mean().reset_index(), x="Cuisine",
                y="Social Media Followers", color="Location"), None),
        ("CuisineWise Average Marketing budget In Specific Region",
         px.bar(df.groupby(['Cuisine', 'Location'])['Marketing Budget'].mean().reset_index(), x="Cuisine",
                y="Marketing Budget", color="Location"), None),
        ("LocationWise Average Service Quality Score",
         px.bar(df.groupby('Location')['Service Quality Score'].mean().reset_index(), x="Location",
                y="Service Quality Score"), "lightyellow"),
        ("Avg Review Length", px.histogram(df, x="Avg Review Length", nbins=10), "teal"),
        ("Cuisine and Location Wise Ambience Score",
         px.bar(df.groupby(['Location', 'Cuisine'])['Ambience Score'].mean().reset_index(), x="Cuisine",
                y="Ambience Score", color="Location"), None),
        ("Average Weekend Reservation By Cuisine",
         px.bar(df.groupby('Cuisine')['Weekend Reservations'].mean().reset_index(), x='Cuisine',
                y='Weekend Reservations'), None),
        ("Average Revenue by Cuisine in Specific Location",
         px.bar(df.groupby(['Cuisine', 'Location'])['Revenue'].mean().reset_index(), x="Cuisine", y="Revenue",
                color="Location",
                color_discrete_map={"Downtown": "#1f77b4", "Rural": "#ff7f0e", "Suburban": "#2ca02c"}), None),
    ]

    for title, fig, color in plots:
        st.markdown(f"<h3 style='text-align:center;'>{title}</h3>", unsafe_allow_html=True)

        if color:
            fig.update_traces(
                marker_color=color,
                marker_line_color="black",
                marker_line_width=1.5,
            )

        fig.update_layout(
            xaxis_title="<b>" + fig.layout.xaxis.title.text + "</b>" if fig.layout.xaxis.title else "",
            yaxis_title="<b>" + fig.layout.yaxis.title.text + "</b>" if fig.layout.yaxis.title else "",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#2c3e50"),
        )

        st.plotly_chart(fig, use_container_width=True)


df.drop('Name', axis=1, inplace=True)

locationencoder = LabelEncoder()
cuisineencoder = LabelEncoder()
parkingencoder = LabelEncoder()

df['Location'] = locationencoder.fit_transform(df['Location'])
df['Cuisine'] = cuisineencoder.fit_transform(df['Cuisine'])
df['Parking Availability'] = parkingencoder.fit_transform(df['Parking Availability'])

scaler = StandardScaler()
x = df.drop('Revenue', axis=1)
y = df['Revenue']
xs = scaler.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(xs, y, test_size=0.2, random_state=42)

if c2:  # reused checkbox for showing split sizes
    st.markdown("<h3 style='text-align:center;'>Dataset Split</h3>", unsafe_allow_html=True)
    st.write("Training features :", xtrain.shape)
    st.write("Testing features  :", xtest.shape)
    st.write("Training target   :", ytrain.shape)
    st.write("Testing target    :", ytest.shape)

# Model selection & training (same logic)
if model_name == 'Linear':
    model = LinearRegression().fit(xtrain, ytrain)
elif model_name == 'Ridge':
    model = Ridge(alpha=a).fit(xtrain, ytrain)
elif model_name == 'Lasso':
    model = Lasso(alpha=a).fit(xtrain, ytrain)
elif model_name == 'Polynomial':
    poly = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model = poly.fit(xtrain, ytrain)
elif model_name == 'Xgboost':
    model = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, n_jobs=-1, verbosity=0).fit(xtrain, ytrain)
elif model_name == 'SVR':
    model = SVR(kernel='rbf').fit(xtrain, ytrain)
elif model_name == 'Elastic net':
    model = ElasticNet(alpha=a).fit(xtrain, ytrain)
elif model_name == 'GridSearch':
    param_grid = {
        'n_estimators': [100, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_child_weight': [1, 3],
    }
    grid = GridSearchCV(XGBRegressor(random_state=42), param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid.fit(xtrain, ytrain)
    model = grid.best_estimator_
    st.markdown(f"<h4>Best Parameters: {grid.best_params_}</h4>", unsafe_allow_html=True)
    st.markdown(f"<h4>Best CV MSE: {-grid.best_score_:.2f}</h4>", unsafe_allow_html=True)


def result(model):
    ypred = model.predict(xtest)
    mse = mean_squared_error(ytest, ypred)
    r2 = r2_score(ytest, ypred)
    return mse, r2


if b1:
    mse, r2 = result(model)
    col_mse, col_r2 = st.columns(2)
    with col_mse:
        st.metric("Mean Squared Error", f"{mse:,.0f}")
    with col_r2:
        st.metric("R² Score", f"{r2:.4f}")


if c5:
    st.markdown("<h2 style='text-align:center; color:#27ae60;'>🍴 Make a Prediction</h2>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        Location = col1.selectbox("Location", ['Rural', 'Suburban', 'Downtown'])
        Cuisine = col2.selectbox("Cuisine", ['Japanese', 'Mexican', 'Italian', 'Indian', 'French', 'American'])
        Rating = col3.number_input("Rating", 0.1, 5.0, 4.0, step=0.1)

        SeatingCapacity = col1.number_input("Seating Capacity", 30, 90, 40)
        AverageMealPrice = col2.number_input("Average Meal Price", 25, 80, 50)
        MarketingBudget = col3.number_input("Marketing Budget", 600, 10000, 5000)

        SocialMediaFollowers = col1.number_input("Social Media Followers", 300, 150000, 5000)
        ChefExperienceYears = col2.number_input("Chef Experience Years", 1, 25, 5)
        NumberofReviews = col3.number_input("Number of Reviews", 25, 1500, 50)

        AvgReviewLength = col1.number_input("Avg Review Length", 25, 400, 100)
        AmbienceScore = col2.number_input("Ambience Score", 1.0, 10.0, 5.0)
        ServiceQualityScore = col3.number_input("Service Quality Score", 1.0, 10.0, 5.0)

        ParkingAvailability = col1.selectbox("Parking Availability", ['Yes', 'No'])
        WeekendReservations = col2.number_input("Weekend Reservations", 0, 100, 50)
        WeekdayReservations = col3.number_input("Weekday Reservations", 0, 100, 50)

        submitted = st.form_submit_button("🚀 Predict Revenue", use_container_width=True)

        if submitted:
            check = pd.DataFrame({
                'Location': [Location], 'Cuisine': [Cuisine], 'Rating': [Rating],
                'Seating Capacity': [SeatingCapacity], 'Average Meal Price': [AverageMealPrice],
                'Marketing Budget': [MarketingBudget], 'Social Media Followers': [SocialMediaFollowers],
                'Chef Experience Years': [ChefExperienceYears], 'Number of Reviews': [NumberofReviews],
                'Avg Review Length': [AvgReviewLength], 'Ambience Score': [AmbienceScore],
                'Service Quality Score': [ServiceQualityScore], 'Parking Availability': [ParkingAvailability],
                'Weekend Reservations': [WeekendReservations], 'Weekday Reservations': [WeekdayReservations],
            })

            check['Location'] = locationencoder.transform(check['Location'])
            check['Cuisine'] = cuisineencoder.transform(check['Cuisine'])
            check['Parking Availability'] = parkingencoder.transform(check['Parking Availability'])

            check_scaled = scaler.transform(check)
            ypredic = model.predict(check_scaled)

            st.markdown(f"""
                <div class="prediction-box">
                Predicted Monthly Revenue: ₹ {ypredic[0]:,.0f}
                </div>

            """, unsafe_allow_html=True)
