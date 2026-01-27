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

st.title("🍽️ Restaurant Revenue Forecaster")
@st.cache_data
def load_data():
    df=pd.read_csv("restaurant_data.csv")
    return df

df = load_data()

c1 = st.sidebar.checkbox("Show Raw Original Data")
if c1:
    st.write("<h2><center>Original Data</center></h2>", unsafe_allow_html=True)
    st.dataframe(df)
c2 = st.sidebar.checkbox("EDA Report")
if c2:
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Top Five Rows</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(5))
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Last Five Rows</h3>",
                unsafe_allow_html=True)
    st.dataframe(df.tail(5))
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>All Column Info</h3>",
                unsafe_allow_html=True)
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>All Column Numeric Description</h3>",
                unsafe_allow_html=True)
    st.write(df.describe())
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Null Value Count</h3>",
                unsafe_allow_html=True)
    st.write(df.isnull().sum())
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>CapacityWise count of restaurants</h3>",
                unsafe_allow_html=True)
    fig = px.histogram(df, x="Seating Capacity", nbins=10)
    fig.update_layout(
        xaxis_title="<b>Seating Capacity</b>",
        yaxis_title="<b>Number of Restaurants</b>",
    )
    fig.update_traces(
        marker_color="pink",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    st.plotly_chart(fig)
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>LocationWise Count of Restaurants</h3>",
                unsafe_allow_html=True)
    g = df.groupby('Location')['Name'].count().reset_index()
    fig = px.bar(g, x="Location", y="Name", labels={"Location": "Location", "Name": "Number of Restaurants"})
    fig.update_traces(
        marker_color="pink",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    fig.update_layout(
        xaxis_title="<b>Location</b>",
        yaxis_title="<b>Number of Restaurants</b>"
    )
    st.plotly_chart(fig)
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Cuisine TypeWise Count of Restaurants</h3>",
                unsafe_allow_html=True)
    f = df.groupby('Cuisine')['Name'].count().reset_index()
    fig = px.bar(f, x="Cuisine", y="Name")
    fig.update_traces(
        marker_color="pink",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    fig.update_layout(
        xaxis_title="<b>Cuisine</b>",
        yaxis_title="<b>Number of Restaurants</b>"
    )
    st.plotly_chart(fig)
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Count of Restaurants By Rating</h3>",
                unsafe_allow_html=True)
    fig = px.histogram(df, x="Rating", nbins=5)
    fig.update_layout(
        xaxis_title="<b>Rating</b>",
        yaxis_title="<b>Number of Restaurants</b>"
    )
    fig.update_traces(
        marker_color="pink",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    st.plotly_chart(fig)
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Count of CustomerReviews By Cuisine</h3>",
                unsafe_allow_html=True)
    q = df.groupby('Cuisine')['Number of Reviews'].count().reset_index()
    fig = px.bar(q, x="Cuisine", y="Number of Reviews")
    fig.update_traces(
        marker_color="darkblue",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    fig.update_layout(
        xaxis_title="<b>Cuisine</b>",
        yaxis_title="<b>Number of Reviews</b>"
    )
    st.plotly_chart(fig)
    st.markdown(
        "<h3 style='text-align: center; font-weight: bold;'>Count of Restaurants By Parking Availability In Specific Region</h3>",
        unsafe_allow_html=True)
    h = df.groupby(['Parking Availability', 'Location'])['Name'].count().reset_index()
    fig = px.bar(h, x="Location", y="Name", color="Parking Availability", barmode='stack')
    fig.update_layout(
        xaxis_title="<b>Parking Availability</b>",
        yaxis_title="<b>Number of Restaurants</b>"
    )
    st.plotly_chart(fig)
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>CuisineWise Average Meal Price</h3>",
                unsafe_allow_html=True)
    j = df.groupby('Cuisine')['Average Meal Price'].mean().reset_index()
    fig = px.bar(j, x="Cuisine", y="Average Meal Price")
    fig.update_traces(
        marker_color="brown",
        marker_line_color="White",
        marker_line_width=1.5,
    )
    fig.update_layout(
        xaxis_title="<b>Cuisine</b>",
        yaxis_title="<b>Average Meal Price</b>",
    )
    st.plotly_chart(fig)
    st.markdown(
        "<h3 style='text-align: center; font-weight: bold;'>CuisineWise Average Social Media Followers In Specific Region</h3>",
        unsafe_allow_html=True)
    k = df.groupby(['Cuisine', 'Location'])['Social Media Followers'].mean().reset_index()
    fig = px.bar(k, x="Cuisine", y="Social Media Followers", color="Location")
    fig.update_layout(
        xaxis_title="<b>Cuisine</b>",
        yaxis_title="<b>Social Media Followers</b>",
    )
    st.plotly_chart(fig)
    st.markdown(
        "<h3 style='text-align: center; font-weight: bold;'>CuisineWise Average Marketing budget In Specific Region</h3>",
        unsafe_allow_html=True)
    l = df.groupby(['Cuisine', 'Location'])['Marketing Budget'].mean().reset_index()
    fig = px.bar(l, x="Cuisine", y="Marketing Budget", color="Location")
    fig.update_layout(
        xaxis_title="<b>Cuisine</b>",
        yaxis_title="<b>Marketing Budget</b>",
    )
    st.plotly_chart(fig)
    st.markdown(
        "<h3 style='text-align: center; font-weight: bold;'>LocationWise Average Service Quality Score</h3>",
        unsafe_allow_html=True)
    o = df.groupby('Location')['Service Quality Score'].mean().reset_index()
    fig = px.bar(o, x="Location", y="Service Quality Score")
    fig.update_traces(
        marker_color="lightyellow",
        marker_line_color="Black",
        marker_line_width=2,
    )
    fig.update_layout(
        xaxis_title="<b>Location</b>",
        yaxis_title="<b>Service Quality Score</b>",
    )
    st.plotly_chart(fig)
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Avg Review Length</h3>",
                unsafe_allow_html=True)
    fig = px.histogram(df, x="Avg Review Length", nbins=10)
    fig.update_layout(
        xaxis_title="<b>Avg Review Length</b>",
        yaxis_title="<b>Count</b>",
    )
    fig.update_traces(
        marker_color="teal",
        marker_line_color="black",
        marker_line_width=1.5,
    )
    st.plotly_chart(fig)
    st.markdown(
        "<h3 style='text-align: center; font-weight: bold;'>Cuisine and Location Wise Ambience Score</h3>",
        unsafe_allow_html=True)
    p = df.groupby(['Location', 'Cuisine'])['Ambience Score'].mean().reset_index()
    fig = px.bar(p, x="Cuisine", y="Ambience Score", color="Location")
    fig.update_layout(
        xaxis_title="<b>Cuisine</b>",
        yaxis_title="<b>Ambience Score</b>",
    )
    st.plotly_chart(fig)
    st.markdown(
        "<h3 style='text-align: center; font-weight: bold;'>Average Weekend Reservation By Cuisine</h3>",
        unsafe_allow_html=True)
    w = df.groupby('Cuisine')['Weekend Reservations'].mean().reset_index()
    fig = px.bar(w, x='Cuisine', y='Weekend Reservations')
    fig.update_layout(
        xaxis_title="<b>Cuisine</b>",
        yaxis_title="<b>Weekend Reservations</b>",
    )
    st.plotly_chart(fig)
    st.markdown(
        "<h3 style='text-align: center; font-weight: bold;'>Average Revenue by Cuisine in Specific Location </h3>",
        unsafe_allow_html=True)
    z = df.groupby(['Cuisine', 'Location'])['Revenue'].mean().reset_index()
    fig = px.bar(z, x="Cuisine", y="Revenue", color="Location", color_discrete_map={
        "Downtown": "#1f77b4",
        "Rural": "#ff7f0e",
        "Suburban": "#2ca02c",
    }, )
    fig.update_layout(
        xaxis_title="<b>Cuisine</b>",
        yaxis_title="<b>Revenue</b>",
    )
    st.plotly_chart(fig)

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
c2 = st.sidebar.checkbox("To Show Size of Training and Testing")
if c2:
    st.write("xtrain", xtrain.shape)
    st.write("xtest", xtest.shape)
    st.write("ytrain", ytrain.shape)
    st.write("ytest", ytest.shape)
model_name = st.sidebar.selectbox("Select model:--", ['Linear', 'Ridge', 'Lasso', 'Polynomial', 'Xgboost', 'SVR', 'Elastic net','GridSearch'])


if model_name == 'Linear':
    linear = LinearRegression()
    model = linear.fit(xtrain, ytrain)
elif model_name == 'Ridge':
    a = st.sidebar.number_input("enter alpha value", 0.1, 2.0, .2)
    ridge = Ridge(alpha=a)
    model = ridge.fit(xtrain, ytrain)
elif model_name == 'Lasso':
    a = st.sidebar.number_input("enter alpha value", 0.1, 2.0, .2)
    lasso = Lasso(alpha=a)
    model = lasso.fit(xtrain, ytrain)
elif model_name == 'Polynomial':
    d = st.sidebar.number_input("Enter Degree", 1, 10, 2)
    degree = d
    poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model = poly.fit(xtrain, ytrain)
elif model_name == 'Xgboost':
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, n_jobs=-1, verbosity=0)
    model = xgb.fit(xtrain, ytrain)
elif model_name == 'SVR':
    svr = SVR(kernel='rbf')
    model = svr.fit(xtrain, ytrain)
elif model_name == 'Elastic net':
    a = st.sidebar.number_input("enter alpha value", 0.01, 2.0, .2)
    elastic_net = ElasticNet(alpha=a)
    model = elastic_net.fit(xtrain, ytrain)
elif model_name == 'GridSearch':
    xg = XGBRegressor(random_state=42)
    param_grid = {'n_estimators': [100, 300],
                  'learning_rate': [0.05, 0.1],
                  'max_depth': [3, 5],
                  'min_child_weight': [1, 3],
                  }
    grid = GridSearchCV(estimator=xg, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=3)
    model = grid.fit(xtrain, ytrain)
    st.write("Best Parameters:", grid.best_params_)
    st.write("Best MSE:", -grid.best_score_)
def result(model):
    ypred = model.predict(xtest)
    mse = mean_squared_error(ytest, ypred)
    r2 = r2_score(ytest, ypred)
    return mse, r2
ans = result(model)
b1 = st.sidebar.button("Ans")
if b1:
    st.write(ans)
c5 = st.sidebar.checkbox("For Prediction")
if c5:
    col1, col2, col3 = st.columns(3)
    Location = col1.selectbox("Location", ['Rural', 'Suburban', 'Downtown'])
    Cuisine = col2.selectbox("Cuisine", ['Japanese', 'Mexican', 'Italian', 'Indian', 'French', 'American', ])
    Rating = col3.number_input("Rating", 0.1, 5.0, 4.0)
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
    check = pd.DataFrame({
        'Location': [Location],
        'Cuisine': [Cuisine],
        'Rating': [Rating],
        'Seating Capacity': [SeatingCapacity],
        'Average Meal Price': [AverageMealPrice],
        'Marketing Budget': [MarketingBudget],
        'Social Media Followers': [SocialMediaFollowers],
        'Chef Experience Years': [ChefExperienceYears],
        'Number of Reviews': [NumberofReviews],
        'Avg Review Length': [AvgReviewLength],
        'Ambience Score': [AmbienceScore],
        'Service Quality Score': [ServiceQualityScore],
        'Parking Availability': [ParkingAvailability],
        'Weekend Reservations': [WeekendReservations],
        'Weekday Reservations': [WeekdayReservations],
    })
    check['Location'] = locationencoder.transform(check['Location'])
    check['Cuisine'] = cuisineencoder.transform(check['Cuisine'])
    check['Parking Availability'] = parkingencoder.transform(check['Parking Availability'])
    check_scaled = scaler.transform(check)
    b0 = st.button("Predict")
    if b0:
        ypredic = model.predict(check_scaled)
        st.write("Your Restaurants Predicted Revenue is", ypredic)