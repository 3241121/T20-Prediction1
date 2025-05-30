import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor
from PIL import Image

pipe = pickle.load(open('pipe.pkl','rb'))

teams =  ['Afghanistan',
    'Australia',
    'Bangladesh',
    'Canada',
    'England',
    'India',
    'Ireland',
    'Namibia',
    'Nepal',
    'Netherlands',
    'New Zealand',
    'Oman',
    'Pakistan',
    'Papua New Guinea',
    'Scotland',
    'South Africa',
    'Sri Lanka',
    'United States of America',
    'West Indies'
]

cities = ['Brisbane', 'Cape Town', 'Dehra Dun', 'Greater Noida', 'Mirpur', 'Delhi', 'Colombo', 'Abu Dhabi', 'Sharjah', 'Chittagong', 'Birmingham', 'Cardiff', 'Bengaluru', 'Bristol', 'Hamilton', 'Durban', 'Nottingham', 'Dublin', 'Dominica', 'London', 'Lauderhill', 'Indore', 'Wellington', 'Edinburgh', 'King City', 'Deventer', 'Nagpur', 'Adelaide', 'Dubai', 'Hambantota', 'Chester-le-Street', 'Pallekele', 'Barbados', 'Sydney', 'Johannesburg', 'Auckland', 'Visakhapatnam', 'Belfast', 'Dharamsala', 'Kandy', 'Mumbai', 'Al Amarat', 'Ranchi', 'Lucknow', 'Trinidad', 'Lahore', 'Paarl', 'Centurion', 'Rajkot', 'Kolkata', 'Christchurch', 'Bready', 'Victoria', 'Mount Maunganui', 'Bloemfontein', 'Incheon', 'Chennai', 'Guyana', 'Perth', 'Nairobi', 'Pune', 'St Lucia', 'Hobart', 'Rotterdam', 'Dehradun', 'Amstelveen', 'Dhaka', 'Melbourne', 'Townsville', 'Cuttack', 'Manchester', 'Napier', 'Antigua', 'St Vincent', 'Basseterre', 'Bangalore', 'Nelson', 'St Kitts', "St George's", 'Chandigarh', 'Potchefstroom', 'Chattogram', 'Fatullah', 'Thiruvananthapuram', 'Kanpur', 'The Hague', 'Jamaica', 'Guwahati', 'Southampton', 'Sylhet', 'Harare', 'Canberra', 'Karachi', 'Ahmedabad', 'Morrisville', 'Gros Islet', 'Port Elizabeth', 'Carrara', 'Kampala', 'Dharmasala', 'Taunton', 'Providence', 'Hyderabad', 'East London']


title ='MEN T20 SCORE PREDICTOR'
st.markdown(f"<h1 style='text-align: center'>{title}</h1>",unsafe_allow_html=True)



image2=Image.open("ICC_t20.jpeg")

st.image(image2)

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city',sorted(cities))

col3,col4,col5 = st.columns(3)

with col3:
    current_score = int(st.number_input('Current Score', step = 1))
with col4:
    overs = int(st.number_input('Overs done(works for over>5)', step = 1))
with col5:
    wickets = int(st.number_input('Wickets out', step = 1))

last_five = st.number_input('Runs scored in last 5 overs', step = 1)


if st.button('Predict Score'):
    balls_left = 120 - (overs*6)
    wickets_left = 10 -wickets
    crr = current_score/overs

    input_df = pd.DataFrame(
     {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':city, 'current_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
    result = pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))


