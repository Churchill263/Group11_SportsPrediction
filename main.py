import streamlit as st
import pickle
import numpy as np
import random
import sklearn
import pandas as pd


scaler = pickle.load(open("scaler.pkl","rb"))

model = pickle.load(open("model.pkl", "rb"))["model"]


st.title("Group Number 11 Player Prediction")


st.header("Enter Player Attributes")
potential = st.number_input("Potential", value=0)
value_eur = st.number_input("Value (EUR)", value=0)
wage_eur = st.number_input("Wage (EUR)", value=0)
age = st.number_input("Age", value=0)
international_reputation = st.number_input("International Reputation", value=0)
release_clause_eur = st.number_input("Release Clause (EUR)", value=0)
shooting = st.number_input("Shooting", value=0)
passing = st.number_input("Passing", value=0)
dribbling = st.number_input("Dribbling", value=0)
physic = st.number_input("Physic", value=0)
attacking_crossing = st.number_input("Attacking Crossing", value=0)
attacking_short_passing = st.number_input("Attacking Short Passing", value=0)
skill_curve = st.number_input("Skill Curve", value=0)
skill_long_passing = st.number_input("Skill Long Passing", value=0)
skill_ball_control = st.number_input("Skill Ball Control", value=0)
movement_reactions = st.number_input("Movement Reactions", value=0)
power_shot_power = st.number_input("Power Shot Power", value=0)
power_long_shots = st.number_input("Power Long Shots", value=0)
mentality_aggression = st.number_input("Mentality Aggression", value=0)
mentality_vision = st.number_input("Mentality Vision", value=0)
mentality_composure = st.number_input("Mentality Composure", value=0)

if st.button("Predict"):
    # Create a feature vector from the user inputs
    # Create a dictionary with input variables
    input_features = {
        "potential": potential,
        "value_eur": value_eur,
        "wage_eur": wage_eur,
        "age": age,
        "international_reputation": international_reputation,
        "release_clause_eur": release_clause_eur,
        "shooting": shooting,
        "passing": passing,
        "dribbling": dribbling,
        "physic": physic,
        "attacking_crossing": attacking_crossing,
        "attacking_short_passing": attacking_short_passing,
        "skill_curve": skill_curve,
        "skill_long_passing": skill_long_passing,
        "skill_ball_control": skill_ball_control,
        "movement_reactions": movement_reactions,
        "power_shot_power": power_shot_power,
        "power_long_shots": power_long_shots,
        "mentality_aggression": mentality_aggression,
        "mentality_vision": mentality_vision,
        "mentality_composure": mentality_composure
    }
    data = pd.DataFrame(input_features,index=[0])
    # Scale the input features using the scaler
    scaler.fit(data)
    scaled_input = scaler.fit_transform(data)
    data = pd.DataFrame(scaled_input, columns= data.columns,index=[0])

    # Make a prediction
    prediction = model.predict(data)
    st.write(f"The predicted value is: {prediction[0]:.2f} ")


if st.button("Reset"):
    # Clear input fields by resetting their values to 0
    potential = value_eur = wage_eur = age = international_reputation = release_clause_eur = 0
    shooting = passing = dribbling = physic = attacking_crossing = attacking_short_passing = 0
    skill_curve = skill_long_passing = skill_ball_control = movement_reactions = power_shot_power = power_long_shots = 0
    mentality_aggression = mentality_vision = mentality_composure = 0
