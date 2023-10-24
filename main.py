import streamlit as st
import pickle
import numpy as np
import random
import sklearn



scaler = pickle.load(open("scaler.pkl","rb"))

model = pickle.load(open("model.pkl", "rb"))


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

counter = 0
if st.button("Predict"):
    # Create a feature vector from the user inputs
    input_features = [
        potential, value_eur, wage_eur, age, international_reputation, release_clause_eur,
        shooting, passing, dribbling, physic, attacking_crossing, attacking_short_passing,
        skill_curve, skill_long_passing, skill_ball_control, movement_reactions,
        power_shot_power, power_long_shots, mentality_aggression, mentality_vision,
        mentality_composure
    ]

    # Scale the input features using the scaler
    scaled_input_features = scaler.transform(input_features)
    data = pd.DataFrame(scaled_input,index[0])

    # Make a prediction
    prediction = model.predict(data)
    st.write(f"The predicted value is: {prediction[0]:.2f} ")


if st.button("Reset"):
    # Clear input fields by resetting their values to 0
    potential = value_eur = wage_eur = age = international_reputation = release_clause_eur = 0
    shooting = passing = dribbling = physic = attacking_crossing = attacking_short_passing = 0
    skill_curve = skill_long_passing = skill_ball_control = movement_reactions = power_shot_power = power_long_shots = 0
    mentality_aggression = mentality_vision = mentality_composure = 0
