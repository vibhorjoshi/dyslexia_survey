import streamlit as st
import pickle
import numpy as np
import requests
import joblib
import sqlite3
from datetime import datetime

# Load the pickled model
url = 'https://github.com/vibhorjoshi/dyslexia_survey/raw/main/model.pkl'

try:
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP errors

    with open('model.pkl', 'wb') as file:
        file.write(response.content)

    # Load the pickled model using joblib
    with open('model.pkl', 'rb') as file:
        model = joblib.load(file)  # Ensure compatibility with joblib

except requests.RequestException as e:
    st.error(f"An error occurred while downloading the file: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")

# Check if the model is loaded before using it
if 'model' in locals():
    try:
        prediction = model.predict(scores_reshaped)[0]
        st.write(f"Prediction: {prediction}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
else:
    st.error("Model is not loaded.")

# Define quiz questions and options for both rounds
questions_round1 = [
    "Are the letters 'A' and 'A' the same?",
    "Identify the fruit shown in the image: 🍎.",
    "Are the letters 'B' and 'D' the same?",
    "Select the letter 'G' from the options below.",
    "What is the first letter of the word 'CAT'?",
    "What is the lowercase version of the letter 'H'?",
    "Identify the sound you hear from the audio clip.",
    "Describe what you see in the image below.",
    "Identify which hand is the left hand and which is the right hand in the image below.",
    "Identify the sound you hear from the audio clip."
]

options_round1 = [
    ["Yes", "No"],
    ["Apple", "Banana", "Orange", "Grapes"],
    ["Yes", "No"],
    ["E", "F", "G", "H"],
    ["C", "A", "T", "B"],
    ["h", "H", "i", "I"],
    ["Cat", "Dog", "Bird", "Cow"],
    ["Tree", "House", "Car", "Mountain"],
    ["Left Hand", "Right Hand"],
    ["Bell", "Whistle", "Clap", "Knock"]
]

questions_round2 = [
    "How often do you struggle with understanding complex texts?",
    "How often do you need to read instructions multiple times to understand?",
    "How often do you find it challenging to follow detailed sequences of tasks?",
    "How often do you make mistakes in reading aloud, such as skipping words or lines?",
    "How often do you find it hard to concentrate on long reading passages?",
    "How often do you confuse similar-sounding words when reading?",
    "How often do you struggle with spelling difficult words?",
    "How often do you find it hard to comprehend abstract concepts in reading?",
    "How often do you experience difficulty with reading comprehension tests?",
    "How often do you need extra time to complete reading assignments?"
]

options_round2 = [
    ["Never", "Rarely", "Sometimes", "Often"],
    ["Never", "Rarely", "Sometimes", "Often"],
    ["Never", "Rarely", "Sometimes", "Often"],
    ["Never", "Rarely", "Sometimes", "Often"],
    ["Never", "Rarely", "Sometimes", "Often"],
    ["Never", "Rarely", "Sometimes", "Often"],
    ["Never", "Rarely", "Sometimes", "Often"],
    ["Never", "Rarely", "Sometimes", "Often"],
    ["Never", "Rarely", "Sometimes", "Often"],
    ["Never", "Rarely", "Sometimes", "Often"]
]

# Create database connection
conn = sqlite3.connect('dyslexia_results.db')
c = conn.cursor()

# Function to update or create the results table with the correct schema
def create_or_update_table():
    # Check if the table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'")
    table_exists = c.fetchone()
    
    if table_exists:
        # Check if the table has the 'round' column
        c.execute("PRAGMA table_info(results)")
        columns = [col[1] for col in c.fetchall()]
        if 'round' not in columns:
            # Update the schema to include the 'round' column
            c.execute("ALTER TABLE results ADD COLUMN round INTEGER")
    else:
        # Create the results table with the correct schema
        c.execute('''CREATE TABLE results
                     (timestamp TEXT, round INTEGER, language_vocab REAL, memory REAL, speed REAL,
                      visual_discrimination REAL, audio_discrimination REAL, survey_score REAL,
                      prediction INTEGER)''')
    conn.commit()

# Ensure the results table exists and has the correct schema
create_or_update_table()

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a Page", ["Home", "About", "Round 1", "Round 2", "Suggestions"])

def calculate_scores(answers, round_num):
    # Map string answers to numeric values
    answer_values = [options.index(answer) for answer in answers]
    if round_num == 1:
        language_vocab = (answer_values[0] + answer_values[1] + answer_values[2] + answer_values[3] + answer_values[4] + answer_values[5] + answer_values[7]) / 7
        memory = (answer_values[1] + answer_values[8]) / 2
        speed = st.sidebar.number_input("Time taken to complete the quiz (in seconds)", min_value=0, value=60)
        visual_discrimination = (answer_values[0] + answer_values[2] + answer_values[3] + answer_values[5]) / 4
        audio_discrimination = (answer_values[6] + answer_values[9]) / 2
        survey_score = sum(answer_values) / len(questions_round1)
        answer_values = [options.index(answer) for answer in answers]
        survey_score = np.mean(answer_values)
        total_score = survey_score * 5 if round_num == 1 else survey_score * 1
        return [language_vocab, memory, speed, visual_discrimination, audio_discrimination,total_score]
    else:
        language_vocab = np.mean([answer_values[0], answer_values[2], answer_values[4], answer_values[6], answer_values[8]])
        memory = np.mean([answer_values[1], answer_values[3], answer_values[5], answer_values[7], answer_values[9]])
        speed = st.sidebar.number_input("Time taken to complete the quiz (in seconds)", min_value=0, value=60)
        visual_discrimination = np.mean([answer_values[0], answer_values[2], answer_values[4], answer_values[6], answer_values[8]])
        audio_discrimination = np.mean([answer_values[1], answer_values[3], answer_values[5], answer_values[7], answer_values[9]])
        survey_score = np.mean(answer_values)
        total_score = survey_score * 5 if round_num == 1 else survey_score * 0
        return [language_vocab, memory, speed, visual_discrimination, audio_discrimination,total_score]
 
def adjust_features(scores):
    # Assuming the model expects exactly 6 features. Adjust or select features accordingly.
    if len(scores) > 6:
        return scores[:6]
    while len(scores) < 6:
        scores.append(0)
    return scores

    
def display_prediction(prediction, scores, round_num):
    # Define the ranges for prediction scores
    ranges = {
        0: "0 - 0.5",
        1: "0.5 - 1",
        2: "1 - 1.5"
    }
    
    # Display prediction result
    prediction_text = ["Low Risk", "Moderate Risk", "High Risk"][prediction]
    st.write(f"**Prediction:** {prediction_text} ({ranges[prediction]})")
    
    # Provide a detailed explanation of what the prediction means
    if prediction == 0:
        st.write("""
            **Low Risk**: Based on your answers, you are at a low risk of having dyslexia. This means your responses did not indicate significant signs typically associated with dyslexia. However, if you still have concerns, consider seeking a professional evaluation.
        """)
    elif prediction == 1:
        st.write("""
            **Moderate Risk**: Based on your answers, you are at a moderate risk of having dyslexia. This suggests some of your responses align with common characteristics of dyslexia. It may be beneficial to consult with a specialist for a comprehensive assessment.
        """)
    else:
        st.write("""
            **High Risk**: Based on your answers, you are at a high risk of having dyslexia. Many of your responses are consistent with symptoms of dyslexia. It is strongly recommended to seek a professional evaluation to understand your condition better and explore possible interventions.
        """)
       
            
   
    
    st.write("Detailed scores:")
   

    # Display scores and ranges
    st.write("### Your Score Details:")
    st.write(f"**Language and Vocabulary:** {scores[0]:.2f}")
    st.write(f"**Memory:** {scores[1]:.2f}")
    st.write(f"**Speed:** {scores[2]:.2f} seconds")
    st.write(f"**Visual Discrimination:** {scores[3]:.2f}")
    st.write(f"**Audio Discrimination:** {scores[4]:.2f}")
    st.write(f"**Survey Score:** {scores[5]:.2f}")
    st.write(f"### **Round:** {round_num}")

# Handle different pages
if page == "Home":
    st.title("Welcome to the Dyslexia Prediction System")
    st.write("This application is designed to help screen for potential signs of dyslexia. It is not a diagnostic tool but can provide useful insights based on your responses.")

elif page == "About":
    st.title("About This App")
    st.write("This app is designed to help assess potential indicators of dyslexia through a series of questions. The results can guide you on whether further evaluation is needed. Please remember that this is not a diagnostic tool.")
    
elif page == "Round 1":
    st.title("Dyslexia Quiz - Round 1")
    st.write("Please answer the following questions:")

    answers_round1 = []
    for question in questions_round1:
        answers_round1.append(st.radio(question["question"], question["options"]))

    if st.button("Submit Round 1"):
        scores_round1 = calculate_scores(answers_round1, round_num=1)
        scores_adjusted_round1 = adjust_features(scores_round1)
        prediction_round1 = model.predict([scores_adjusted_round1])[0]  # Predict using the model
        display_prediction(prediction_round1, scores_adjusted_round1, round_num=1)
        
        # Save results to the database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO results (timestamp, round, language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                  (timestamp, 1, *scores_round1, prediction_round1))
        conn.commit()

elif page == "Round 2":
    st.title("Dyslexia Quiz - Round 2")
    st.write("Please answer the following questions:")

    answers_round2 = []
    for question in questions_round2:
        answers_round2.append(st.radio(question["question"], question["options"]))

    if st.button("Submit Round 2"):
        scores_round2 = calculate_scores(answers_round2, round_num=2)
        scores_adjusted_round2 = adjust_features(scores_round2)
        prediction_round2 = model.predict([scores_adjusted_round2])[0]  # Predict using the model
        display_prediction(prediction_round2, scores_adjusted_round2, round_num=2)
        
        # Save results to the database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO results (timestamp, round, language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                  (timestamp, 2, *scores_round2, prediction_round2))
        conn.commit()

elif page == "Suggestions":
    st.title("Suggestions for Improvement")
    st.write("Here you can include suggestions for improvement based on the user's performance.")
    
    # You can add additional features here if needed, such as personalized recommendations.
    st.write("**Consider focusing on improving memory through targeted exercises.**")

# Close the database connection
conn.close()
