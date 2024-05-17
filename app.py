import streamlit as st
import pickle
import numpy as np
import sqlite3
from datetime import datetime

# Load the pickled model
with open('model.pickle', 'rb') as file:
    model = pickle.load(file)

# Define quiz questions and options
questions = [
    "How often do you find it difficult to follow instructions?",
    "How often do you forget things you have just read?",
    "How often do you struggle to read words correctly?",
    "How often do you mix up letters in words?",
    "How often do you find it hard to concentrate?",
    "How often do you lose your place while reading?",
    "How often do you mispronounce words?",
    "How often do you confuse similar-looking words?",
    "How often do you find it hard to remember what you read?",
    "How often do you need to re-read sentences to understand them?"
]

options = ["Never", "Rarely", "Sometimes", "Often"]

# Create database connection
conn = sqlite3.connect('dyslexia_results.db')
c = conn.cursor()

# Create results table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS results
             (timestamp TEXT, language_vocab REAL, memory REAL, speed REAL,
              visual_discrimination REAL, audio_discrimination REAL, survey_score REAL,
              prediction INTEGER)''')
conn.commit()

# Sidebar for quiz input
st.sidebar.header("Quiz Questions")
answers = []
for i, question in enumerate(questions):
    answers.append(st.sidebar.radio(f"{i+1}. {question}", options, index=2))

# Function to calculate scores
def calculate_scores(answers):
    # Map string answers to numeric values
    answer_values = [options.index(answer) for answer in answers]
    language_vocab = (answer_values[0] + answer_values[1] + answer_values[2] + answer_values[3] + answer_values[4] + answer_values[5] + answer_values[7]) / 28
    memory = (answer_values[1] + answer_values[8]) / 8
    speed = st.sidebar.number_input("Time taken to complete the quiz (in seconds)", min_value=0, value=60)
    visual_discrimination = (answer_values[0] + answer_values[2] + answer_values[3] + answer_values[5]) / 16
    audio_discrimination = (answer_values[6] + answer_values[9]) / 8
    survey_score = sum(answer_values) / 80
    return [language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score]

# Calculate scores
scores = calculate_scores(answers)

# Main page
st.title("Dyslexia Prediction Model")
st.write("Enter your quiz answers in the sidebar to predict the likelihood of having dyslexia.")

# Predict button
if st.button("Predict"):
    # Reshape scores for prediction
    scores_reshaped = np.array(scores).reshape(1, -1)
    prediction = model.predict(scores_reshaped)
    
    # Display result
    if prediction == 0:
        st.success("Low chance of dyslexia.")
    elif prediction == 1:
        st.warning("Moderate chance of dyslexia.")
    else:
        st.error("High chance of dyslexia.")
    
    st.write("Detailed scores:")
    st.write(f"Language Vocabulary Score: {scores[0]:.2f}")
    st.write(f"Memory Score: {scores[1]:.2f}")
    st.write(f"Speed Score: {scores[2]:.2f}")
    st.write(f"Visual Discrimination Score: {scores[3]:.2f}")
    st.write(f"Audio Discrimination Score: {scores[4]:.2f}")
    st.write(f"Survey Score: {scores[5]:.2f}")
    
    # Save the results to the database
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO results (timestamp, language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (timestamp, scores[0], scores[1], scores[2], scores[3], scores[4], scores[5], prediction[0]))
    conn.commit()

    st.write("Results saved to the database.")

if __name__ == '__main__':
    st.run()


    
