import streamlit as st
import pickle
import numpy as np
import sqlite3
from datetime import datetime

# Load the pickled model
with open('model.pickle', 'rb') as file:
    model = pickle.load(file)

# Define quiz questions and options for both rounds
questions_round1 = [
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

options = ["Never", "Rarely", "Sometimes", "Often"]

# Create database connection
conn = sqlite3.connect('dyslexia_results.db')
c = conn.cursor()

# Create results table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS results
             (timestamp TEXT, round INTEGER, language_vocab REAL, memory REAL, speed REAL,
              visual_discrimination REAL, audio_discrimination REAL, survey_score REAL,
              prediction INTEGER)''')
conn.commit()

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a Round", ["Round 1", "Round 2"])

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
        return [language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score]
    else:
        language_vocab = np.mean([answer_values[0], answer_values[2], answer_values[4], answer_values[6], answer_values[8]])
        memory = np.mean([answer_values[1], answer_values[3], answer_values[5], answer_values[7], answer_values[9]])
        speed = st.sidebar.number_input("Time taken to complete the quiz (in seconds)", min_value=0, value=60)
        visual_discrimination = np.mean([answer_values[0], answer_values[2], answer_values[4], answer_values[6], answer_values[8]])
        audio_discrimination = np.mean([answer_values[1], answer_values[3], answer_values[5], answer_values[7], answer_values[9]])
        survey_score = np.mean(answer_values)
        return [language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score]

if page == "Round 1":
    st.title("Dyslexia Prediction Model - Round 1")
    st.write("Enter your quiz answers in the sidebar to predict the likelihood of having dyslexia.")
    
    # Sidebar for quiz input
    st.sidebar.header("Round 1 Quiz Questions")
    answers_round1 = []
    for i, question in enumerate(questions_round1):
        answers_round1.append(st.sidebar.radio(f"{i+1}. {question}", options, index=2))
    
    # Calculate scores
    scores_round1 = calculate_scores(answers_round1, 1)
    
    # Predict button
    if st.button("Predict Round 1"):
        # Reshape scores for prediction
        scores_reshaped = np.array(scores_round1).reshape(1, -1)
        prediction = model.predict(scores_reshaped)
        
        # Display result
        if prediction == 0:
            st.success("Low chance of dyslexia.")
        elif prediction == 1:
            st.warning("Moderate chance of dyslexia.")
        else:
            st.error("High chance of dyslexia.")
        
        st.write("Detailed scores:")
        st.write(f"Language Vocabulary Score: {scores_round1[0]:.2f}")
        st.write(f"Memory Score: {scores_round1[1]:.2f}")
        st.write(f"Speed Score: {scores_round1[2]:.2f}")
        st.write(f"Visual Discrimination Score: {scores_round1[3]:.2f}")
        st.write(f"Audio Discrimination Score: {scores_round1[4]:.2f}")
        st.write(f"Survey Score: {scores_round1[5]:.2f}")
        
        # Save the results to the database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO results (timestamp, round, language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (timestamp, 1, scores_round1[0], scores_round1[1], scores_round1[2], scores_round1[3], scores_round1[4], scores_round1[5], prediction[0]))
        conn.commit()

        st.write("Results saved to the database.")
        
        # Redirect to round 2 if survey score >= 2
        if scores_round1[5] >= 2:
            st.write("You have been redirected to Round 2.")
            st.experimental_rerun()

elif page == "Round 2":
    st.title("Dyslexia Prediction Model - Round 2")
    st.write("Enter your quiz answers in the sidebar to predict the likelihood of having dyslexia.")
    
    # Sidebar for round 2 quiz input
    st.sidebar.header("Round 2 Quiz Questions")
    answers_round2 = []
    for i, question in enumerate(questions_round2):
        answers_round2.append(st.sidebar.radio(f"{i+1}. {question}", options, index=2))
    
    # Calculate scores
    scores_round2 = calculate_scores(answers_round2, 2)
    total_score_round2 = sum(scores_round2)
    mean_score_round2 = total_score_round2 / len(scores_round2)

    
    # Predict button for round 2
    if st.button("Predict Round 2"):
        # Reshape scores for prediction
        scores_reshaped = np.array(scores_round2).reshape(1, -1)
        prediction = model.predict(scores_reshaped)
        
        # Display result
        if prediction == 0:
            st.success("Low chance of dyslexia.")
        elif prediction == 1:
            st.warning("Moderate chance of dyslexia.")
        else:
            st.error("High chance of dyslexia.")
        
        st.write("Detailed scores:")
        st.write(f"Language Vocabulary Score: {scores_round2[0]:.2f}")
        st.write(f"Memory Score: {scores_round2[1]:.2f}")
        st.write(f"Speed Score: {scores_round2[2]:.2f}")
        st.write(f"Visual Discrimination Score: {scores_round2[3]:.2f}")
        st.write(f"Audio Discrimination Score: {scores_round2[4]:.2f}")
        st.write(f"Survey Score: {scores_round2[5]:.2f}")
         #Display total and mean score
        st.write(f"Total Score Round 2: {total_score_round2}")
        st.write(f"Mean Score Round 2: {mean_score_round2}")
        
        # Save the results to the database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO results (timestamp, round, language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
          (timestamp,Round, scores[0], scores[1], scores[2], scores[3], scores[4], scores[5], prediction[0]))








    
