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
    "Are the letters 'A' and 'A' the same?",
    "Identify the fruit shown in the image: 🍎 (Apple, Banana, Orange, Grapes).",
    "Are the letters 'B' and 'D' the same?",
    "Select the letter 'G' from the options: (E, F, G, H).",
    "What is the first letter of the word 'CAT'?",
    "What is the lowercase version of the letter 'H'?",
    "Identify the sound you hear from the audio clip: (options might include 'Cat', 'Dog', 'Bird', 'Cow').",
    "Describe what you see in the image below: (options might include 'Tree', 'House', 'Car', 'Mountain').",
    "Identify which hand is the left hand and which is the right hand in the image below.",
    "Identify the sound you hear from the audio clip: (options might include 'Bell', 'Whistle', 'Clap', 'Knock')."
]

# Update the code that displays and processes these questions accordingly
answers_round1 = []
for i, question in enumerate(questions_round1):
    answers_round1.append(st.sidebar.radio(f"{i+1}. {question}", options,index=2))

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
page = st.sidebar.selectbox("Choose a Page", ["Home", "About", "Round 1", "Round 2"])

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

if page == "Home":
    st.title("Welcome to the Dyslexia Prediction Model")
    st.write("This application helps predict the likelihood of having dyslexia based on quiz answers.")
    st.image("Webpage/dyslexia1.jpg", use_column_width=True)
    st.write("Navigate through the sidebar to participate in the quiz or learn more about the project.")

elif page == "About":
    st.title("About the Dyslexia Prediction Model")
    st.image("Webpage/ai1.jpg", use_column_width=True, caption="AI Image") if st.file_uploader("Upload 'ai1.jpg'") else st.write("Please upload 'ai1.jpg'.")
    st.header("Introduction to Dataset")
    st.write("""
        This project utilizes machine learning to analyze quiz scores and predict an applicant's likelihood of having dyslexia. The dataset contains scores from various sections:
        - Language Vocabulary
        - Memory
        - Processing Speed
        - Visual Discrimination
        - Auditory Discrimination
        In addition, a survey score is included. These scores are used to determine a "Dyslexia Risk Label" ranging from 0 to 2:
        - 0: Low Risk
        - 1: Moderate Risk
        - 2: High Risk
        This model aims to identify individuals who might benefit from further evaluation for dyslexia.
    """)
    st.image("Webpage/features.png", width=800)
    
    st.header("Calculation of Scores")
    st.image("Webpage/score.png", width=800)

    st.header("Working of the Model")
    st.write("""
        This project aims to predict an applicant's dyslexia risk ("Label") based on the following quiz scores:
        - Language Vocabulary
        - Memory
        - Processing Speed
        - Visual Discrimination
        - Auditory Discrimination
        We additionally include a survey score.
        To find the optimal model for our dataset, we compared five different machine learning algorithms within the "DyslexiaML" file. These algorithms were:
        - Decision Tree
        - Random Forest
        - Support Vector Machine (SVM)
        - Random Forest with Grid Search
        - SVM with Grid Search
    """)
    st.image("Webpage/graph.png", width=800)
    st.image("Webpage/error.png", width=200)
    st.write("""
        On the basis of our findings, we then created the final model using RandomForestClassifier with GridSearchCV in order to make the most accurate predictions, in 'DyslexiaML_final' file. This model was then tested on a new dataset to find the labels, which were then compared with the actual label values. After this check we found out that our model was able to make predictions for dyslexia with a 5.8% error rate.
    """)
    st.image("Webpage/cm.jpeg", width=800)
    st.markdown("[Link to our GitHub Repository](https://github.com)")

elif page == "Round 1":
    st.title("Dyslexia Prediction Model - Round 1")
    st.write("Enter your quiz answers in the sidebar to predict the likelihood of having dyslexia.")
    
    # Sidebar for quiz input
    st.sidebar.header("Round 1 Quiz Questions")
    answers_round1 = []
    for i, question in enumerate(questions_round1):
        answers_round1.append(st.sidebar.radio(f"{i+1}. {question}", options, index=2))
    
    # Calculate scores and predict
    if st.sidebar.button("Submit Round 1 Answers"):
        scores = calculate_scores(answers_round1, 1)
        prediction = model.predict([scores])[0]

        # Save results to the database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO results (timestamp, round, language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (timestamp, 1, *scores, prediction))
        conn.commit()
        
        # Display the prediction result
        prediction_text = ["Low Risk", "Moderate Risk", "High Risk"][prediction]
        st.write(f"**Prediction:** {prediction_text}")

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
        st.write("Your answers and scores have been saved to the database.")

elif page == "Round 2":
    st.title("Dyslexia Prediction Model - Round 2")
    st.write("Enter your quiz answers in the sidebar to predict the likelihood of having dyslexia.")
    
    # Sidebar for quiz input
    st.sidebar.header("Round 2 Quiz Questions")
    answers_round2 = []
    for i, question in enumerate(questions_round2):
        answers_round2.append(st.sidebar.radio(f"{i+1}. {question}", options, index=2))
    
    # Calculate scores and predict
    if st.sidebar.button("Submit Round 2 Answers"):
        scores = calculate_scores(answers_round2, 2)
        prediction = model.predict([scores])[0]

        # Save results to the database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO results (timestamp, round, language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (timestamp, 2, *scores, prediction))
        conn.commit()
        
        # Display the prediction result
        prediction_text = ["Low Risk", "Moderate Risk", "High Risk"][prediction]
        st.write(f"**Prediction:** {prediction_text}")

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
        st.write("Your answers and scores have been saved to the database.")

# Close the database connection
conn.close()

