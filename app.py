import streamlit as st
import pickle
import numpy as np
import sqlite3
from datetime import datetime
import joblib
import requests

url = 'https://github.com/vibhorjoshi/dyslexia_survey/raw/main/model.pkl'

try:
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP errors

    with open('model.pkl', 'wb') as file:
        file.write(response.content)

    # Load the pickled model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

except requests.RequestException as e:
    st.error(f"An error occurred while downloading the file: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")
    
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
    st.write(f"Language Vocabulary Score: {scores[0]:.2f}")
    st.write(f"Memory Score: {scores[1]:.2f}")
    st.write(f"Speed Score: {scores[2]:.2f}")
    st.write(f"Visual Discrimination Score: {scores[3]:.2f}")
    st.write(f"Audio Discrimination Score: {scores[4]:.2f}")
    st.write(f"Survey Score: {scores[5]:.2f}")

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
        To find the optimal model for our dataset, we compared five different machine learning algorithms within the "DyslexiaML" file.
    """)
elif page == "Round 1":
    st.title("Dyslexia Prediction Model - Round 1")
    st.write("Enter your quiz answers in the sidebar to predict the likelihood of having dyslexia.")

    # Sidebar for round 1 quiz input
    st.sidebar.header("Round 1 Quiz Questions")
    answers_round1 = []
    for i, question in enumerate(questions_round1):
        answers_round1.append(st.sidebar.radio(f"{i+1}. {question}", options, index=2))

    if st.sidebar.button("Submit Round 1 Answers"):
        scores = calculate_scores(answers_round1, 1)
        scores_reshaped = np.array(scores).reshape(1, -1)
        prediction = model.predict(scores_reshaped)[0]
       
        # Store Round 1 results in the session state
        

        # Save results to the database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO results (timestamp, round, language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score,prediction ) VALUES (?, ?, ?, ?, ?, ?, ?, ?,?)",
                  (timestamp, 1, *scores, prediction))
        conn.commit()

        # Display the prediction result
        display_prediction(prediction, scores, 1,)
        
        # Redirect to round 2 if survey score >= 2
        if scores == 0.5:
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

    if st.sidebar.button("Submit Round 2 Answers"):
        scores = calculate_scores(answers_round2, 1)
        scores_reshaped = np.array(scores).reshape(1, -1)
        prediction = model.predict(scores_reshaped)[0]
        
        # Store Round 2 results in the session state
       
        # Save results to the database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO results (timestamp, round, language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (timestamp, 2, *scores, prediction))
        conn.commit()

        # Display the prediction result
        display_prediction(prediction,scores,2)

 #Suggestions Page
elif page == "Suggestions":
    st.title("Personalized Suggestions")
    st.write("Based on your quiz results, here are some suggestions to help you:")

    # Display suggestions based on scores
    st.sidebar.header("Detailed Suggestions")
    if "results" in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        results = conn.execute("SELECT * FROM results ORDER BY timestamp DESC LIMIT 1").fetchone()
        if results:
            _, round_num, language_vocab, memory, speed, visual_discrimination, audio_discrimination, survey_score, prediction = results
            
            st.sidebar.write(f"**Language Vocabulary Score:** {language_vocab:.2f}")
            st.sidebar.write(f"**Memory Score:** {memory:.2f}")
            st.sidebar.write(f"**Speed Score:** {speed:.2f}")
            st.sidebar.write(f"**Visual Discrimination Score:** {visual_discrimination:.2f}")
            st.sidebar.write(f"**Audio Discrimination Score:** {audio_discrimination:.2f}")
            st.sidebar.write(f"**Survey Score:** {survey_score:.2f}")
            st.sidebar.write(f"**Prediction:** {['Low Risk', 'Moderate Risk', 'High Risk'][prediction]}")

            total_score = language_vocab + memory + speed + visual_discrimination + audio_discrimination
            st.write(f"**Total Score (Out of 10):** {total_score:.2f}")

            st.write("### Recommendations:")
            if prediction == 0:
                st.write("Based on your total score, it appears that you are at low risk of dyslexia. It is still advisable to consult with a professional for confirmation.")
            elif prediction == 1:
                st.write("Based on your total score, it appears that you are at moderate risk of dyslexia. We recommend consulting with a specialist for further evaluation.")
            else:
                st.write("Based on your total score, it appears that you are at high risk of dyslexia. It is strongly recommended to seek professional evaluation and support.")

# Close the database connection when done
conn.close()


# Run the Streamlit app

create_or_update_table()
    
# Close the database connection when done
conn.close()
