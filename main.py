from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/round2')
def round2():
    return render_template('round2.html')

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    answers = request.form
    total_score = sum(int(answers[f'q{i}']) for i in range(1, 11))

    if total_score >= 50:
        return redirect(url_for('round2'))
    else:
        return render_template('quiz.html', message="Your total score is less than 50. Please try again.")

@app.route('/submit_round2', methods=['POST'])
def submit_round2():
    answers = request.form
    total_score = sum(int(answers[f'q{i}']) for i in range(1, 11))

    # Process round 2 results here

    return render_template('round2.html', message="Thank you for completing the quiz!")

if __name__ == '__main__':
    app.run(debug=True)
