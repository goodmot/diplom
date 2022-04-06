from flask import Flask, render_template
from flask import request
from ai import ai

predictior = ai.AI()

app = Flask(__name__)
@app.route('/')

@app.route('/home')
def index():
    return render_template("index.html")

@app.route('/answer', methods=['POST'])
def answer():
    phrase = request.form.get('phrase')
    predicted = predictior.predict(text=phrase)
    return render_template("answer.html", data={"phrase": phrase, "predicted": predicted})

def get_app():
    return app
