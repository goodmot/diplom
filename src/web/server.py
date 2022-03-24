from flask import Flask, render_template
from flask import request
from ai import ai

predictior = ai.AI()

'''
физическая слабость;боль в мышцах и суставах;повышенная потливость;повышение температуры до 37 градусов и вышепожелтение белков глаз, слизистой рта, кожи;потемнение мочи;обесцвечивание кала;сильный зуд по телу;появление на кожных покровах ладоней, плеч, шеи красных пятен, сосудистых звездочек, кровоподтеков
'''

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
