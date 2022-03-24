from flask import Flask, render_template

app = Flask(__name__)
@app.route('/')

@app.route('/home')
def index():
    return render_template("index.html")

@app.route('/answer', methods=['GET', 'POST'])
def answer():
    return render_template("answer.html")

if __name__ == "__main__":
    app.run(debug=True)
