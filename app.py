from flask import Flask, render_template, Request
from flask_cors import cross_origin, CORS
import os

app = Flask(__name__)
CORS(app)

PORT = os.getenv('$PORT', 5000)

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    return f"Flask App running on Port: {PORT}"


if __name__=='__main__':
    app.run(debug=True)