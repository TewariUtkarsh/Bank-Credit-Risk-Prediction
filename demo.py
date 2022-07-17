from flask import Flask
from flask_cors import cross_origin, CORS
import os

from banking.logger import logging

app = Flask(__name__)
CORS(app)

PORT = os.getenv('$PORT', 5000)

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    
    message = f"Testing the Logging Component"
    logging.info(message)
    
    return "Bank Credit Risk Prediction"
    


if __name__=='__main__':
    app.run(debug=True)