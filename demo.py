from flask import Flask
from flask_cors import cross_origin, CORS
import os, sys

from banking.logger import logging
from banking.exception import BankingException

app = Flask(__name__)
CORS(app)

PORT = os.getenv('$PORT', 5000)

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    try:
        raise Exception("Testing Exception and Logging Components")
    except Exception as e:
        banking = BankingException(e, sys)
        logging.info(banking)
        return ("Tesing Logging and Execption")
    
    
if __name__=='__main__':
    app.run(debug=True)