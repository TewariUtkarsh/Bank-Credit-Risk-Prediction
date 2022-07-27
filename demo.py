from flask import Flask
from flask_cors import cross_origin, CORS
import os, sys
from banking.component import data_ingestion

from banking.logger import logging
from banking.exception import BankingException
from banking.component.data_ingestion import DataIngestion
from banking.config.configuration import Configuration
from banking.pipeline.pipeline import Pipeline
from banking.utils.util import del_existing_dir

app = Flask(__name__)
CORS(app)

PORT = os.getenv('$PORT', 5000)

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    return ("Tesing Logging and Execption")
    
    
if __name__=='__main__':
    # app.run(debug=True)
    # raise Exception("Testing Exception and Logging Components")
    try:
        p = Pipeline()
        p.run_pipeline()
        print(p.display_experiment())
        pass
    except Exception as e:
        banking = BankingException(e, sys)
        logging.info(banking)

    # del_existing_dir(directory='logs')