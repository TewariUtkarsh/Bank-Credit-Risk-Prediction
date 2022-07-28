from banking.exception import BankingException
from banking.logger import logging
import os, sys
import pandas as pd
import numpy as np
from banking.utils.util import load_model_object_from_file


class BankingData:

    def __init__(self,
        laufkont:int
        ,laufzeit:int
        ,moral:int
        ,verw:int
        ,hoehe:int
        ,sparkont:int
        ,beszeit:int
        ,rate:int
        ,famges:int
        ,buerge:int
        ,wohnzeit:int
        ,verm:int
        ,alter:int
        ,weitkred:int
        ,wohn:int
        ,bishkred:int
        ,beruf:int
        ,pers:int
        ,telef:int
        ,gastarb:int,
        credit_risk:int=None
    ) -> None:
        try:
            self.laufkont=laufkont
            self.laufzeit=laufzeit
            self.moral=moral
            self.verw=verw
            self.hoehe=hoehe
            self.sparkont=sparkont
            self.beszeit=beszeit
            self.rate=rate
            self.famges=famges
            self.buerge=buerge
            self.wohnzeit=wohnzeit
            self.verm=verm
            self.alter=alter
            self.weitkred=weitkred
            self.wohn=wohn
            self.bishkred=bishkred
            self.beruf=beruf
            self.pers=pers
            self.telef=telef
            self.gastarb=gastarb
            self.credit_risk=credit_risk
        except Exception as e:
            raise BankingException(e, sys) from e

    def get_dict_for_data(self) -> dict:
        try:
            input_data_for_df = {
                "laufkont":  [self.laufkont],
                "laufzeit": [self.laufzeit],
                "moral": [self.moral],
                "verw": [self.verw],
                "hoehe": [self.hoehe],
                "sparkont": [self.sparkont],
                "beszeit": [self.beszeit],
                "rate": [self.rate],
                "famges": [self.famges],
                "buerge": [self.buerge],
                "wohnzeit": [self.wohnzeit],
                "verm": [self.verm],
                "alter": [self.alter],
                "weitkred": [self.weitkred],
                "wohn": [self.wohn],
                "bishkred": [self.bishkred],
                "beruf": [self.beruf],
                "pers": [self.pers],
                "telef": [self.telef],
                "gastarb": [self.gastarb]
            }
            input_data_for_display = {
                "Status":  [self.laufkont],
                "Duration": [self.laufzeit],
                "Credit History": [self.moral],
                "Purpose": [self.verw],
                "Amount": [self.hoehe],
                "Savings": [self.sparkont],
                "Employment Duration": [self.beszeit],
                "Installment Rate": [self.rate],
                "Personal Status Sex": [self.famges],
                "Other Debtors": [self.buerge],
                "Present Residence": [self.wohnzeit],
                "Property": [self.verm],
                "Age": [self.alter],
                "Other Installment Plans": [self.weitkred],
                "Housing": [self.wohn],
                "Number Credits": [self.bishkred],
                "Job": [self.beruf],
                "People Liable": [self.pers],
                "Telephone": [self.telef],
                "Foreign Worker": [self.gastarb]
            }
            return [input_data_for_display, input_data_for_df]
        except Exception as e:
            raise BankingException(e, sys) from e

    def get_input_dataframe(self) -> pd.DataFrame:
        try:
            input_dict = self.get_dict_for_data()[1]
            columns = ["laufkont",
                "laufzeit",
                "moral",
                "verw",
                "hoehe" ,
                "sparkont",
                "beszeit",
                "rate",
                "famges",
                "buerge",
                "wohnzeit",
                "verm",
                "alter",
                "weitkred",
                "wohn",
                "bishkred",
                "beruf",
                "pers",
                "telef",
                "gastarb"]
            input_df = pd.DataFrame(input_dict, columns=columns)
            return input_df

        except Exception as e:
            raise BankingException(e, sys) from e




class BankingPredictor:
    
    def __init__(self, model_dir:str) -> None:
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise BankingException(e, sys) from e

    def load_model_from_path(self) -> str:
        try:
            model_dir_files = [int(file) for file in os.listdir(self.model_dir)]
            best_model_dir_name = str(max(model_dir_files))
            best_model_dir = os.path.join(self.model_dir, best_model_dir_name)
            best_model_file_name = [file for file in os.listdir(best_model_dir)][0]
            best_model_file_path = os.path.join(best_model_dir, best_model_file_name)
            model = load_model_object_from_file(best_model_file_path)
            return model
        except Exception as e:
            raise BankingException(e, sys) from e

    def predict(self, X):
        try:
            model = self.load_model_from_path()
            prediction = model.predict(X)
            return prediction
        except Exception as e:
            raise BankingException(e, sys) from e

