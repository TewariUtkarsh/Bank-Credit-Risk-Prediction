import os
import sys


class BankingException(Exception):

    def __init__(self, error_message:Exception, error_details:sys) -> None:
        """
        This class is a custom Exception Handling Class.

        Parameters
        ----------
        error_message : Exception
            Exception object
        error_details : sys
            sys module object which contains the details about traceback and exceptions.

        Attributes
        ----------
        error_message : str
            Customized error message created using exception object and traceback info.

        Returns
        -------
        error_message : str
            Customized error message for the error/exception generated.
        """
        super().__init__(error_message)
        self.error_message = BankingException.get_custom_error_message(
                                                    error_message=error_message,
                                                    error_details=error_details
                                                )


    @staticmethod
    def get_custom_error_message(error_message:Exception, error_details:sys) -> str:
        """
        This function is responsible for generating a custom  
        exception message.

        Parameters
        ----------
        error_message : Exception
            Exception object
        error_details : sys
            sys module object which contains the details about traceback and exceptions.

        Returns
        -------
        custom_error_message : str
            Customized message for the error/exception generated
        """

        _,_, exec_tb = error_details.exc_info()
        try_block_line_number = exec_tb.tb_lineno
        execption_block_number = exec_tb.tb_frame.f_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename
        
        custom_error_message = f"""
        Error Occured in Script:
        [ {file_name} ] at
        try block line number: [{try_block_line_number}] and
        exception block line number: [{execption_block_number}]
        error message: [{error_message}]
        """

        return custom_error_message


    def __str__(self) -> str:
        """
        Called when BankingException(e, sys) is executed.
        
        Returns
        -------
        self.error_message : str
            Initialized error message.
        """
        return self.error_message

    def __repr__(self) -> str:
        """
        String Representaion: Called when BankingException is executed.
        
        Returns
        -------
        str : str
            Returns the name of the class
        """
        return BankingException.__name__.str()




