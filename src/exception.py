# To handle all the error like divisionByZero

import sys


def error_message_detail(error, error_detail: sys):
    _,_,exc_tb = error_detail.exc_info()   #it will give three output but we are capturing the third detail called error_detail

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occured python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message






# defining the custom exception
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        #a constructor
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail = error_detail)

    def __str__(self):
        return self.error_message