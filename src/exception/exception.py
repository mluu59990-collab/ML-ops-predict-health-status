import sys
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        self.error_message = error_message
        _,_,exc_tb = error_detail.exc_info()
        self.lineno =exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename
    def __str__(self):
        return "Error occurred in script: {0} at line number: {1} with message: {2}".format(
            self.filename,self.lineno,self.error_message
        )