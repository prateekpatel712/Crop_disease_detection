import sys

class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)

        _,_,exc_tb = error_detail.exc_info()

        self.error_message = (
            f"Error occurred in file: {exc_tb.tb_frame.f_code.co_filename}, "
            f"line: {exc_tb.tb_lineno}, "
            f"message: {error_message}"
        )
    def __str__(self):
            return self.error_message
        