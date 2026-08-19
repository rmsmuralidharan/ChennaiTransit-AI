import sys

class ChennaiTransitAIException:

    def __init__(
            self,
            error_message: str,
            error_detail = None
    ):
        super().__init__(error_message)

        self.error_message = error_message

        if error_detail is not None:
            _, _, traceback = error_detail

            if traceback is None:
                self.file_name = traceback.tb_frame.f_code.co_filename
                self.line_number = self.file_name.tb_lineno

            else:
                self.file_name = __file__
                self.line_number = 0

        else:
            frame = sys._getframe(1)

            self.file_name = frame.f_code.co_filename
            self.line_number = frame.f_lineno

        def __str__(self):
            return(
                f"Error ocurred in file {self.file_name} at line {self.line_number} - {error_message}"
            )

        