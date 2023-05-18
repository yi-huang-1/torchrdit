import logging

class Logger():
    """Class of logging for global use"""

    def __init__(self,
                 log_level = logging.INFO,
                 flog_level = logging.INFO,
                 file_path = None):

        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create a stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)

        # Add the stream handler to the logger
        self.logger.addHandler(stream_handler)

        # Create a file handler if needed
        if file_path != None:
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(flog_level)
            file_handler.setFormatter(formatter)

            # Add the file handler to the logger
            self.logger.addHandler(file_handler)


    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
