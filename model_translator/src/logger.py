from enum import IntEnum

TEST_FLAG = False
class LogLevel(IntEnum):
    INFO = 1
    WARNING = 2
    ERROR = 3

class Log(object):
    c_current_log_level = LogLevel.INFO

    @staticmethod
    def print_error(msg):
        if Log.c_current_log_level <= LogLevel.ERROR:
           print("ERROR: " + msg + '\n')

    @staticmethod
    def print_info(msg):
        if Log.c_current_log_level <= LogLevel.INFO:
            print("INFO: " + msg + '\n')

    @staticmethod
    def print_warning(msg):
        if Log.c_current_log_level <= LogLevel.WARNING:
            print("WARNING: " + msg + '\n')