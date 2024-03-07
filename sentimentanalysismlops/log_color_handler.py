import logging as log
import sys


class LogColorHandler(log.StreamHandler):
    DEBUG_COLOR = "38;5;8"
    INFO_COLOR = "32;188;121"
    WARNING_COLOR = "33"
    ERROR_COLOR = "31"
    WHITE = "0"

    LEVEL_COLOR_MAP = {
        log.DEBUG: DEBUG_COLOR,
        log.INFO: INFO_COLOR,
        log.WARNING: WARNING_COLOR,
        log.ERROR: ERROR_COLOR,
    }

    def __init__(self):
        super().__init__(sys.stdout)

    def emit(self, record):
        csi = f"{chr(27)}["  # control sequence introducer
        color = self.LEVEL_COLOR_MAP.get(record.levelno, self.WHITE)
        formatted = self.format(record)
        print(f"{csi}{color}m{formatted}{csi}m")
