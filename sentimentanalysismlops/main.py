import logging as log
from ml_pipe import MlPipe
from log_color_handler import LogColorHandler
from appsettings import AppSettings


def config_log():
    appsettings = AppSettings()
    color_handler = LogColorHandler()
    str_level = appsettings["log_level"]
    log_level = log.getLevelName(str_level)
    log.basicConfig(format=appsettings["log_format"], level=log_level,
                    handlers=[color_handler])


if __name__ == '__main__':
    config_log()
    logger = log.getLogger(__name__)
    logger.info("main module started")
    pipe = MlPipe()
    b = pipe.dataset.batch_size

