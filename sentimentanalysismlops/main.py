import logging as log
import os.path

from ml_pipe import MlPipe
from log_color_handler import LogColorHandler
from appsettings import AppSettings
import argparse


appsettings = AppSettings()


def config_log():
    color_handler = LogColorHandler()
    str_level = appsettings["log_level"]
    log_level = log.getLevelName(str_level)
    log.basicConfig(format=appsettings["log_format"], level=log_level,
                    handlers=[color_handler])


if __name__ == '__main__':
    config_log()
    logger = log.getLogger(__name__)
    parser = argparse.ArgumentParser(description='Process commands.')

    parser.add_argument('command', help='Command to run, e.g, "train" or "eval".')
    parser.add_argument('--text', help='text to evaluate', default=None)

    args = parser.parse_args()
    is_train = False

    if args.command == "train":
        logger.warning("Training mode")
        is_train = True
    elif args.command == "eval":
        if args.text is not None:
            logger.warning("Evaluation mode")
            logger.warning(f"Evaluating text: {args.text}")
        else:
            logger.error("No text provided for evaluation")
    else:
        logger.error(f"Unknown command: {args.command}")

    logger.info("main module started")
    pipe = MlPipe()

    if is_train:
        pipe.train()
    elif os.path.exists(appsettings["model_path"]):
        pipe.load_model(appsettings["model_path"])
        mark = pipe.predict_review(args.text)
        print(f"The mark is: {mark}")
    else:
        logger.error(f"File \"{appsettings['model_path']}\" does not exist, configure file path in config.json")
