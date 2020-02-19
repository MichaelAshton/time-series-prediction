"""
Configures logging for any other module.
Source: https://xmedeko.blogspot.ch/2017/10/python-logger-by-class.html
"""

import logging.config

import os
import yaml


def setup_logging(
    default_path="logging.yaml", default_level=logging.DEBUG, env_key="LOG_CFG"
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
            pass
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


setup_logging()


def add_logger(cls: type):
    aname = "_{}__log".format(cls.__name__)
    setattr(cls, aname, logging.getLogger(cls.__module__ + "." + cls.__name__))
    return cls
