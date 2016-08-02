from enum import Enum
import logging.config

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


class Loggers(Enum):
    def __str__(self):
        return str(self.value)

    ROOT = 'root'
    STANDARD = 'standard'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(levelname)s: %(name)s: %(message)s '
                      '(%(asctime)s; %(filename)s:%(lineno)d)',
            'datefmt': "%Y-%m-%d %H:%M:%S",
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'standard': {
            'handlers': ['console'],
            'level': 'INFO',
        },
        'root': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    }
}


def defineLogger(logger_name):
    if not isinstance(logger_name, Loggers):
        raise TypeError("Must be a valid Logger")
    logging.config.dictConfig(LOGGING)
    return logging.getLogger(logger_name.value)
