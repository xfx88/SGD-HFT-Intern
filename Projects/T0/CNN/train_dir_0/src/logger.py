import logging
import threading

format = "||%(asctime)s||%(name)s||%(levelname)s||%(module)s||%(funcName)s||%(lineno)s||%(message)s||"
# format = "||%(asctime)s||%(levelname)s||%(module)s||%(lineno)s||%(message)s||"

initLock = threading.Lock()
rootLoggerInitialized = False

def getLogger(name=None, level=logging.INFO):
    global rootLoggerInitialized
    with initLock:
        if not rootLoggerInitialized:
            logger = logging.getLogger(name)
            logger.setLevel(level)

            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(format))
            logger.addHandler(handler)

            rootLoggerInitialized = True

            return logger

    return logging.getLogger(name)