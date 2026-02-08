import logging
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
LOG_DIR = os.path.abspath(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, 'regeneration.log')

# Configure a dedicated logger for regeneration flows
logger = logging.getLogger('resonate.regen')
logger.setLevel(logging.INFO)

# File handler
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
fh.setFormatter(formatter)

# Stream handler (console)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(sh)

# Convenience function
def get_logger():
    return logger
