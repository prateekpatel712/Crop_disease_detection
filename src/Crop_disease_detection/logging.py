import logging
import sys

LOG_FORMAT = "[[%(asctime)s] %(levelname)s in %(filename)s:%(lineno)d - %(message)s]"

logging.basicConfig(
level=logging.INFO, 
format=LOG_FORMAT,
handlers=[logging.StreamHandler(sys.stdout)]
)
    
    
    