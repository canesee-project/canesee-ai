# @author mhashim6 on 1/23/20


import os
import time
from initializer import init_models
from app import start_app
from utils import log

if __name__ == '__main__':
    log('initializing models...')
    init_models()
    log('listening for incoming connections...')
    while True:
        if os.path.exists("/dev/rfcomm0"):
            log("Connected\n\n")
            try:
                log('starting app...')
                start_app()
            except Exception as e:
                log(f'Disconnected:\n{e}\n\n Trying again in 15 seconds')
                time.sleep(15)
        else:
            # wait till the connection is established.
            time.sleep(5)
