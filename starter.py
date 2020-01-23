import os
import time
import asyncio
from app import start_app

if __name__ == '__main__':
    while True:
        if os.path.exists("/dev/rfcomm0"):
            print("Connected\n\n")
            try:
                asyncio.run(start_app())
            except Exception as e:
                print(f'Disconnected:\n{e}\n\n Trying again in 15 seconds')
                time.sleep(15)
        else:
            # wait till the connection is established.
            time.sleep(5)
