import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
from bluetooth import BluetoothConnection

tasks = queue.Queue(0)


def fetch_new_tasks(remote: BluetoothConnection):
    while True:
        line = await remote.read()
        print("fetch: ", line)
        tasks.put(line, block=False)


def process_tasks(remote: BluetoothConnection):
    while True:
        while tasks.empty():
            await asyncio.sleep(0.2)
        task = tasks.get(block=False)
        print("process: ", task)


def start_app():
    remote = BluetoothConnection()
    executor = ThreadPoolExecutor(2)
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(loop.run_in_executor(executor, fetch_new_tasks, remote))
    asyncio.ensure_future(loop.run_in_executor(executor, process_tasks, remote))
    loop.run_forever()


if __name__ == '__main__':
    start_app()
