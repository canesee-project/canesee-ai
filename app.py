import asyncio
import queue

from bluetooth import BluetoothConnection

tasks = queue.Queue(0)


async def fetch_new_tasks(remote: BluetoothConnection):
    while True:
        line = await remote.read()
        print("fetch: ", line)
        tasks.put(line, block=False)


async def process_tasks():
    while True:
        while tasks.empty():
            await asyncio.sleep(0.2)
        task = tasks.get(block=False)
        print("process: ", task)


async def start_app():
    remote = BluetoothConnection()
    await asyncio.gather(fetch_new_tasks(remote), process_tasks())
