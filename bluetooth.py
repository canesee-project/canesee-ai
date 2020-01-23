import serial


class BluetoothConnection:
    def __init__(self):
        self.port = serial.Serial("/dev/rfcomm0", baudrate=9600)

    async def read(self):
        return self.port.readline()

    async def send(self, text):
        self.port.write(text)