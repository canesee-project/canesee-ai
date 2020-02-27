# @author mhashim6 on 1/23/20


import serial
import time


class BluetoothConnection:
    def __init__(self):
        self.port = serial.Serial("/dev/rfcomm0", baudrate=115200)
        time.sleep(1)
        self.port.flushInput()

    def read(self):
        line = self.port.readline()
        return line

    def send(self, text):
        self.port.write((text+'\n').encode())
        self.port.flushOutput()
        time.sleep(0.2)
        print("sent: ", text.encode())
