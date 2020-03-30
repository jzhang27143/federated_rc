from typing import NamedTuple
import socket
import pickle
import os

class UpdateObject(NamedTuple):
    n_samples: int
    model_parameters: list

def send_model_file(filename, socket_conn, buffer_size=1024):
    filesize = os.path.getsize(filename)
    model_serial = open(filename, 'rb')
    send_buffer = model_serial.read(buffer_size)

    # File size sent first to allow socket to persist
    socket_conn.send(str(filesize).encode())
    while send_buffer:
        socket_conn.send(send_buffer)
        send_buffer = model_serial.read(buffer_size)
    model_serial.close()

def receive_model_file(filename, socket_conn, buffer_size=1024):
    bytes_remaining = int(socket_conn.recv(1024).decode())
    with open(filename, 'wb') as model_serial:
        while bytes_remaining > 0:
            data = socket_conn.recv(buffer_size)
            model_serial.write(data)
            bytes_remaining -= len(data)
    model_serial.close()

