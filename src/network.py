from typing import NamedTuple
import select
import socket
import pickle
import os

class UpdateObject(NamedTuple):
    n_samples: int
    model_parameters: list

# send and recv are not sufficient due to non-blocking socket
def _send_buffer(socket_conn, data, buffer_size):
    bytes_remaining = buffer_size
    data_buffer = data
    while bytes_remaining > 0:
        try:
            bytes_sent = socket_conn.send(data_buffer)
            bytes_remaining -= bytes_sent
            data_buffer = data_buffer[bytes_sent:]
        except BlockingIOError:
            select.select([], [socket_conn], [])

def _pad_buffer(data_buffer, buffer_size):
    return data_buffer + bytes(buffer_size - len(data_buffer))

def send_model_file(filename, socket_conn, buffer_size=1024):
    filesize = os.path.getsize(filename)
    model_serial = open(filename, 'rb')
    send_buffer = model_serial.read(buffer_size)

    # File size sent first to allow socket to persist
    fsize_msg = _pad_buffer(str(filesize).encode(), buffer_size)
    _send_buffer(socket_conn, fsize_msg, buffer_size)

    while send_buffer:
        if len(send_buffer) != buffer_size:
            send_buffer = _pad_buffer(send_buffer, buffer_size)
        _send_buffer(socket_conn, send_buffer, buffer_size)
        send_buffer = model_serial.read(buffer_size)
    model_serial.close()

def _receive_buffer(socket_conn, buffer_size):
    bytes_received = 0
    data_buffer = b''
    while bytes_received < buffer_size:
        try:
            data = socket_conn.recv(buffer_size)
            data_buffer += data
            bytes_received += len(data)
        except BlockingIOError:
            select.select([socket_conn], [], [])
    return data_buffer

def receive_model_file(filename, socket_conn, buffer_size=1024):
    filesize = int(_receive_buffer(socket_conn, buffer_size).decode().rstrip('\0'))
    bytes_remaining = ((filesize + buffer_size - 1) // buffer_size) * buffer_size
    bytes_written = 0

    with open(filename, 'wb') as model_serial:
        while bytes_remaining > 0:
            data = _receive_buffer(socket_conn, buffer_size if bytes_remaining > buffer_size
                    else bytes_remaining)
            bytes_remaining -= len(data)

            if bytes_written == filesize:
                continue
            elif bytes_written + len(data) > filesize: # Stop writing at last data buffer
                data = data[:filesize - bytes_written]
                model_serial.write(data)
                bytes_written = filesize
            else:
                model_serial.write(data)
                bytes_written += len(data)

    model_serial.close()

