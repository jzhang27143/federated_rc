import errno
import os
import pickle
import select
import socket
import torch.nn as nn
from typing import NamedTuple


class InitialObject(NamedTuple):
    grad_threshold: float
    model: nn.Module


class UpdateObject(NamedTuple):
    n_samples: int
    model_parameters: list
    client_sent: bool = True
    session_alive: bool = True
    parameter_indices: bool = False


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
        except OSError as e:
            return e.errno
    return 0

def _pad_buffer(data_buffer, buffer_size):
    return data_buffer + bytes(buffer_size - len(data_buffer))

def send_model_file(filename, socket_conn, buffer_size=1024):
    filesize = os.path.getsize(filename)
    model_serial = open(filename, 'rb')
    send_buffer = model_serial.read(buffer_size)

    # File size sent first to allow socket to persist
    fsize_msg = _pad_buffer(str(filesize).encode(), buffer_size)
    err = _send_buffer(socket_conn, fsize_msg, buffer_size)

    while not err and send_buffer:
        if len(send_buffer) != buffer_size:
            send_buffer = _pad_buffer(send_buffer, buffer_size)
        err = _send_buffer(socket_conn, send_buffer, buffer_size)
        send_buffer = model_serial.read(buffer_size)
    model_serial.close()
    return err

def _receive_buffer(socket_conn, buffer_size):
    bytes_received = 0
    data_buffer = b''
    while bytes_received < buffer_size:
        try:
            data = socket_conn.recv(buffer_size)
            if len(data) == 0: # connection closed by client
                return data_buffer, errno.ECONNABORTED
            data_buffer += data
            bytes_received += len(data)
        except BlockingIOError:
            select.select([socket_conn], [], [])
        except OSError as e:
            return data_buffer, e.errno
    return data_buffer, 0

def receive_model_file(filename, socket_conn, buffer_size=1024):
    filesize_buffer, err = _receive_buffer(socket_conn, buffer_size)
    if not err:
        filesize = int(filesize_buffer.decode().rstrip('\0'))
        bytes_remaining = ((filesize + buffer_size - 1) // buffer_size) * buffer_size
        bytes_written = 0
    else:
        return err, 0

    with open(filename, 'wb') as model_serial:
        while not err and bytes_remaining > 0:
            data, err = _receive_buffer(socket_conn, buffer_size if bytes_remaining > buffer_size
                        else bytes_remaining)
            bytes_remaining -= len(data)

            if bytes_written == filesize: # Don't write padding bytes
                continue
            elif bytes_written + len(data) > filesize: # Stop writing at last data buffer
                data = data[:filesize - bytes_written]
                model_serial.write(data)
                bytes_written = filesize
            else:
                model_serial.write(data)
                bytes_written += len(data)

    model_serial.close()
    return err, filesize
