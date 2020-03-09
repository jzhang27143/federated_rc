import socket

BUFFER_SIZE = 1024

def send_model_file(filename, socket_conn):
    model_serial = open(filename, 'rb')
    send_buffer = model_serial.read(BUFFER_SIZE)

    while send_buffer:
        socket_conn.send(send_buffer)
        send_buffer = model_serial.read(1024)
    model_serial.close()

def receive_model_file(filename, socket_conn):
    with open(filename, 'wb') as model_serial:
        while True:
            data = socket_conn.recv(BUFFER_SIZE)
            if not data:
                break
            model_serial.write(data)
    model_serial.close()
