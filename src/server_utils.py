def show_connections(fserver_obj):
    print('TODO: show all connections')

def show_next_port(fserver_obj):
    print('TODO: show next port')

def show_server_ip(fserver_obj):
    print('TODO: show server ip')

def shell_help():
    print('TODO: display all shell commands')

def server_shell(fserver_obj):
    while True:
        input_cmd = input('>> ')
        if input_cmd == 'show all connections':
            show_connections(fserver_obj)
        elif input_cmd == 'show next port':
            show_next_port(fserver_obj)
        elif input_cmd == 'show server ip':
            show_server_ip(fserver_obj)
        elif input_cmd == '':
            continue
        else:
            shell_help()
