from federatedrc.server import ServerConfig

server_config = ServerConfig(
    wlan_ip = '192.168.1.100',
    port = '8880',
    model_file_name = 'mnist_sample_cnn_server.pt',
    rx_history_file_name = 'rx_history.png',
    grad_threshold = 1.5
)
