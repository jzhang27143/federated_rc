from src.server import ServerConfig

server_config = ServerConfig(
    wlan_ip = 'auto-discover',
    port = '8880',
    model_file_name = 'mnist_sample_cnn_server.pt',
    grad_threshold = 0.5
)
