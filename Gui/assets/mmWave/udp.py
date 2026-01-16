import socket


def send_udp_packet(target_host, target_port):
    # 创建一个UDP套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 构造要发送的数据
    message = "Hello, this is a UDP packet!"

    try:
        # 发送数据包到目标主机和端口
        udp_socket.sendto(message.encode(), (target_host, target_port))
        print(f"UDP packet sent to {target_host}:{target_port}")
    except socket.error as e:
        print(f"Error sending UDP packet: {e}")
    finally:
        udp_socket.close()


# 指定目标主机和端口
target_host = '192.168.100.86'
target_port = 8888  # 替换为您希望发送数据包的端口号

# 发送UDP数据包
send_udp_packet(target_host, target_port)