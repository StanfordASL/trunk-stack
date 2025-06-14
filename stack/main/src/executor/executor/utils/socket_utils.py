import struct
import numpy as np
import socket
import time

###################################################
# Socket utilities for sending and receiving data #
#     File should be equivalent to GNN MPC        #
###################################################

def pack_state(t: float, x0: np.ndarray) -> bytes:
    assert x0.ndim == 2
    rows, cols = x0.shape
    header = struct.pack('fii', t, rows, cols)  # 4+4+4 bytes
    body = x0.astype(np.float32).tobytes()
    return header + body

def unpack_state(data: bytes):
    t, rows, cols = struct.unpack('fii', data[:12])
    x0 = np.frombuffer(data[12:], dtype=np.float32).reshape((rows, cols))
    return t, x0


def pack_control(u: np.ndarray) -> bytes:
    n = u.size
    header = struct.pack('i', n)
    body = u.astype(np.float32).tobytes()
    return header + body

def unpack_control(data: bytes):
    n = struct.unpack('i', data[:4])[0]
    u = np.frombuffer(data[4:], dtype=np.float32)
    return u

def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Socket closed")
        data += packet
    return data

def send_state(sock: socket.socket, t: float, x0: np.ndarray):
    payload = pack_state(t, x0)
    sock.sendall(payload)

def recv_state(sock: socket.socket):
    header = recv_exact(sock, 12)
    t, rows, cols = struct.unpack('fii', header)
    body = recv_exact(sock, 4 * rows * cols)
    x0 = np.frombuffer(body, dtype=np.float32).reshape((rows, cols))
    return t, x0

def send_control(sock: socket.socket, u: np.ndarray):
    payload = pack_control(u)
    sock.sendall(payload)

def recv_control(sock: socket.socket):
    header = recv_exact(sock, 4)
    n = struct.unpack('i', header)[0]
    body = recv_exact(sock, 4 * n)
    u = np.frombuffer(body, dtype=np.float32)
    return u

def setup_socket_server(host: str, port: int):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    conn, addr = s.accept()
    return conn

def setup_socket_client(host: str, port: int):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    while True:
        try:
            s.connect((host, port))
            return s
        except (ConnectionRefusedError, OSError):
            print("Waiting for server to be available...")
            time.sleep(1)
