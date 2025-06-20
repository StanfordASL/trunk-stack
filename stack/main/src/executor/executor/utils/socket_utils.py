import struct
import numpy as np
import socket
import time

def pack_state(t: float, x0: np.ndarray, motor_state: np.ndarray) -> bytes:
    assert x0.ndim == 2
    assert motor_state.shape == (6,3), "Motor state must be 6-dimensional"
    rows, cols = x0.shape
    header = struct.pack('fii', t, rows, cols)  # 4+4+4 bytes
    body = x0.astype(np.float32).tobytes()
    motor_data = motor_state.astype(np.float32).tobytes()
    return header + body + motor_data

def unpack_state(data: bytes):
    t, rows, cols = struct.unpack('fii', data[:12])
    state_size = rows * cols * 4  # 4 bytes per float32
    x0 = np.frombuffer(data[12:12+state_size], dtype=np.float32).reshape((rows, cols))
    motor_state = np.frombuffer(data[12+state_size:], dtype=np.float32).reshape((6, 3))
    return t, x0, motor_state


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

def send_state(sock: socket.socket, t: float, x0: np.ndarray, motor_state: np.ndarray):
    payload = pack_state(t, x0, motor_state)
    sock.sendall(payload)

def recv_state(sock: socket.socket):
    header = recv_exact(sock, 12)
    t, rows, cols = struct.unpack('fii', header)
    body = recv_exact(sock, 4 * (rows * cols + 6*3))  # +6 for motor state
    state_size = rows * cols * 4
    x0 = np.frombuffer(body[:state_size], dtype=np.float32).reshape((rows, cols))
    motor_state = np.frombuffer(body[state_size:], dtype=np.float32)
    return t, x0, motor_state

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
    
    print(f"Socket ready on {host}:{port}")
    return conn

def setup_socket_client(host: str, port: int):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print("Waiting for server to be available...")

    while True:
        try:
            s.connect((host, port))
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(5)

    print(f"Connected to server at {host}:{port}")
    return s
