import socket

HOST = "127.0.0.1"
PORT = 50002  # Must match wrapper.exe

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect((HOST, PORT))
    print("✅ Connected to wrapper successfully!")
    s.close()
except Exception as e:
    print("❌ Connection failed:", e)
