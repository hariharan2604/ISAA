import socket
import time

def start_client():
    host = "localhost"
    port = 9090
    client_socket = socket.socket()
    client_socket.connect((host, port))
    print("This is the Receiver")
    print("The Name received from Server: ", client_socket.recv(1024).decode())
    print("The Regno received from Server: ", client_socket.recv(1024).decode())
    advertising_window = int(input("Enter The Advertising Window Size: "))
    client_socket.send(str(advertising_window).encode())
    while True:
        frame = client_socket.recv(1024).decode()
        time.sleep(1)
        print(f"Frame {int(frame)} received")
        ack = int(input("Enter 1 to send ACK, 0 to send NACK: "))
        if ack:
            client_socket.send(frame.encode())
            time.sleep(1)
            print(f"ACK {int(frame)} sent.")
        else:
            client_socket.send("0".encode())
            print(f"ACK {int(frame)} is lost")
            break
    client_socket.close()

start_client()
