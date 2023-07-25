import socket
import time

def start_server():
    host = "localhost"
    port = 9090
    server_socket = socket.socket()
    server_socket.bind((host, port))
    server_socket.listen()
    client_socket, address = server_socket.accept()
    print("This is the Sender.....")
    print("Connected to ", address)
    name = input("Enter a name: ")
    reg_no = input("Enter Regno: ")
    client_socket.send(name.encode())
    client_socket.send(reg_no.encode())
    advertising_window = int(client_socket.recv(1024).decode())
    print(f"The Advertising Window Size received: {advertising_window}")
    congestion_window = 1
    i = 1
    while True:
        if congestion_window < advertising_window:
            client_socket.send(str(i).encode())
            time.sleep(1)
            print(f"Frame {i} sent..")
            time.sleep(1)
            ack = int(client_socket.recv(1024).decode())
            if ack:
                time.sleep(2)
                print(f"ACK {i} received..")
                i += 1
                congestion_window += 1
            else:
                print(f"ACK {i} not received!!")
                i += 1
                congestion_window /= 2
                congestion_window = int(congestion_window)
                print(f"The Updated Congestion Window: {congestion_window}")
                break
        else:
            print("Congestion Window is greater than Advertising Window")
            congestion_window /= 2
            congestion_window = int(congestion_window)
            print(f"The Updated Congestion Window: {congestion_window}")
            break
    client_socket.close()
    server_socket.close()

start_server()
