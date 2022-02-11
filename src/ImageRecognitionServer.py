# ImageRecognitionServer - Server that receives and stores images from RPi
# @author Lim Rui An, Ryan
# @version 1.0
# @since 2022-02-10
# @modified 2022-02-10

from SymbolRecognizer import SymbolRecognizer as SymRec
from pcComm import *

# Constant PATH variables
WEIGHT_PATH = "../weights/e40b16v8best.pt"
YOLO_PATH = "../yolov5"
IMAGE_PATH = "../testimg/1_5.jpg"

RECEIVER_PATH = "../receivedimg/"
RECEIVER_FILE_PATH = RECEIVER_PATH + 'out.jpg'

# Other Constants
ANNOTATION_CLASSES = ['1_blue', '2_green', '3_red', '4_white', '5_yellow', '6_blue', '7_green', '8_red', '9_white', 'a_red', 'b_green', 'bullseye', 'c_white', 'circle_yellow', 'd_blue', 'down_arrow_red', 'e_yellow', 'f_red', 'g_green', 'h_white', 'left_arrow_blue', 'right_arrow_green', 's_blue', 't_yellow', 'u_red', 'up_arrow_white', 'v_green', 'w_white', 'x_blue', 'y_yellow', 'z_red']
NUM_CLASSES = 31

# Main runtime
def main():
    # Connect to RPi
    pc_obj = pc()
    pc_obj.connect()

    sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    sock.connect(("192.168.10.10", 3333))
    #sock.listen(2)

    sock.send(bytes("IRS Pinging RPi", 'utf-8'))

    print("Receiving File Name")
    #imgNum = recv_w_timeout(sock, 1)
    RECEIVER_FILE_PATH = RECEIVER_PATH + sock.recv(1024).decode('utf-8') + '.jpg'
    #print(imgNum.decode('utf-8'))
    print("Data Received")

    #while true:
    with open(RECEIVER_FILE_PATH, "wb") as f:
        #while True:
        # read 1024 bytes from the socket (receive)
        print("Receiving Data")
        bytes_read = recv_w_timeout(sock, 1)#sock.recv(1024) #experiment with this to get fastest time!!
        print("Data Received")
        #if not bytes_read:
            # file transmitting is done
        #    print("break")
        #    break
        # write to the file the bytes we just received
        for i in range(len(bytes_read)):
            f.write(bytes_read[i])
        #progress.update(len(bytes_read))
        print("File write done")
    # Get result from processed image
    sr = SymRec(WEIGHT_PATH, ANNOTATION_CLASSES, NUM_CLASSES)
    #msg = sr.ProcessSourceImages(IMAGE_PATH)
    msg = sr.ProcessSourceImages(RECEIVER_FILE_PATH)
    print("Detected: " + msg)
    sock.send(bytes(msg, 'utf-8'))

    sock.close()

def recv_w_timeout(sock, timeout = 1):
    # Make socket non-blocking
    sock.setblocking(0)

    # Data buffers
    total_data = []
    data = '';

    # Track time for checking timeouts
    startTime = time.time()

    # Loop to grab data from stream
    while True:
        # If data has already been received, wait for timeout and break
        if total_data and time.time() - startTime > timeout:
            break
        # If no data has been received, wait a bit more before timing out
        elif time.time() - startTime > timeout * 2: # MIGHT NOT NEED THIS
            break
        # Try to receive Data
        try:
            data = sock.recv(2048)
            if data: # Valid data attained
                #total_data.append(data.decode('utf-8'))
                total_data.append(data)
                # Reset timeout start time
                startTime = time.time()
            else:
                # No valid data received, wait for a bit
                time.sleep(0.1)
        except:
            pass
    # Concatenate the received data and return it
    #return ''.join(total_data)
    return total_data

if __name__ == "__main__":
    main()
