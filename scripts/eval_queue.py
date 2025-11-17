import queue
import threading
import sys
import subprocess

q = queue.Queue()

def evaluate():
    while(True):
        if(q.not_empty):
            command = q.get()
            subprocess.run(["python", "python/" + command[0]] +  command[1:]) # automatically waits for process to finish

evaluate_daemon = threading.Thread(target=evaluate)
evaluate_daemon.daemon = True
evaluate_daemon.start()

while(True):
    command = input()
    if command == "exit":
        sys.exit()
    else:
        # print(command)
        command_list = command.split(" ")
        # print(command_list)
        q.put(command_list)