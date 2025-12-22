import subprocess
import signal
import sys
import time
from dotenv import load_dotenv

import os
load_dotenv()
N_WORKERS = int(os.getenv("N_WORKERS"))

processes = []

def handle_sigint(sig, frame):
    
    print("\n[CTRL + C] Terminating all workers...")
    
    for p in processes:
        if p.poll() is None: # if the process is still running then return None ... if so we need to terminate it
            print(f" - Killing PID {p.pid}")
            p.terminate()
            
    time.sleep(1)
    
    for p in processes:
        if p.poll() is None:
            p.kill()
            
    sys.exit(0) 

signal.signal(signal.SIGINT, handle_sigint) # on Ctrl + C, execute handle_sigint (setup signal handler)

try:

    p_coordinator = subprocess.Popen(["python3", "coordinator.py"]) # Initialize coordinator

    processes.append(p_coordinator)
    
    for i in range(N_WORKERS):
        
        p = subprocess.Popen(["python3", "worker.py", "--id", str(i)]) # Open process, control it via p
        
        processes.append(p)

    for p in processes:
        p.wait()    # the script that spawned the processes, is waiting on them (so it doesnt exit)

except KeyboardInterrupt:
    handle_sigint(None, None)
