import subprocess
import sys
import time

N = int(sys.argv[1])
PORT_0 = 6969

for i in range(N):
    print("Starting process {i}")
    cmd = f"python3 api.py --port {PORT_0 + i} --selenium"
    proc = subprocess.Popen([cmd], shell=True,
             stdin=None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)

    time.sleep(5)

# pkill python
# pkill Geckomain
