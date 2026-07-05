import subprocess
import time
import sys
import os

def main():
    cmd = "ssh -n -o StrictHostKeyChecking=no -R 80:127.0.0.1:9700 nokey@localhost.run"
    log_file = "llm_tunnel.log"
    print("Starting localhost.run tunnel wrapper on port 9700...", flush=True)
    
    # Remove old log file if it exists to avoid parsing stale URLs
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except Exception:
            pass

    while True:
        try:
            print("Launching SSH tunnel...", flush=True)
            with open(log_file, "a", encoding="utf-8") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=True
                )
                
                for line in iter(process.stdout.readline, ''):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    f.write(line)
                    f.flush()
                
                process.wait()
            print(f"SSH process exited with code {process.returncode}. Restarting in 2s...", flush=True)
        except Exception as e:
            print(f"Tunnel error: {e}. Restarting in 2s...", flush=True)
        time.sleep(2)

if __name__ == "__main__":
    main()
