import sys,time,os
DEFAULT_RAYSTATION_PID_PATH = f'C:/Users/{os.getlogin()}/raystation_pid.txt'

if __name__=='__main__':
	pid = sys.argv[1]
	with open(DEFAULT_RAYSTATION_PID_PATH, "w") as f:
	    f.write(pid)
	
	while True:
		time.sleep(100)