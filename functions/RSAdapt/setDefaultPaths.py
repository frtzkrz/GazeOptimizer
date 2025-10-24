import sys,os
DEFAULT_CONNECT_PATH = r'C:\Program Files\RaySearch Laboratories\RayStation 2023B-R\ScriptClient'
DEFAULT_RAYSTATION_PID_PATH = f'C:/Users/{os.getlogin()}/raystation_pid.txt'
def setDefaultPaths():
    if DEFAULT_CONNECT_PATH not in sys.path:
        sys.path.insert(0,DEFAULT_CONNECT_PATH)
    
    with open(DEFAULT_RAYSTATION_PID_PATH, "r") as f:
        os.environ['RAYSTATION_PID'] = f.read()