from pathlib import Path
import time
from datetime import datetime
def create_dir_and_return_path(pathname):
    print("Creating directory with current dir at " + str(Path.cwd()))
    datetime.now()
    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    Path(f"{pathname}_{dt_string}").mkdir(parents=True, exist_ok=True)


    return f"{pathname}_{dt_string}/"
