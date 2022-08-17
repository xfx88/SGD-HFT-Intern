import os
import shutil

path = '/home/yby/SGD-HFT-Intern/Projects/T0/Data_56'
tgt_path = '/home/yby/SGD-HFT-Intern/Projects/T0/Data/'

def move_files():
    files = os.listdir(path)
    for f in files:
        ticker = f.split("_")[1][:6]
        date = f.split("_")[0]
        current_path = path + "/" + f
        target_path = f"{tgt_path}{ticker}/"
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
        shutil.move(current_path, f"{target_path}/{date}.pkl")


if __name__ == "__main__":
    move_files()