import os
from src.train import run_train

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data_npy')
    run_train(data_root=data_path)