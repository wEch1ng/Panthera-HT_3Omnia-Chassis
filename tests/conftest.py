import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import torch

torch.set_num_threads(1)
