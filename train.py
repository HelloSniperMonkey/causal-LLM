# teacher forcing, scheduled sampling, and other training strategies. will be implemented from scratch. if possible
# the biggest complexity of implementing from scratch is the maths required for backpropagation through time (BPTT) and handling variable-length sequences.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from dataset import TextDataset