from modrec_skorch_gpu_cluster import Discriminator, SaveBestParam, StopRestore, Score_ConfusionMatrix
import pickle

with open('result.pkl', 'rb') as f:
    res = pickle.load(f)

print(1)