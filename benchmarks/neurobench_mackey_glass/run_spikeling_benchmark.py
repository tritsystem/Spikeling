import os
import sys
import torch
from torch.utils.data import Subset, DataLoader

from neurobench.datasets import MackeyGlass
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark
from neurobench.metrics.workload import SMAPE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spikeling_resonator_reservoir import SpikelingResonatorReservoir

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mackey_glass")

TAUS = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [17]
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

lyaptime_pts = 75
sMAPE_scores = {}

for tau in TAUS:
    scores = []
    for repeat_id in range(REPEATS):
        filepath = os.path.join(data_dir, f"mg_{tau}.npy")
        offset = repeat_id * 0.5 * lyaptime_pts

        mg = MackeyGlass(filepath, start_offset=offset, bin_window=1)
        train_set = Subset(mg, mg.ind_train)
        test_set = Subset(mg, mg.ind_test)

        model_inst = SpikelingResonatorReservoir(in_channels=1, seed_id=repeat_id)
        model_inst.train()
        train_data, train_labels = train_set[:]
        warmup = 0.6
        warmup_pts = round(warmup * mg.pts_per_lyaptime)
        train_labels = train_labels[warmup_pts:]
        model_inst.fit(train_data, train_labels, warmup_pts)

        test_loader = DataLoader(test_set, batch_size=mg.testtime_pts, shuffle=False)
        model = TorchModel(model_inst)

        benchmark = Benchmark(model, test_loader, [], [], [[], [SMAPE]])
        results = benchmark.run()
        score = results["SMAPE"]
        scores.append(score)
        print(f"tau={tau} repeat={repeat_id}: sMAPE={score:.4f}")

    sMAPE_scores[tau] = scores

print()
all_scores = [s for scores in sMAPE_scores.values() for s in scores]
print(f"Spikeling Resonator-bank reservoir, {len(TAUS)} tau value(s) x {REPEATS} repeats:")
print(f"  mean sMAPE = {sum(all_scores)/len(all_scores):.4f}")
print(f"  per-tau means: " + ", ".join(f"tau={t}: {sum(v)/len(v):.4f}" for t, v in sMAPE_scores.items()))
print()
print("Official NeuroBench leaderboard reference (14 tau x 30 repeats, tuned hyperparameters):")
print("  LSTM sMAPE = 13.37")
print("  ESN  sMAPE = 14.79")
