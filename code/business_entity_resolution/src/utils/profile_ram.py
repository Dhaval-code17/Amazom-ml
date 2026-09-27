import sys
import psutil
import gc
from pathlib import Path

_SRC_DIR = Path(r"d:\amazom ml\code\business_entity_resolution\src")
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from person_b.blocking import MultiChannelBlocker
from person_c.smoke_test_real_data import load_target_micro_corpus

DATA_DIR = Path(r"D:\amazom ml\6ab10eb3b23ba_student_resource\student_resource\dataset\train")

def get_ram_mb():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

print(f"Base RAM: {get_ram_mb():.2f} MB")

# We use load_target_micro_corpus with a specific noise fraction to get exactly ~10,000 records
# We know S2 has ~5M records, so 10,000 is about 0.002
t0_ram = get_ram_mb()
records = load_target_micro_corpus(DATA_DIR / "train_source2.tsv", set(), noise_fraction=0.002)
gc.collect()
t1_ram = get_ram_mb()
dict_ram = t1_ram - t0_ram

print(f"Loaded {len(records)} normalized records.")
print(f"RAM for dicts: {dict_ram:.2f} MB")
if len(records) > 0:
    print(f"Est per dict: {dict_ram / len(records) * 1024:.2f} KB")

blocker = MultiChannelBlocker(lsh_threshold=0.5, num_perm=128)
blocker.index_target_records(records)
gc.collect()
t2_ram = get_ram_mb()
idx_ram = t2_ram - t1_ram

print(f"RAM for MultiChannelBlocker index: {idx_ram:.2f} MB")
if len(records) > 0:
    print(f"Est per index entry: {idx_ram / len(records) * 1024:.2f} KB")
