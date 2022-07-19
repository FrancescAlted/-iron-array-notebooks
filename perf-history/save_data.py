import sys
import iarray as ia
import os
import pandas as pd
from math import log10, floor


def round_sig(x, sig=3):
    if x == 0:
        return 0
    return round(x, sig-int(floor(log10(abs(x))))-1)

max_nparams = 5
filename = str(sys.argv[1])
f = open(filename, mode="r")
lines = f.read().split("\n")


mem_info = []
for line in lines:
    line = line.split(" ")
    if line[0] == "MEM":
        mem_info.append(float(line[1]))
    elif line[0] == "FUNC":
        # stop_mem - start_mem
        bl_memory = round_sig(float(line[2]))
        ss_mem = round_sig(float(line[4]) - bl_memory)
        # stop - start time
        time = round_sig(float(line[5]) - float(line[3]))
    elif line[0] == "CMDLINE":
        # Get script params
        script_info = line[2:]
        if len(script_info[1:]) < max_nparams:
            for i in range(len(script_info[1:]), max_nparams):
                script_info.append("0")

f.close()

# Memory peak in MiB
peak_mem = round_sig(max(max(mem_info) - bl_memory, ss_mem))
# version
version = ia.__version__.split("-")[0]

params = ["param" + str(i) for i in range(1, max_nparams + 1)]
names = ["script"] + params + ["bl_memory", "peak_memory", "ss_memory", "time", "version"]
values = script_info + [bl_memory, peak_mem, ss_mem, time, version]
data = {names[i]: values[i] for i in range(0, len(names))}
data_df = pd.DataFrame(data, index=[0])

csv_path = "perf-history.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df = df.astype({"param1": str})
    df = df.astype({"param2": str})
    df = df.astype({"param3": str})
    df = df.astype({"param4": str})
    df = df.astype({"param5": str})
    df = df.astype({"version": str})
    ids = ["script"] + params + ["version"]
    df = pd.concat([df, data_df], ignore_index=True)
    df = df.drop_duplicates(subset=ids, keep='last')
else:
    df = data_df
df.to_csv(csv_path, index=False)
