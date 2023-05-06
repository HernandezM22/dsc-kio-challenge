import pandas as pd
import ast

df = pd.read_csv("file/path")
df = df[df["system"].notna()].reset_index(drop=True)

events = pd.json_normalize(df["event"].apply(ast.literal_eval), max_level=1)
server = pd.json_normalize(df["host"].apply(ast.literal_eval), max_level=1)

df["event"] = events["dataset"]
df["host"] = server["name"]

cpu = df[df["event"]=="system.cpu"].reset_index(drop=True)
memory = df[df["event"]=="system.memory"].reset_index(drop=True)
network = df[df["event"]=="system.network"].reset_index(drop=True)

network_metrics = pd.json_normalize(network["system"].apply(ast.literal_eval), max_level=4)
cpu_metrics = pd.json_normalize(cpu["system"].apply(ast.literal_eval), max_level=4)
memory_metrics = pd.json_normalize(memory["system"].apply(ast.literal_eval), max_level=4)

memory_metrics["timestamp"] = memory["@timestamp"]
memory_metrics["event"] = memory["event"]
memory_metrics["node"] = memory["host"]

cpu_metrics["timestamp"] = cpu["@timestamp"]
cpu_metrics["event"] = cpu["event"]
cpu_metrics["node"] = cpu["host"]

network_metrics["timestamp"] = network["@timestamp"]
network_metrics["event"] = network["event"]
network_metrics["node"] = network["host"]

network_metrics.to_pickle("/home/mauricio/repos/dsc-kio-challenge/app/data/network.pkl")
memory_metrics.to_pickle("/home/mauricio/repos/dsc-kio-challenge/app/data/memory.pkl")
cpu_metrics.to_pickle("/home/mauricio/repos/dsc-kio-challenge/app/data/cpu.pkl")