"""List runs from the Arc2-pretrain-final-ACT-torch project to find the run ID."""
import wandb

api = wandb.Api()
project = "trelis/Arc2-pretrain-final-ACT-torch"

print(f"Fetching runs from {project}...")
runs = api.runs(project)

print(f"\nFound {len(runs)} runs:")
for run in runs:
    print(f"  Name: {run.name:40s} ID: {run.id}")
