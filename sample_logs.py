import os
import random
from inspect_ai.log import read_eval_log, write_eval_log

def sample_log(log_path, n_samples=100):
    print(f"Sampling {n_samples} from {log_path}...")
    try:
        log = read_eval_log(log_path)
        if hasattr(log, 'samples') and log.samples and len(log.samples) > n_samples:
            # Randomly sample
            log.samples = random.sample(log.samples, n_samples)
            
            # Update results if needed? Actually, let's keep the results/metadata 
            # so they see the overall scores, but only a few samples for browsing.
            
            write_eval_log(log)
            print(f"Reduced {log_path} to {n_samples} samples.")
    except Exception as e:
        print(f"Error sampling {log_path}: {e}")

if __name__ == "__main__":
    log_dir = "supplementary/logs"
    for filename in os.listdir(log_dir):
        if filename.endswith(".eval"):
            # Skip very small ones if any
            path = os.path.join(log_dir, filename)
            if os.path.getsize(path) > 1024 * 1024: # > 1MB
                sample_log(path, 100)
