from datasets import load_dataset

def inspect(name):
    print(f"=== Inspecting {name} ===")
    try:
        # Load in streaming mode to save time/memory
        ds = load_dataset(name, streaming=True)
        # Get first split
        split = list(ds.keys())[0]
        # Get first sample
        sample = next(iter(ds[split]))
        print(f"Split: {split}")
        print(f"Columns: {list(sample.keys())}")
        print(f"Sample snippet:")
        for k, v in sample.items():
            str_v = str(v)
            if len(str_v) > 500:
                str_v = str_v[:500] + "..."
            print(f"  {k}: {str_v}")
    except Exception as e:
        print(f"Error loading {name}: {e}")
    print()

inspect("WithinUsAI/claude_mythos_distilled_25k")
inspect("HelioAI/Claude-Opus-4.8-DeepThink-462x-105M")
inspect("openbmb/Ultra-FineWeb-L3")
