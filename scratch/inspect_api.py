import json
import urllib.request

def get_splits_and_configs(dataset_name):
    print(f"=== Getting splits and configs for {dataset_name} ===")
    url = f"https://datasets-server.huggingface.co/splits?dataset={dataset_name}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            splits = data.get('splits', [])
            print(f"Splits and Configs found:")
            for s in splits:
                print(f"  Config: {s.get('config')}, Split: {s.get('split')}")
            # Try fetching first rows for the first config/split
            if splits:
                first = splits[0]
                config = first.get('config')
                split = first.get('split')
                print(f"Fetching first rows for config={config}, split={split}...")
                rows_url = f"https://datasets-server.huggingface.co/first-rows?dataset={dataset_name}&config={config}&split={split}"
                req_rows = urllib.request.Request(rows_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req_rows) as res_rows:
                    data_rows = json.loads(res_rows.read().decode())
                    features = data_rows.get('features', [])
                    print(f"  Columns: {[f['name'] for f in features]}")
                    rows = data_rows.get('rows', [])
                    if rows:
                        row_data = rows[0].get('row', {})
                        print("  Sample keys & snippet:")
                        for k, v in row_data.items():
                            str_v = str(v)
                            if len(str_v) > 200:
                                str_v = str_v[:200] + "..."
                            print(f"    {k}: {str_v}")
    except Exception as e:
        print(f"Error: {e}")
    print()

get_splits_and_configs("openbmb/Ultra-FineWeb-L3")
get_splits_and_configs("HelioAI/Claude-Opus-4.8-DeepThink-462x-105M")
