import json, os, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

base = r"c:\Users\jittu\AMD Hackathon"

# Get ALL HTML tables from Copy1 (which has the stage1 3GPP training)
path = os.path.join(base, 'phase_2_Completed_backip_AMD', 'Phase2_Training-Copy1.ipynb')
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

print(f"NOTEBOOK: Phase2_Training-Copy1.ipynb  |  Cells: {len(nb['cells'])}")

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source'])
    if 'outputs' not in cell:
        continue
    
    for out in cell['outputs']:
        if 'data' in out and 'text/html' in out['data']:
            html = ''.join(out['data']['text/html'])
            td_values = re.findall(r'<td[^>]*>([\d.]+)</td>', html)
            th_values = re.findall(r'<th[^>]*>([^<]+)</th>', html)
            
            if th_values and td_values:
                ncols = len(th_values)
                rows = [td_values[j:j+ncols] for j in range(0, len(td_values), ncols)]
                if len(rows) > 2:  # Only interesting tables
                    print(f"\nCell {i}: Table with headers {th_values}")
                    print(f"  {len(rows)} rows total:")
                    for row in rows:
                        print(f"    {row}")

# Also check Phase2_Training-Copy2.ipynb
print("\n" + "="*80)
path2 = os.path.join(base, 'phase_2_Completed_backip_AMD', 'Phase2_Training-Copy2.ipynb')
with open(path2, encoding='utf-8') as f:
    nb2 = json.load(f)

print(f"NOTEBOOK: Phase2_Training-Copy2.ipynb  |  Cells: {len(nb2['cells'])}")

for i, cell in enumerate(nb2['cells']):
    if 'outputs' not in cell:
        continue
    for out in cell['outputs']:
        if 'data' in out and 'text/html' in out['data']:
            html = ''.join(out['data']['text/html'])
            td_values = re.findall(r'<td[^>]*>([\d.]+)</td>', html)
            th_values = re.findall(r'<th[^>]*>([^<]+)</th>', html)
            if th_values and td_values:
                ncols = len(th_values)
                rows = [td_values[j:j+ncols] for j in range(0, len(td_values), ncols)]
                if len(rows) > 2:
                    print(f"\nCell {i}: Table with headers {th_values}")
                    print(f"  {len(rows)} rows total:")
                    for row in rows:
                        print(f"    {row}")
