
import json
import os

nb_path = r'c:\Users\Jun\Desktop\Thesis\Projects\Neural_Path_Integral\notebooks\08b_3D_Visualization_Enhanced.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes = 0

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        modified = False
        for line in cell['source']:
            if "'maxiter':" in line:
                # Replace 10 (or whatever value) with 20
                # We assume the line looks like "    'maxiter': 10," or similar
                # Let's just use a robust replacement or string manipulation
                # If I simply replace "10" with "20", it might be risky if "10" appears elsewhere.
                # But looking at previous context, checking for "'maxiter':" is safe enough.
                # We will split by key and reconstructed.
                if "10" in line: 
                    new_line = line.replace("10", "20")
                    new_source.append(new_line)
                    modified = True
                    changes += 1
                elif "30" in line: # Fallback if it was reverted or I missed it
                    new_line = line.replace("30", "20")
                    new_source.append(new_line)
                    modified = True
                    changes += 1
                else:
                    new_source.append(line)
            else:
                new_source.append(line)
        
        if modified:
            cell['source'] = new_source

if changes > 0:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print(f"Updated maxiter to 20 in {changes} places.")
else:
    print("maxiter not found or already 20.")
