
import json
import os

notebook_path = r'c:\Users\Jun\Desktop\Thesis\Projects\Neural_Path_Integral\notebooks\09_Final_Model_Comparison.ipynb'
new_cells_path = r'c:\Users\Jun\Desktop\Thesis\Projects\Neural_Path_Integral\cells_to_add.json'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Locate and remove placeholder cell
cells = nb['cells']
if cells:
    last_cell = cells[-1]
    # Check if it's the specific placeholder cell
    if last_cell.get('cell_type') == 'markdown' and \
       len(last_cell.get('source', [])) > 2 and \
       "Visualization sections will be added next" in last_cell['source'][-1]:
        print("Found placeholder cell. Removing it...")
        cells.pop()

with open(new_cells_path, 'r', encoding='utf-8') as f:
    new_data = json.load(f)
    new_cells = new_data['cells']

# Append new cells
cells.extend(new_cells)
nb['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully updated notebook. Appended {len(new_cells)} cells.")
