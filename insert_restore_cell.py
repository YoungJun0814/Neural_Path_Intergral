
import json
import os

notebook_path = r'c:\Users\Jun\Desktop\Thesis\Projects\Neural_Path_Integral\notebooks\09_Final_Model_Comparison.ipynb'
restore_cell_path = r'c:\Users\Jun\Desktop\Thesis\Projects\Neural_Path_Integral\restore_models.json'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(restore_cell_path, 'r', encoding='utf-8') as f:
    restore_data = json.load(f)
    restore_cells = restore_data['cells']

# Find index of "Part 4: Static Visualization"
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'markdown' and 'Part 4:' in ''.join(cell.get('source', [])):
        insert_idx = i
        break

if insert_idx != -1:
    # Insert before Part 4
    for cell in reversed(restore_cells):
        nb['cells'].insert(insert_idx, cell)
    print(f"Successfully inserted restoration cells before Part 4 at index {insert_idx}.")
else:
    # Append if not found (fallback)
    nb['cells'].extend(restore_cells)
    print("Part 4 not found. Appended cells to the end.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
