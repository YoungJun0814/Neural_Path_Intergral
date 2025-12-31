
import json

nb_path = r'c:\Users\Jun\Desktop\Thesis\Projects\Neural_Path_Integral\notebooks\08b_3D_Visualization_Enhanced.ipynb'

print(f"Opening notebook: {nb_path}")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new source code for the visualization cell
new_viz_code = [
    "# =============================================================================\n",
    "# 1. Interactive All-in-One 3D Surface (Interpolated)\n",
    "# =============================================================================\n",
    "from scipy.interpolate import griddata\n",
    "import numpy as np\n",
    "import plotly.graph_objects as go\n",
    "\n",
    "# 1. Create Regular Grid\n",
    "# We create a smooth meshgrid for better surface visualization\n",
    "m_min, m_max = df_surface['moneyness'].min(), df_surface['moneyness'].max()\n",
    "t_min, t_max = df_surface['T'].min(), df_surface['T'].max()\n",
    "\n",
    "grid_x, grid_y = np.meshgrid(\n",
    "    np.linspace(m_min, m_max, 50),\n",
    "    np.linspace(t_min, t_max, 50)\n",
    ")\n",
    "\n",
    "# 2. Interpolation Function\n",
    "def interpolate_surface(iv_data):\n",
    "    return griddata(\n",
    "        (df_surface['moneyness'], df_surface['T']),\n",
    "        iv_data,\n",
    "        (grid_x, grid_y),\n",
    "        method='linear'\n",
    "    )\n",
    "\n",
    "# Interpolate all surfaces\n",
    "iv_heston_grid = interpolate_surface(iv_heston)\n",
    "iv_bates_grid = interpolate_surface(iv_bates)\n",
    "iv_svjj_grid = interpolate_surface(iv_svjj)\n",
    "iv_merton_grid = interpolate_surface(iv_merton)\n",
    "\n",
    "# 3. Create 3D Figure\n",
    "fig = go.Figure()\n",
    "\n",
    "# -- Market Data (Points) --\n",
    "fig.add_trace(go.Scatter3d(\n",
    "    x=df_surface['moneyness'], y=df_surface['T'], z=df_surface['iv'],\n",
    "    mode='markers', marker=dict(size=3, color='white', opacity=0.8),\n",
    "    name='Market Data (Points)'\n",
    "))\n",
    "\n",
    "# -- Model Surfaces --\n",
    "surfaces = [\n",
    "    ('Bates', iv_bates_grid, 'Viridis', True),\n",
    "    ('SVJJ', iv_svjj_grid, 'Plasma', False),\n",
    "    ('Heston', iv_heston_grid, 'Cividis', False),\n",
    "    ('Merton', iv_merton_grid, 'Blues', False)\n",
    "]\n",
    "\n",
    "for name, z_grid, color, visible in surfaces:\n",
    "    fig.add_trace(go.Surface(\n",
    "        x=grid_x, y=grid_y, z=z_grid,\n",
    "        colorscale=color,\n",
    "        opacity=0.9,\n",
    "        name=f'{name} Model',\n",
    "        visible=visible,\n",
    "        showscale=True,\n",
    "        colorbar=dict(title=f'{name} IV', x=0.9 if visible else 1.1)\n",
    "    ))\n",
    "\n",
    "fig.update_layout(\n",
    "    template='plotly_dark',\n",
    "    paper_bgcolor='black',\n",
    "    plot_bgcolor='black',\n",
    "    title='🌊 3D Implied Volatility Surface: Market (Points) vs Model (Surface)',\n",
    "    scene=dict(\n",
    "        xaxis_title='Moneyness (K/S0)',\n",
    "        yaxis_title='Time to Maturity (T)',\n",
    "        zaxis_title='Implied Volatility',\n",
    "        camera=dict(eye=dict(x=1.6, y=1.6, z=1.3))\n",
    "    ),\n",
    "    width=1000, height=800,\n",
    "    margin=dict(l=0, r=0, b=0, t=100)\n",
    ")\n",
    "\n",
    "# Add interactive buttons to switch models\n",
    "buttons = []\n",
    "for i, (name, _, _, _) in enumerate(surfaces):\n",
    "    # Visibility: Market (0) is always True. Then 4 surfaces follow.\n",
    "    # We want [True, False, False, False, False] -> Turn on index i+1\n",
    "    visibility = [True] + [False] * len(surfaces)\n",
    "    visibility[i+1] = True\n",
    "    \n",
    "    buttons.append(dict(\n",
    "        label=name,\n",
    "        method='update',\n",
    "        args=[{'visible': visibility},\n",
    "              {'title': f'🌊 {name} Model Surface vs Market Data'}]\n",
    "    ))\n",
    "\n",
    "fig.update_layout(\n",
    "    updatemenus=[dict(\n",
    "        type='buttons',\n",
    "        direction='left',\n",
    "        buttons=buttons,\n",
    "        pad={'r': 10, 't': 10},\n",
    "        showactive=True,\n",
    "        x=0.5, xanchor='center', y=1.15, yanchor='top'\n",
    "    )]\n",
    ")\n",
    "\n",
    "fig.show()\n"
]

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "1. Interactive All-in-One 3D Surface (Interpolated)" in source_str:
            print("Found the target visualization cell. Updating code...")
            cell['source'] = new_viz_code
            found = True
            break

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print("SUCCESS: Notebook updated with fixed 3D visualization code.")
else:
    print("ERROR: Could not find the visualization cell.")
