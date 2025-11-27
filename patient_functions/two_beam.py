import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import h5py



    
"""

def plot_weight_search(patient, gaze_angle_key_1, gaze_angle_key_2, n_steps=10):
    n_plots = len(patient.roi_names)
    ws, costs, gaze_angle_dvhs = full_weight_search(
        patient=patient,
        gaze_angle_key_1=gaze_angle_key_1,
        gaze_angle_key_2=gaze_angle_key_2,
        full_output=True, 
        n_steps=n_steps
        )
    
    opt_idx = np.argmin(np.asarray(costs))
    opt_cost = costs[opt_idx]
    opt_w = ws[opt_idx]
    
    norm = plt.Normalize(min(costs), max(costs))
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(4, 3, figsize=(10,10), constrained_layout=True)
    plt.suptitle(f'{gaze_angle_key_1} + {gaze_angle_key_2} ({patient.patient_id})')
    ax = axes.flatten()[0]
    ax.scatter(ws, costs, c=costs, cmap=cmap, norm=norm, s=80)
    ax.scatter(opt_w, opt_cost, color='red', s=100)
    ax.set_title(f'Weights vs. Costs')
    ax.set_xlabel("Weight")
    ax.set_ylabel("Cost")

    dvhs_1 = patient.gaze_angle_dvhs[gaze_angle_key_1]
    dvhs_2 = patient.gaze_angle_dvhs[gaze_angle_key_2]

    for roi_name, ax in zip(patient.roi_names, axes.flatten()[1:]):
        for w, cost, gaze_angle_dvh in zip(ws, costs, gaze_angle_dvhs):
            color = 'red' if cost == opt_cost else cmap(norm(cost)) 
            z = 10 if cost==opt_cost else 1
            alpha = 1 if cost==opt_cost else 0.5
            gaze_angle_dvh.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': color, 'zorder': z, 'alpha': alpha})

            ax.set_title(roi_name)
            ax.set_xlabel("Dose (Gy)")
            ax.set_ylabel("Volume (%)")
        
        dvhs_1.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': 'black', 'ls': '--', 'label': f'{gaze_angle_key_1}', 'zorder': 2, 'lw': 1})
        dvhs_2.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': 'black', 'ls': '-.', 'label': f'{gaze_angle_key_2}', 'zorder': 2, 'lw': 1})
    plt.legend()
    plt.show()
    #plt.savefig(f'plots/{self.patient.patient_id}/two_beams/{self.gaze_angle_key_1}_{self.gaze_angle_key_2}_dvhs.png', dpi=200)

def plot_gaze_combo_heatmap(self, costs, weights, ax):

    mask = np.triu(np.ones_like(costs, dtype=bool), k=1).T
    finite_vals = costs[np.isfinite(costs)]
    vmin = finite_vals.min()
    vmax = finite_vals.max()
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mask_diag = np.eye(weights.shape[0], dtype=bool)
    weights = np.round(weights, 2).astype(str)
    weights[mask_diag] = ""  # empty string to skip annotation

    min_pos = np.unravel_index(np.argmin(costs), costs.shape)
    row, col = min_pos

    # Draw red rectangle around the cell
    rect = Rectangle((col, row), 1, 1, fill=False, edgecolor='red', linewidth=3)
    
    sns.heatmap(data=costs, annot=weights, mask=mask, xticklabels=self.patient.gaze_angle_keys, yticklabels=self.patient.gaze_angle_keys, ax=ax, cmap=cmap, vmin=norm.vmin, vmax=norm.vmax, cbar_kws={'label': 'Cost'}, fmt="")
    ax.add_patch(rect)
    return cmap, norm

                
def calculate_gaze_combos(self, n_steps=10):
    print("Calculating all gaze angle combinations...")
    n = len(self.patient.gaze_angle_keys)
    all_costs= []
    all_weights = []
    all_gaze_angle_dvhs = []

    for i, angle_1 in enumerate(self.patient.gaze_angle_keys):
        cost_row = [np.inf]*i
        weight_row = [np.inf]*i
        dvh_row = [False]*i
        for j, angle_2 in enumerate(self.patient.gaze_angle_keys[i:]):
            self.set_new_gaze_angles((angle_1, angle_2))
            w, cost, dvh = self.full_weight_search(n_steps=n_steps)
            cost_row.append(cost)
            weight_row.append(w)
            dvh_row.append(dvh)
        all_costs.append(cost_row)
        all_weights.append(weight_row)
        all_gaze_angle_dvhs.append(dvh_row)
        print_progress_bar(i+1, n)
    return np.asarray(all_weights), np.asarray(all_costs), np.asarray(all_gaze_angle_dvhs)

def plot_all_gaze_combos(self, n_steps=10):
    weights, costs, dvhs = self.calculate_gaze_combos(n_steps=n_steps)
    n_angles = len(self.patient.gaze_angle_keys)

    fig, axes = plt.subplots(4, 3, figsize=(12,12), constrained_layout=True)
    axes = axes.flatten()
    cmap, norm = self.plot_gaze_combo_heatmap(costs, weights, axes[0])

    opt_idx = np.unravel_index(costs.argmin(), costs.shape)
    opt_cost = costs[opt_idx]
    opt_w = weights[opt_idx]
    opt_dvh = dvhs[opt_idx[0]][opt_idx[1]]

    angle_1 = opt_dvh.angle_key
    angle_2 = opt_dvh.angle_key_2
    dvhs_1 = self.patient.gaze_angle_dvhs[angle_1]
    dvhs_2 = self.patient.gaze_angle_dvhs[angle_2]

    for roi_name, ax in zip(self.patient.roi_names, axes[1:]):
        for w, cost, gaze_angle_dvh in zip(weights.flatten(), costs.flatten(), dvhs.flatten()):
            if gaze_angle_dvh is not False:
                if cost == opt_cost:
                    gaze_angle_dvh.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': 'red', 'zorder': 3, 'alpha': 1, 'label': f'{np.round(opt_w,2)}*{angle_1} + {np.round(1-opt_w, 2)}*{angle_2}'})
                else:
                    color=cmap(norm(cost))
                    #gaze_angle_dvh.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': color, 'zorder': 1, 'alpha': 0.01})

        dvhs_1.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': 'black', 'ls': '--', 'label': f'{angle_1}', 'zorder': 4, 'lw': 1})
        dvhs_2.roi_dvhs[roi_name].plot(ax=ax, plot_args={'color': 'black', 'ls': '-.', 'label': f'{angle_2}', 'zorder': 4, 'lw': 1})
        ax.set_title(roi_name)
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Volume (%)")
        ax.grid()
    ax.legend()
    plt.savefig(f'plots/{self.patient.patient_id}/two_beams/find_optimal_combination_{n_angles}_angles.png', dpi=300)"""