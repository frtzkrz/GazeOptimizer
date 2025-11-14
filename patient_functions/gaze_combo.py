import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_gaze_combos(self):
    all_costs = []
    all_weights = []
    gaze_angle_dvhs = []
    n = len(self.gaze_angle_keys)
    for i, angle_1 in enumerate(self.gaze_angle_keys):
        
        for angle_2 in self.gaze_angle_keys[i:]:
            self.initiate_two_beams(gaze_angle_keys=(angle_1, angle_2), weight=0.5)
            w, cost = self.two_beam.full_weight_search()
            cost_row.append(cost)
            weight_row.append(w)
            gaze_angle_row.append((angle_1, angle_2))
        all_costs.append(cost_row)
        all_weights.append(weight_row)
        gaze_angle_combos.append(gaze_angle_row)
    
    mask = np.triu(np.ones_like(all_costs, dtype=bool),k=1).T
    all_costs = np.asarray(all_costs)
    all_weights = np.asarray(all_weights)
    



    fig, ax = plt.subplots()
    sns.heatmap(data=all_costs, annot=all_weights, mask=mask, xticklabels=self.gaze_angle_keys, yticklabels=self.gaze_angle_keys, cmap="coolwarm")
    plt.show()

    return all_costs, all_weights, gaze_angle_combos

def find_optimal_gaze_combo(self):
    all_costs, all_weights, gaze_angle_combos = self.calculate_gaze_combos()
    opt_idx = np.unravel_index(all_costs.argmin(), all_costs.shape)
    opt_cost = all_costs[opt_idx]
    opt_w = all_weights[opt_idx]
    opt_angles = gaze_angle_combos[opt_idx[0]][opt_idx[1]]

    return opt_cost, opt_w, opt_angles

