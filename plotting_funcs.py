from numpy import linalg as LA
import Algorithm
import numpy as np

#Functions to plot V_t and S_t vs curly-phi (field space distance)

#calculate field space distance travelled (curly-phi)
def find_cumulative_path(phi_path):
    i=np.arange(len(phi_path)-1)
    flip_phi_path = np.flip(phi_path,axis=0)
    dphi_norms= LA.norm(flip_phi_path[i+1] - flip_phi_path[i],axis=1)
    phi_cumulative = np.zeros(len(dphi_norms)+1)
    phi_cumulative[1:] = np.cumsum(dphi_norms)
    return flip_phi_path, phi_cumulative

#returns grid of curly-phis, S_t integrand contributions and V_t_grid to use for plots
def generate_grids_for_plot(final_phi_path,V_t_grid,init_phi_path):
    flip_final_phi_path, final_phi_cumulative = find_cumulative_path(final_phi_path)
    init_phi_cumulative = find_cumulative_path(init_phi_path)[1]
    V_t_grid_flip = np.flip(V_t_grid)
    V_final_phi_path_flip = Algorithm.V(flip_final_phi_path[:,0],flip_final_phi_path[:,1])
    dVtdPhi = np.gradient(V_t_grid_flip,final_phi_cumulative)
    integrand = (Algorithm.V(flip_final_phi_path[:,0],flip_final_phi_path[:,1])-V_t_grid_flip)**2/(-dVtdPhi)**3
    return final_phi_cumulative, init_phi_cumulative, integrand, V_t_grid_flip, V_final_phi_path_flip