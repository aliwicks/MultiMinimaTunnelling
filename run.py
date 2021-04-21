import Algorithm
import plotting_funcs
import matplotlib.pyplot as plt
import numpy as np

grid_pts=80                         #number of path points
upsilon_init= 1.                    #Initial step size for Newtons algortithm
delta_init= 0.1                       #Initial step size for gradient descent algortithm
init_phi0 = np.array([1.2,0.6])     #Initial phi_0 release value
converge_thresh = 1.e-4             #Convergence threshold for gradient descent algorithm

old_phi0_val, old_action_val, old_V_t_grid, old_grid, init_guess_grid, action_list = Algorithm.find_min_action(init_phi0,grid_pts,upsilon_init,delta_init,converge_thresh)

path80x = old_grid[:,0]
path80y = old_grid[:,1]

init_path80x = init_guess_grid[:,0]
init_path80y = init_guess_grid[:,1]

x = np.linspace(0,2.,300)
y = np.linspace(0.,2.,300)
phiX, phiY = np.meshgrid(x,y)

plt.contour(phiX,phiY, Algorithm.V(phiX,phiY),150,cmap='viridis')
plt.plot(path80x,path80y,label='final path',color='r')
plt.plot(init_path80x,init_path80y,label='init path',color='black')
plt.legend()
plt.colorbar()



fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

final_phi_cumulative_80pts, init_phi_cumulative_80pts, integrand_80pts, V_t_grid_flip_80pts, V_final_phi_path_flip_80pts = plotting_funcs.generate_grids_for_plot(old_grid,old_V_t_grid,init_guess_grid)
ax1.plot(final_phi_cumulative_80pts,V_t_grid_flip_80pts,'-x',label='V_t(phi) - final')
ax1.plot(final_phi_cumulative_80pts,V_final_phi_path_flip_80pts,'-x',label='V(phi) - along the path')
ax2.plot(final_phi_cumulative_80pts,integrand_80pts,'-x',label='Action',color='red')
plt.title('Min curly_phi0= '+str(round(final_phi_cumulative_80pts[-1],2))+' with action '+str(round(old_action_val,2)) )
fig.legend(loc=3)


plt.show()