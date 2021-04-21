import numpy as np
from timeit import default_timer as timer
import numpy.linalg as lin

#Enter number of phi grid points here
grid_pts=80

#grid point identifier
x_i = np.linspace(0.,1.,grid_pts)

gamma_xy =1.5


#########################################################################################
#Below are inputs related to the potential function under study and its derivatives:


#Enter potential function here
def V0(phiX,phiY):
    return -np.cos(5.*phiX)**2  - np.cos(5.*phiY)**2 - gamma_xy*phiY*phiX

#Centering false vacuum at (0,0) to V=0
def V(phiX,phiY):
    return V0(phiX,phiY) - V0(0.,0.)

#Enter first derivative of potential wrt field phi_x
def dvdphix(phiX,phiY):
    return 5.*np.sin(10.*phiX) - gamma_xy*phiY

#Enter first derivative of potential wrt field phi_y
def dvdphiy(phiX,phiY):
    return  5.*np.sin(10.*phiY) - gamma_xy*phiX

#Enter second derivative of potential wrt field phi_x
def d2vdphixx(phiX,phiY):
    return 50.*np.cos(10.*phiX)

#Enter second derivative of potential wrt field phi_y
def d2vdphiyy(phiX,phiY):
    return  50.*np.cos(10.*phiY)

#Enter second derivative of potential wrt field phi_x and phi_y
def d2vdphixy(phiX,phiY):
    return -gamma_xy

#Must also enter all further first and second derivatives of potential here if dealing with a scenario with more than the two fieldsc onsidered in this example

#########################################################################################

#########################################################################################
# Discretisation of S_t integral and all its relevant derivatives calculated in Appendix B of paper (REF)


#Compute S_t action of a particular phi path with a given V_t grid configuration (See EqB.1 of BLAH)
def S(phi_grid,V_t_grid):
    final_sum = 0.
    for i in range(0,len(phi_grid)-1):
        sum_i = G_i(i, phi_grid)*F_i(i, phi_grid, V_t_grid)
        final_sum += sum_i
    return (27.*np.pi**2/2.)*final_sum

#See EqB.3 of BLAH
def deltaphi(i, phi_grid):
    deltaphi = phi_grid[i+1] - phi_grid[i]
    return deltaphi

#See EqB.4 of BLAH
def deltaphiDOTdeltaphi(i, phi_grid):
    deltaphiDOTdeltaphi = np.dot(deltaphi(i,phi_grid),deltaphi(i,phi_grid))
    return deltaphiDOTdeltaphi

#See EqB.7 of BLAH
def G_i(i, phi_grid):
    G_i = deltaphiDOTdeltaphi(i, phi_grid)**2
    return G_i

#See EqB.8 of BLAH
def F_i(i, phi_grid, V_t_grid):
    F_i = (( V(phi_grid[i,0],phi_grid[i,1]) + V(phi_grid[i+1,0],phi_grid[i+1,1]) - V_t_grid[i] - V_t_grid[i+1] )**2)/(V_t_grid[i+1] - V_t_grid[i])**3
    return F_i

#See EqB.19-B.26 of BLAH
def dGdPhi(i,field_idx,phi_grid):
    dGdPhi = 4.*deltaphiDOTdeltaphi(i, phi_grid)*deltaphi(i,phi_grid)[field_idx]
    return dGdPhi

#See EqB.39-B.46 of BLAH
def dFdPhi_derivs(dPhi_idx,field_idx,phi_grid):
    if field_idx == 0:
        dVdPhi = dvdphix(phi_grid[dPhi_idx,0],phi_grid[dPhi_idx,1])
    elif field_idx == 1:
        dVdPhi = dvdphiy(phi_grid[dPhi_idx,0],phi_grid[dPhi_idx,1])
    return dVdPhi

#See EqB.39-B.46 of BLAH
def dFdPhi(dF_idx,dPhi_idx,field_idx,phi_grid, V_t_grid):
    dVdPhi = dFdPhi_derivs(dPhi_idx,field_idx,phi_grid)
    return 2.*(V(phi_grid[dF_idx,0],phi_grid[dF_idx,1])+V(phi_grid[dF_idx+1,0],phi_grid[dF_idx+1,1]) - V_t_grid[dF_idx] - V_t_grid[dF_idx+1])*dVdPhi/(V_t_grid[dF_idx+1] - V_t_grid[dF_idx])**3

#See EqB.36 of BLAH
def dSdPhi_i(i,field_idx,phi_grid,V_t_grid):   #Not for i=0 boundary point
    i += 1
    deriv_sum = dFdPhi(i-1,i,field_idx,phi_grid,V_t_grid)*G_i(i-1,phi_grid) + F_i(i-1,phi_grid, V_t_grid)*dGdPhi(i-1,field_idx,phi_grid) + dFdPhi(i,i,field_idx,phi_grid, V_t_grid)*G_i(i,phi_grid) + F_i(i,phi_grid, V_t_grid)*-dGdPhi(i,field_idx,phi_grid)
    return (27.*np.pi**2/2.)*deriv_sum

grad_S_vector = np.vectorize(dSdPhi_i)
grad_S_vector.excluded.add(1)
grad_S_vector.excluded.add(2)
grad_S_vector.excluded.add(3)

#See EqB.95-B.102 of BLAH
def d2GdPhi2(i,base_r_fld_idx,base_c_fld_idx,phi_grid):
    return 4.*deltaphiDOTdeltaphi(i, phi_grid) + 8.*deltaphi(i, phi_grid)[base_r_fld_idx]*deltaphi(i, phi_grid)[base_c_fld_idx]

#See EqB.63-B.81 of BLAH
def d2FdPhi2_derivs(d2Phi_idx,base_r_fld_idx,base_c_fld_idx,phi_grid,V_t_grid):
    if base_r_fld_idx == 0:
        dVdPhi_r = dvdphix(phi_grid[d2Phi_idx,0],phi_grid[d2Phi_idx,1])
        if base_c_fld_idx == 0:
            dVdPhi_c = dvdphix(phi_grid[d2Phi_idx,0],phi_grid[d2Phi_idx,1])
            d2VdPhi2 = d2vdphixx(phi_grid[d2Phi_idx,0],phi_grid[d2Phi_idx,1])
        elif base_c_fld_idx == 1:
            dVdPhi_c = dvdphiy(phi_grid[d2Phi_idx,0],phi_grid[d2Phi_idx,1])
            d2VdPhi2 = d2vdphixy(phi_grid[d2Phi_idx,0],phi_grid[d2Phi_idx,1])
    elif base_r_fld_idx == 1:
        dVdPhi_r = dvdphiy(phi_grid[d2Phi_idx,0],phi_grid[d2Phi_idx,1])
        if base_c_fld_idx == 0:
            dVdPhi_c = dvdphix(phi_grid[d2Phi_idx,0],phi_grid[d2Phi_idx,1])
            d2VdPhi2 = d2vdphixy(phi_grid[d2Phi_idx,0],phi_grid[d2Phi_idx,1])
        elif base_c_fld_idx == 1:
            dVdPhi_c = dvdphiy(phi_grid[d2Phi_idx,0],phi_grid[d2Phi_idx,1])
            d2VdPhi2 = d2vdphiyy(phi_grid[d2Phi_idx,0],phi_grid[d2Phi_idx,1])
    return dVdPhi_r, dVdPhi_c, d2VdPhi2

#See EqB.63-B.71 of BLAH
def d2FdPhi2_ii(d2F_idx,d2Phi_idx,base_r_fld_idx,base_c_fld_idx,phi_grid,V_t_grid):
    dVdPhi_r, dVdPhi_c, d2VdPhi2= d2FdPhi2_derivs(d2Phi_idx,base_r_fld_idx,base_c_fld_idx,phi_grid,V_t_grid)
    return 2.*(V(phi_grid[d2F_idx,0],phi_grid[d2F_idx,1])+V(phi_grid[d2F_idx+1,0],phi_grid[d2F_idx+1,1]) - V_t_grid[d2F_idx] - V_t_grid[d2F_idx+1])*d2VdPhi2/(V_t_grid[d2F_idx+1] - V_t_grid[d2F_idx])**3 + 2.*dVdPhi_r*dVdPhi_c/(V_t_grid[d2F_idx+1] - V_t_grid[d2F_idx])**3

#See EqB.72-B.81 of BLAH
def d2FdPhi2_iipm1(d2F_idx,d2Phi_idx1,d2Phi_idx2,base_r_fld_idx,base_c_fld_idx,phi_grid,V_t_grid):
    dVdPhi_i_r = d2FdPhi2_derivs(d2Phi_idx1,base_r_fld_idx,base_c_fld_idx,phi_grid,V_t_grid)[0]
    dVdPhi_ipm1_c = d2FdPhi2_derivs(d2Phi_idx2,base_r_fld_idx,base_c_fld_idx,phi_grid,V_t_grid)[1]
    return 2.*dVdPhi_i_r*dVdPhi_ipm1_c/(V_t_grid[d2F_idx+1] - V_t_grid[d2F_idx])**3

#See EqB.54-B.56 of BLAH
def d2SdPhi2_ii(i,base_r_fld_idx,base_c_fld_idx,phi_grid,V_t_grid):
    i += 1
    deriv_sum = d2FdPhi2_ii(i-1,i,base_r_fld_idx,base_c_fld_idx,phi_grid, V_t_grid)*G_i(i-1, phi_grid) + F_i(i-1,phi_grid,V_t_grid)*d2GdPhi2(i-1,base_r_fld_idx,base_c_fld_idx,phi_grid) + d2FdPhi2_ii(i,i,base_r_fld_idx,base_c_fld_idx,phi_grid,V_t_grid)*G_i(i,phi_grid) + F_i(i,phi_grid,V_t_grid)*d2GdPhi2(i,base_r_fld_idx,base_c_fld_idx,phi_grid)
    + dFdPhi(i-1,i,base_r_fld_idx, phi_grid, V_t_grid)*dGdPhi(i-1,base_c_fld_idx,phi_grid) + dFdPhi(i-1,i,base_c_fld_idx, phi_grid, V_t_grid)*dGdPhi(i-1,base_r_fld_idx,phi_grid) + dFdPhi(i,i,base_r_fld_idx, phi_grid, V_t_grid)*-dGdPhi(i,base_c_fld_idx,phi_grid) + dFdPhi(i,i,base_c_fld_idx, phi_grid, V_t_grid)*-dGdPhi(i,base_r_fld_idx,phi_grid)
    return (27.*np.pi**2/2.)*deriv_sum

#See EqB.59-B.60 of BLAH (this is 1 equation)
def d2SdPhi2_iim1(i,base_r_fld_idx,base_c_fld_idx,phi_grid,V_t_grid):
    i += 1
    backward_deriv_sum = d2FdPhi2_iipm1(i-1,i,i-1,base_r_fld_idx,base_c_fld_idx,phi_grid, V_t_grid)*G_i(i-1, phi_grid) + F_i(i-1,phi_grid,V_t_grid)*-d2GdPhi2(i-1,base_c_fld_idx,base_r_fld_idx,phi_grid)
    + dFdPhi(i-1,i,base_r_fld_idx, phi_grid, V_t_grid)*-dGdPhi(i-1,base_c_fld_idx,phi_grid) + dFdPhi(i-1,i-1,base_c_fld_idx, phi_grid, V_t_grid)*dGdPhi(i-1,base_r_fld_idx,phi_grid)
    return (27.*np.pi**2/2.)*backward_deriv_sum

#See EqB.61-B.62 of BLAH (this is 1 equation)
def d2SdPhi2_iip1(i,base_r_fld_idx,base_c_fld_idx,phi_grid,V_t_grid):
    i += 1
    forward_deriv_sum = d2FdPhi2_iipm1(i,i,i+1,base_r_fld_idx,base_c_fld_idx,phi_grid,V_t_grid)*G_i(i,phi_grid) + F_i(i,phi_grid,V_t_grid)*-d2GdPhi2(i,base_c_fld_idx,base_r_fld_idx,phi_grid)
    + dFdPhi(i,i,base_r_fld_idx, phi_grid, V_t_grid)*dGdPhi(i,base_c_fld_idx,phi_grid) + dFdPhi(i,i+1,base_c_fld_idx, phi_grid, V_t_grid)*-dGdPhi(i,base_r_fld_idx,phi_grid)
    return (27.*np.pi**2/2.)*forward_deriv_sum

middle_diag = np.vectorize(d2SdPhi2_ii)
middle_diag.excluded.add(1)
middle_diag.excluded.add(2)
middle_diag.excluded.add(3)
middle_diag.excluded.add(4)

upper_diag = np.vectorize(d2SdPhi2_iip1)
upper_diag.excluded.add(1)
upper_diag.excluded.add(2)
upper_diag.excluded.add(3)
upper_diag.excluded.add(4)

lower_diag = np.vectorize(d2SdPhi2_iim1)
lower_diag.excluded.add(1)
lower_diag.excluded.add(2)
lower_diag.excluded.add(3)
lower_diag.excluded.add(4)

#########################################################################################

#########################################################################################
#Below are boundary derivatives at phi0 for gradient descent step of algorithm:

#See EqB.47-B.48 of BLAH
def dF0dPhi_derivs(dPhi_idx,field_idx,phi_grid, V_t_grid):
    if field_idx == 0:
        dVdPhi = dvdphix(phi_grid[dPhi_idx,0],phi_grid[dPhi_idx,1])
    elif field_idx == 1:
        dVdPhi = dvdphiy(phi_grid[dPhi_idx,0],phi_grid[dPhi_idx,1])
    return dVdPhi

#See EqB.47-B.48 of BLAH
def dF0dPhi_0(field_idx,phi_grid, V_t_grid):
    dVdPhi = dF0dPhi_derivs(0,field_idx,phi_grid, V_t_grid)
    return 3.*(V(phi_grid[1,0],phi_grid[1,1]) - V_t_grid[1])**2*dVdPhi/(V_t_grid[1] - V_t_grid[0])**4

#See EqB.37-B.38 of BLAH
def dSdPhi_0(field_idx,phi_grid,V_t_grid):
    deriv_sum = dF0dPhi_0(field_idx,phi_grid, V_t_grid)*G_i(0,phi_grid) + F_i(0,phi_grid, V_t_grid)*-dGdPhi(0,field_idx,phi_grid)
    return (27.*np.pi**2/2.)*deriv_sum
#########################################################################################

#########################################################################################
#Below collects previously defined derivatives into a Hessian and gradient S_t array to use with the Newton method step of the algorithm:

#Generate non-boundary gradient S_t entries
def generate_grad_S(field_idx,phi_grid,V_t_grid):
    m = len(phi_grid)
    grad_S = np.zeros(m-2)
    i = np.arange(m-2)
    grad_S[i] = grad_S_vector(i,field_idx,phi_grid,V_t_grid)
    return grad_S

#Generate a diagonal block of full Hessian matrix i.e. H_xx or H_yy
def generate_diag_block(field_idx, phi_grid, V_t_grid):
    m = len(phi_grid)
    block = np.zeros((m-2,m-2))   # m-2 as does not contain boundary points at phi_0 and phi_+
    i=np.arange(m-2)
    diag = np.zeros(m-2)
    block[i[:-1],i[1:]] = upper_diag(i[:-1],field_idx,field_idx, phi_grid,V_t_grid) #upper diagonal entries
    diag[i] = middle_diag(i,field_idx,field_idx,phi_grid,V_t_grid)    #diagonal entries
    return block, diag

#Generate an off-diagonal block of overall Hessian matrix i.e. H_xy
def generate_off_diag_block(row_field_idx, col_field_idx,  phi_grid, V_t_grid):
    m = len(phi_grid)
    block = np.zeros((m-2,m-2))   # m-2 as does not contain boundary points at phi_0 and phi_+
    i=np.arange(m-2)
    block[i[:-1],i[1:]] = upper_diag(i[:-1],row_field_idx,col_field_idx, phi_grid,V_t_grid) #upper diagonal entries
    block[i,i] = middle_diag(i,row_field_idx,col_field_idx,phi_grid,V_t_grid) #diagonal entries
    block[i[1:],i[0:-1]] = lower_diag(i[1:],row_field_idx,col_field_idx,phi_grid,V_t_grid) #lower diagonal entries
    return block

# Data mask used to populate Hessian blocks with data
def generate_mask(phi_grid,row_field_idx,col_field_idx):
    num_fields = len(phi_grid[0,:])
    num_phi_pts = len(phi_grid)-2
    mask = np.zeros((num_phi_pts*num_fields,num_phi_pts*num_fields))
    mask[0+(row_field_idx*num_phi_pts):num_phi_pts+(row_field_idx*num_phi_pts), 0+(col_field_idx*num_phi_pts):num_phi_pts+(col_field_idx*num_phi_pts)] = True
    return mask

#Construct full Hessian for 2 field example from Eq5.12 from BLAH (if #fields>2 must add further Hessian blocks to match #fields considered):
def full_hessian(phi_grid, V_t_grid):
    num_fields = len(phi_grid[0,:])
    num_phi_pts = len(phi_grid)-2
    block_00, diag_00 = generate_diag_block(0, phi_grid, V_t_grid)
    block_11, diag_11 = generate_diag_block(1, phi_grid, V_t_grid)
    block_01 = generate_off_diag_block(0,1,phi_grid, V_t_grid)
    hessian = np.zeros((num_phi_pts*num_fields,num_phi_pts*num_fields))
    hessian[generate_mask(phi_grid,0,1)==1] = block_01.reshape(num_phi_pts*num_phi_pts)
    hessian[generate_mask(phi_grid,0,0)==1] = block_00.reshape(num_phi_pts*num_phi_pts)
    hessian[generate_mask(phi_grid,1,1)==1] = block_11.reshape(num_phi_pts*num_phi_pts)
    hessian = hessian + np.transpose(hessian)
    hessian[np.arange(num_phi_pts*num_fields),np.arange(num_phi_pts*num_fields)] = np.hstack((diag_00, diag_11))
    return hessian
#####################################################################################################

#########################################################################################
#Below performs Newtons method algorithm for a fixed phi_0 release point

#Update phi path grid values using Netwons method equation for fixed phi_0 value (see Eq5.10 of BLAH):
def update_grid_values(phi_grid,V_t_grid,upsilon):
    for i in enumerate(phi_grid[0,:]):
        if i[0]==0:
            gradient_S = generate_grad_S(i[0],phi_grid,V_t_grid)
        else:
            gradient_S = np.hstack((gradient_S,generate_grad_S(i[0],phi_grid,V_t_grid)))
    update_values = upsilon*lin.inv(full_hessian(phi_grid,V_t_grid)).dot(np.transpose(gradient_S))
    split_array = np.split(update_values,2)
    update_values = np.transpose(np.vstack((split_array[0],split_array[1])))
    return phi_grid - np.pad(update_values, (1,1), 'constant')[:,1:-1]

path_history = [] #store path history
action_list = []#store S_t action history

#Iterativley update phi path to return path with smallest action for fixed phi_0 value:
def newtons_method(base_grid, V_t_grid, upsilon_init):
    upsilon, error = upsilon_init,100.
    old_phi_grid, old_action, rewind_2_phi_grid, rewind_2_action = base_grid, S(base_grid,V_t_grid), base_grid,S(base_grid,V_t_grid)
    while(abs(error)>1.e-4*upsilon):
        new_phi_grid = update_grid_values(old_phi_grid,V_t_grid,upsilon)
        new_action = S(new_phi_grid,V_t_grid)
        error =  (old_action - new_action)/((abs(old_action) + abs(new_action))/2.)
        if( error > 0.):
            rewind_2_phi_grid, rewind_2_action = old_phi_grid, old_action
            old_phi_grid, old_action = new_phi_grid, new_action
        elif(error<0.):
            upsilon = upsilon/2.
            old_phi_grid, old_action = rewind_2_phi_grid, rewind_2_action
    #path_history.append(old_phi_grid)
    return S(old_phi_grid,V_t_grid) , old_phi_grid
#########################################################################################

#########################################################################################
#Below sets up and runs Newtons method algorithm to compute the field space path with minimum S_t for a fixed phi_0 relase point

#initial phi path guess distrubutes points evenly between false vacuum and phi_0
def construct_grid(phi0,grid_pts):
    x = np.linspace(phi0[0],0.,  grid_pts)
    y = np.linspace(phi0[1],0.,  grid_pts)
    grid = np.transpose(np.vstack((x,y)))
    return grid

#Generate V_t grid and initial guess phi path based on phi_0 value considered
def generate_grids(phi_0,grid_pts):
    phi_grid = construct_grid(phi_0,grid_pts)
    x_i = np.linspace(0.,1.,grid_pts)
    V_t_grid = V(phi_0[0],phi_0[1]) + x_i**2*(3.-2.*x_i)*(V(0.,0.) - V(phi_0[0],phi_0[1]))  #V_t point distrubution function. Can choose depending on the numerical needs of the problem as log as V_t=V at the boundaries.
    return phi_grid, V_t_grid

#Returns path with minimum S_t action for fixed phi_0 release value
def min_action_for_fixed_phi0(phi0_val,grid_pts,upsilon_init):
    print("Calculating for phi0=",phi0_val,"...")
    init_grid, V_t_grid = generate_grids(phi0_val,grid_pts)
    action_val, final_phi_grid = newtons_method(init_grid,V_t_grid,upsilon_init)
    return action_val, final_phi_grid, V_t_grid, init_grid
#########################################################################################


#########################################################################################
#Below performs gradient descent to caculate the phi_0 release value that has a field space path with minimum S_t action

#Update phi_0 entries using gradient descent equation from Eq5.17 of BLAH for 2 field example (if #fields>2 must add further grad_S entries to match #fields considered):
def find_new_phi0_val(phi0_val,grid,V_t_grid,delta):
    grad_S = np.zeros(2)
    grad_S[0], grad_S[1] = dSdPhi_0(0,grid,V_t_grid), dSdPhi_0(1,grid,V_t_grid)
    norm_grad = (grad_S[0]**2 + grad_S[1]**2)**(0.5)
    new_phi0_val = np.zeros(2)
    new_phi0_val[0], new_phi0_val[1] = phi0_val[0] - delta*grad_S[0]*(1./norm_grad), phi0_val[1] - delta*grad_S[1]*(1./norm_grad)
    return new_phi0_val

#Iteratively update phi_0 and return path with minimum S_t action after choosing: initial phi_0 guess, #path points, initial upsilon, initial delta value
def find_min_action(init_phi0,grid_pts,upsilon_init,delta,converge_thresh): #upsilon= adpative step size parameter for Newtons algorithm, delta= adpative step size parameter for gradient descent algorithm
    start, error, counter  = timer(), 1., 0
    old_action_val, old_grid, old_V_t_grid, init_guess_grid = min_action_for_fixed_phi0(init_phi0,grid_pts,upsilon_init)
    old_phi0_val = init_phi0
    rewind_2_action_val, rewind_2_grid, rewind_2_V_t_grid, rewind_2_phi0_val = old_action_val, old_grid, old_V_t_grid, old_phi0_val
    action_list = [[counter, old_action_val,old_phi0_val]]
    while(abs(error) > converge_thresh*delta):
        new_phi0_val = find_new_phi0_val(old_phi0_val,old_grid,old_V_t_grid,delta)
        new_action_val, new_grid, new_V_t_grid, init_grid = min_action_for_fixed_phi0(new_phi0_val,grid_pts,upsilon_init)
        error = (old_action_val - new_action_val)/((np.abs(old_action_val) + np.abs(new_action_val))/2.)
        if(error > 0.):
            counter += 1   # This counts the number of iterations
            print('Iteration number {} completed. Phi_0 val= {}. Action value is {}. Action error is {}'.format(counter,new_phi0_val,new_action_val,error))
            action_list.append([counter, new_action_val,new_phi0_val])
            rewind_2_action_val, rewind_2_grid, rewind_2_V_t_grid, rewind_2_phi0_val = old_action_val, old_grid, old_V_t_grid, old_phi0_val
            old_action_val, old_grid, old_V_t_grid, old_phi0_val = new_action_val, new_grid, new_V_t_grid, new_phi0_val
        elif(error < 0.):
            delta = delta/2.
            old_action_val, old_grid, old_V_t_grid, old_phi0_val = rewind_2_action_val, rewind_2_grid, rewind_2_V_t_grid, rewind_2_phi0_val
            print("Error is,",error,",Reducing delta step size to:",delta, "need error <",converge_thresh*delta,"for convergence")
    end = timer()
    print('Converged after {} iterations. Minumum action={} with min phi_0 ={}. Process took time={} secs'.format(counter,old_action_val,old_phi0_val,end - start))
    return old_phi0_val, old_action_val, old_V_t_grid, old_grid, init_guess_grid, np.array(action_list)
#########################################################################################

