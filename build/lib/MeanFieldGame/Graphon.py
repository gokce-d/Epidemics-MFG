# Import 
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.sparse as sparse
import scipy.sparse.linalg as sparsela
import scipy.linalg as sla
import math
from tqdm import tqdm
import numpy.linalg as la 
import pandas as pd
import random
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
random.seed(7)
import time
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('text', usetex=True)
plt.rc('font', family='serif')




#TRIAL for github





def initializer(n_blocks, n_states, Nt, lambda_s_in, lambda_i_in, lambda_r_in, lambda_type, lambda_duration, p_0):
    before_p =  np.tile(p_0, Nt)
    before_u = np.zeros((n_blocks*n_states, Nt)) 
    if lambda_type == 0: #same for every block time independent
        lambda_s = np.tile(lambda_s_in, n_blocks) #format: [1, 1, 1]
        lambda_i = np.tile(lambda_i_in, n_blocks)
        lambda_r = np.tile(lambda_r_in, n_blocks)
    if lambda_type == 1: #different for each block time independent
        lambda_s = lambda_s_in #format: [1, 1, 1]
        lambda_i = lambda_i_in
        lambda_r = lambda_r_in
    if lambda_type == 2: #different for each block time dependent       
        lambda_s = np.repeat(lambda_s_in, lambda_duration[0].astype(int), axis=1) #format[1,1,1,...,3,3,3];
        lambda_i = np.repeat(lambda_i_in, lambda_duration[1].astype(int), axis=1) #format[1,1,1,...,3,3,3];
        lambda_r = np.repeat(lambda_r_in, lambda_duration[2].astype(int), axis=1) #format[1,1,1,...,3,3,3];
    return (before_p, before_u, lambda_s, lambda_i, lambda_r)    

def Z_calculator(n_blocks, block_dens, lambda_i, graphon, p, lambda_type): # Ariticle: lambda, p
    if (lambda_type==0 or lambda_type==1):
        Z = np.matmul(graphon, np.multiply(np.reshape(np.multiply(block_dens, lambda_i), (n_blocks,1)),p[n_blocks:2*n_blocks,:]))
    if lambda_type==2:
        Z = np.matmul(graphon,np.multiply(np.reshape(block_dens,(n_blocks,1)), lambda_i)*p[n_blocks:2*n_blocks,:])
    return Z

def opt_control_calculator(lambda_s, lambda_i, lambda_r, beta, c_lambda, Z, u, n_blocks, lambda_type,Nt):
    if (lambda_type==0 or lambda_type==1):
        alpha_s = np.reshape(lambda_s, (n_blocks,1)) + np.reshape(beta/c_lambda,(n_blocks,1)) * Z * (u[0:n_blocks,:]-u[n_blocks:2*n_blocks,:]) # n*1 matrix
        alpha_i = np.tile(np.reshape(lambda_i,(n_blocks,1)),Nt)
        alpha_r = np.tile(np.reshape(lambda_r,(n_blocks,1)),Nt)
    if lambda_type==2:
        alpha_s = lambda_s + np.reshape(beta/c_lambda,(n_blocks,1)) * Z * (u[0:n_blocks,:]-u[n_blocks:2*n_blocks,:])
        alpha_i = lambda_i
        alpha_r = lambda_r
    return (alpha_s, alpha_i, alpha_r)

def rate_ODE_p(t, p, death, inter_alpha_s, inter_Z, beta, kappa, gamma, rho, n_states, n_blocks):
    alpha_s = inter_alpha_s(t)
    Z = inter_Z(t)
    rate_p_S = -beta*alpha_s*Z*p[0:n_blocks]+ kappa * p[(n_states-1)*n_blocks:n_states*n_blocks]
    rate_p_I = beta*alpha_s*Z*p[0:n_blocks] - gamma * p[n_blocks:2*n_blocks]
    # 0:n first n blocks of population for susceptible ppl, n:2n for infected ppl
    rate = []
    for k in np.arange(n_blocks):
        rate=np.append(rate, rate_p_S[k])
    for k in np.arange(n_blocks):
        rate=np.append(rate, rate_p_I[k])  
    if death==0:
        rate_p_R = gamma * p[n_blocks:2*n_blocks] - kappa * p[(n_states-1)*n_blocks:n_states*n_blocks]
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_p_R[k])   
    if death==1:
        rate_p_R = rho * gamma * p[n_blocks:2*n_blocks] - kappa * p[(n_states-1)*n_blocks:n_states*n_blocks]
        rate_p_D = (1-rho) * gamma * p[n_blocks:2*n_blocks]
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_p_R[k]) 
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_p_D[k])             
    return rate


def solver_KFP(death, inter_alpha_s, inter_Z, beta, kappa, gamma, rho, n_states, n_blocks, p_0, t_grid, T):
    sol_p = solve_ivp(rate_ODE_p, [0,T], p_0, t_eval = t_grid, args = (death, inter_alpha_s, inter_Z, beta, kappa, gamma, rho, n_states, n_blocks))    
    p = sol_p.y
    return p


def rate_ODE_u(t, u, death, lambda_type, inter_Z, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, n_blocks):
    Z = inter_Z(t) 
    if lambda_type==2:
        lambda_s = lambda_s(t)
    rate_u_S = -(beta * lambda_s * Z * (u[n_blocks:2*n_blocks]- u[0:n_blocks]) - ((beta**2)/(2*c_lambda)) * (Z**2) * (u[n_blocks:2*n_blocks]- u[0:n_blocks])**2)
    rate_u_I = -(gamma * (u[2*n_blocks:3*n_blocks]- u[n_blocks:2*n_blocks]) + c_inf)
    rate = []
    for k in np.arange(n_blocks):
        rate=np.append(rate, rate_u_S[k])
    for k in np.arange(n_blocks):
        rate=np.append(rate, rate_u_I[k]) 
    if death==0:
        rate_u_R = -(kappa*(u[0:n_blocks] - u[2*n_blocks:3*n_blocks]))
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_u_R[k])        
    if death==1:   
#         print("WHAT??")
        rate_u_R = -gamma*(kappa*(u[0:n_blocks] - u[2*n_blocks:3*n_blocks]))
        rate_u_D = - c_dead * np.ones((n_blocks))
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_u_R[k])  
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_u_D[k])  
#     print(rate)
    return rate


def solver_HJB(death, lambda_type, inter_Z, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, n_blocks, u_T, t_grid, T):
    backward_t_grid = T-t_grid ### Changing the time direction bc HJB is backward
    sol_u = solve_ivp(rate_ODE_u, [T, 0], u_T, t_eval = backward_t_grid,   
                  args = (death, lambda_type, inter_Z, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, n_blocks)) # Sovle the equations with initial value
    u = sol_u.y
#     print(u[:,-1])
#     print("flipped: ", np.flip(u,axis=1)[:,0])
    return np.flip(u,axis=1)

def stoch_block_fixed(n_blocks, n_states, Nt, lambda_s_in, lambda_i_in, lambda_r_in, \
                      graphon, beta, kappa, gamma, rho, c_lambda, c_inf, c_dead, \
                      t_grid, T, p_0, u_T, n_print, exp_id, block_dens, lambda_type, lambda_duration, death,epsilon):
    #Save Data and Experiment Details
    datafile_name = '%s.npz' % exp_id
    np.savez(datafile_name, n_blocks=n_blocks, n_states=n_states, Nt=Nt, \
             lambda_s_in=lambda_s_in, lambda_i_in=lambda_i_in, lambda_r_in=lambda_r_in, \
             graphon= graphon, beta=beta, kappa=kappa, gamma=gamma, rho=rho, c_lambda=c_lambda, c_inf=c_inf, c_dead=c_dead, \
             t_grid=t_grid, T=T, p_0=p_0, u_T=u_T, exp_id=exp_id, block_dens=block_dens, lambda_duration=lambda_duration)
    file = open("Experiments.txt", "a")
    file.write("\n" + "Exp id = " + repr(exp_id) + "\n" + "  n_blocks = " + repr(n_blocks) + "  T = " + repr(T) + \
               "  Nt = " + repr(Nt) + "  lambda_s = " + repr(lambda_s_in) + "  lambda_i = " + repr(lambda_i_in) + \
               "  lambda_r = " + repr(lambda_r_in) + "  beta = " + repr(beta) + "  kappa = " + repr(kappa) + "  rho = " + repr(rho) + \
               "  gamma = " + repr(gamma) + "  c_lambda = " + repr(c_lambda) + "  c_inf = " + repr(c_inf) + "  c_dead = " + repr(c_dead) + "\n" + \
               "  lambda_duration= " + repr(lambda_duration) + \
               "  graphon = " + repr(graphon) + "\n" + "  p_0 = " + repr(p_0) + "\n" + "\n" )
    file.close()
    #Algorithm for constant lambda
    if (lambda_type==0 or lambda_type==1):
        before_p, before_u, lambda_s, lambda_i, lambda_r  =  initializer(n_blocks, n_states, Nt, lambda_s_in, lambda_i_in, lambda_r_in, lambda_type, lambda_duration,p_0)
        Z = Z_calculator(n_blocks, block_dens, lambda_i, graphon, before_p, lambda_type)
        inter_Z = interp1d(t_grid, Z)
        alpha_s, alpha_i, alpha_r = opt_control_calculator(lambda_s, lambda_i, lambda_r, beta, c_lambda, Z, before_u, n_blocks, lambda_type,Nt)
        inter_alpha_s = interp1d(t_grid, alpha_s)
        after_p = solver_KFP(death, inter_alpha_s, inter_Z, beta, kappa, gamma, rho, n_states, n_blocks, np.reshape(p_0,(n_blocks*n_states)), t_grid, T)
        after_u = solver_HJB(death, lambda_type, inter_Z, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, n_blocks, np.reshape(u_T,(n_blocks*n_states)), t_grid, T)
        convergence_p = la.norm(after_p - before_p)
        convergence_u = la.norm(after_u - before_u)
        i=0
        print("iter: ", i, "p conv: ", convergence_p, "u conv:", convergence_u)
        while ((la.norm(after_p - before_p) > epsilon) or (la.norm(after_u - before_u) > epsilon)):
            Z = Z_calculator(n_blocks, block_dens, lambda_i, graphon, after_p, lambda_type)
            inter_Z = interp1d(t_grid, Z)
            alpha_s, alpha_i, alpha_r = opt_control_calculator(lambda_s, lambda_i, lambda_r, beta, c_lambda, Z, after_u, n_blocks, lambda_type,Nt)
            inter_alpha_s = interp1d(t_grid, alpha_s)
            before_p = after_p.copy()
            before_u = after_u.copy()
            after_p = solver_KFP(death, inter_alpha_s, inter_Z, beta, kappa, gamma, rho, n_states, n_blocks, np.reshape(p_0,(n_blocks*n_states)), t_grid, T)
            after_u = solver_HJB(death, lambda_type, inter_Z, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, n_blocks, np.reshape(u_T,(n_blocks*n_states)), t_grid, T)
            convergence_p = np.append(convergence_p, la.norm(after_p - before_p))
            convergence_u = np.append(convergence_u, la.norm(after_u - before_u))  
            i +=1
            if i % n_print == 0:
                print("iter: ", i, "p conv: ", convergence_p[-1], "u conv:", convergence_u[-1])
    #Algorithm for time dependent lambda
    if lambda_type==2:
        before_p, before_u, lambda_s, lambda_i, lambda_r  =  initializer(n_blocks, n_states, Nt, lambda_s_in, lambda_i_in, lambda_r_in, lambda_type, lambda_duration,p_0)
        inter_lambda_s = interp1d(t_grid, lambda_s, kind='previous')
        Z = Z_calculator(n_blocks, block_dens, lambda_i, graphon, before_p, lambda_type)
        inter_Z = interp1d(t_grid, Z)
        alpha_s, alpha_i, alpha_r = opt_control_calculator(lambda_s, lambda_i, lambda_r, beta, c_lambda, Z, before_u, n_blocks, lambda_type,Nt)
        inter_alpha_s = interp1d(t_grid, alpha_s)
        after_p = solver_KFP(death, inter_alpha_s, inter_Z, beta, kappa, gamma, rho, n_states, n_blocks, np.reshape(p_0,(n_blocks*n_states)), t_grid, T)
        after_u = solver_HJB(death, lambda_type, inter_Z, beta, kappa, gamma, inter_lambda_s, c_lambda, c_inf, c_dead, n_blocks, np.reshape(u_T,(n_blocks*n_states)), t_grid, T)
        convergence_p = la.norm(after_p - before_p)
        convergence_u = la.norm(after_u - before_u)
        i=0
        print("iter: ", i, "p conv: ", convergence_p, "u conv:", convergence_u)
        while ((la.norm(after_p - before_p) > epsilon) or (la.norm(after_u - before_u) > epsilon)):
            Z = Z_calculator(n_blocks, block_dens, lambda_i, graphon, after_p, lambda_type)   
            inter_Z = interp1d(t_grid, Z)
            alpha_s, alpha_i, alpha_r = opt_control_calculator(lambda_s, lambda_i, lambda_r, beta, c_lambda, Z, after_u, n_blocks, lambda_type,Nt)
            inter_alpha_s = interp1d(t_grid, alpha_s)
            before_p = after_p.copy()
            before_u = after_u.copy()
            after_p = solver_KFP(death, inter_alpha_s, inter_Z, beta, kappa, gamma, rho, n_states, n_blocks, np.reshape(p_0,(n_blocks*n_states)), t_grid, T)
            after_u = solver_HJB(death, lambda_type, inter_Z, beta, kappa, gamma, inter_lambda_s, c_lambda, c_inf, c_dead, n_blocks, np.reshape(u_T,(n_blocks*n_states)), t_grid, T)
            convergence_p = np.append(convergence_p, la.norm(after_p - before_p))
            convergence_u = np.append(convergence_u, la.norm(after_u - before_u))  
            i +=1
            if i % n_print == 0:
                print("iter: ", i, "p conv: ", convergence_p[-1], "u conv:", convergence_u[-1])      
    return (after_u, after_p, alpha_s, alpha_i, alpha_r, Z, convergence_p, convergence_u)

def plotting(t_grid, convergence_u, convergence_p, after_p, alpha_s, alpha_i, Z, nblocks, title, T, type_plot, death, i_control):
    fig, ax = plt.subplots(math.ceil(nblocks/2) +1 ,2, figsize=(10,4*(math.ceil(nblocks/2) +1)))
    i=0
    j=0
    for i in np.arange(math.ceil(nblocks/2)):
        for j in np.arange(2):
            if ((j==1) and (math.ceil(nblocks/2)!=nblocks/2) and (i==math.ceil(nblocks/2)-1)):
                break
            ax[i,j].plot(t_grid, after_p[2*i+j,:], label = "S", c='teal', linestyle="-")
            ax[i,j].plot(t_grid, after_p[nblocks + 2*i +j,:], label = "I", c='magenta', linestyle="-")
            ax[i,j].plot(t_grid, after_p[2*nblocks + 2*i+j,:], label = "R", c='olive', linestyle="-")
            if death == 1:
                ax[i,j].plot(t_grid, after_p[3*nblocks + 2*i+j,:], label = "D", c='darkorange', linestyle="-")
            ax[i,j].legend(fontsize="8")
            ax[i,j].set_title("Density: Block {}".format(2*i+j+1))
    # control and interaction plots
    if type_plot==0:
        color=iter(cm.rainbow(np.linspace(0,1,nblocks)))
        for k in np.arange(nblocks):
            c=next(color)
            ax[math.ceil(nblocks/2),0].plot(t_grid, Z[k,:], label="Z Block {}".format(k+1), c=c)
            ax[math.ceil(nblocks/2),0].legend(fontsize="8")
            ax[math.ceil(nblocks/2),0].set_title("Interaction")
            ax[math.ceil(nblocks/2),1].plot(t_grid, alpha_s[k,:], label = "S, block {}".format(k+1), c=c, linestyle="-")
            if i_control==1:
                ax[math.ceil(nblocks/2),1].plot(t_grid, alpha_i[k,:], label = "I, block {}".format(k+1), c=c, linestyle="-.")
            ax[math.ceil(nblocks/2),1].legend(fontsize="8")
            ax[math.ceil(nblocks/2),1].set_title("Controls")
        plt.setp(ax, xlim=(0,T))   
    if type_plot==1:
        ax[math.ceil(nblocks/2),0].set_yscale('log')
        ax[math.ceil(nblocks/2),0].plot(np.arange(np.size(convergence_u)), convergence_u, label="u conv.", c='magenta') # Convergence: when p^(k+1) is close to p^k
        ax[math.ceil(nblocks/2),0].plot(np.arange(np.size(convergence_p)), convergence_p, label="p conv.", c='olive')    
        ax[math.ceil(nblocks/2),0].legend(fontsize="8")
        ax[math.ceil(nblocks/2),0].set_title("Convergence")
        # control plots
        color=iter(cm.rainbow(np.linspace(0,1,nblocks)))
        for k in np.arange(nblocks):
            c=next(color)
            ax[math.ceil(nblocks/2),1].plot(t_grid, alpha_s[k,:], label = "S, block {}".format(k+1), c=c, linestyle="-")
            if i_control==1:
                ax[math.ceil(nblocks/2),1].plot(t_grid, alpha_i[k,:], label = "I, block {}".format(k+1), c=c, linestyle="-.")
            ax[math.ceil(nblocks/2),1].legend(fontsize="8")
            ax[math.ceil(nblocks/2),1].set_title("Controls")
    plt.suptitle(title)
    plt.show()
    plt.savefig('%s.eps' % title, bbox_inches='tight', format='eps')

def comparison_plotting(t_grid, after_p_1, alpha_s_1, Z_1, after_p_2, alpha_s_2, Z_2, nblocks, title, T, id_1, id_2):
    fig, (ax0, ax1, ax2)= plt.subplots(1 ,3, figsize=(16, 4.5))
    color=iter(cm.gnuplot(np.linspace(0,1,nblocks)))
    for k in np.arange(nblocks):
        c=next(color)
        ax0.plot(t_grid, after_p_1[nblocks + k,:], label="Block {block}, {id_}".format(block=k+1, id_=id_1), c=c)
        ax1.plot(t_grid, Z_1[k,:], label="Block {block}, {id_}".format(block=k+1, id_=id_1), c=c)
        ax2.plot(t_grid, alpha_s_1[k,:], label = "Block {block}, {id_}".format(block=k+1, id_=id_1), c=c)
        ax0.plot(t_grid, after_p_2[nblocks + k,:], label="Block {block}, {id_}".format(block=k+1, id_=id_2), c=c, linestyle = '-.')
        ax1.plot(t_grid, Z_2[k,:], label="Block {block}, {id_}".format(block=k+1, id_=id_2), c=c, linestyle = '-.')
        ax2.plot(t_grid, alpha_s_2[k,:], label = "Block {block}, {id_}".format(block=k+1, id_=id_2), c=c, linestyle = '-.')
                           
    ax0.legend(fontsize="8")   
    ax0.set_title("Infected Density")
    ax1.legend(fontsize="8")   
    ax1.set_title("Interaction")     
    ax2.legend(fontsize="8")
    ax2.set_title("Controls")  
    ax0.set_xlim(0,T)
    ax1.set_xlim(0,T)
    ax2.set_xlim(0,T)    

    plt.suptitle(title)
    plt.savefig('%s.eps' % title, bbox_inches='tight', format='eps')
    
    




















def plot_control_only(t_grid, alpha_s, alpha_i, nblocks, title, T, i_control=1):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    color = iter(cm.rainbow(np.linspace(0, 1, nblocks)))

    for k in np.arange(nblocks):
        c = next(color)
        ax.plot(t_grid, alpha_s[k, :], label="S, block {}".format(k+1), c=c, linestyle="-")
        if i_control == 1:
            ax.plot(t_grid, alpha_i[k, :], label="I, block {}".format(k+1), c=c, linestyle="-.")

    ax.legend(fontsize="8")
    ax.set_title("Controls")
    plt.setp(ax, xlim=(0, T))
    plt.suptitle(title)
    plt.show()











def contact_rate_control_calc(lambda_s, lambda_i, lambda_r, beta, c_lambda, c_nu, Z, u, n_blocks, lambda_type, kappa, Nt):
    if (lambda_type==0 or lambda_type==1):
        alpha_s = np.reshape(lambda_s, (n_blocks,1)) + np.reshape(beta/(2*c_lambda),(n_blocks,1)) * Z * (u[0:n_blocks,:]-u[n_blocks:2*n_blocks,:])
        alpha_i = np.tile(np.reshape(lambda_i,(n_blocks,1)),Nt)
        alpha_r = np.tile(np.reshape(lambda_r,(n_blocks,1)),Nt)
        nu      = np.reshape(kappa/(2*c_nu),(n_blocks,1)) * (u[0:n_blocks,:]-u[2*n_blocks:3*n_blocks,:])
    if lambda_type==2:
        alpha_s = lambda_s + np.reshape(beta/c_lambda,(n_blocks,1)) * Z * (u[0:n_blocks,:]-u[n_blocks:2*n_blocks,:])
        alpha_i = lambda_i
        alpha_r = lambda_r
        nu      = np.reshape(kappa/(2*c_lambda),(n_blocks,1)) * (u[0:n_blocks,:]-u[2*n_blocks:3*n_blocks,:])
    return (alpha_s, alpha_i, alpha_r, nu)







def cdc_rate_ODE_p(t, p, death, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, rho, n_states, n_blocks):
    alpha_s = inter_alpha_s(t)
    nu      = inter_nu(t)
    Z = inter_Z(t)
    rate_p_S = -beta*alpha_s*Z*p[0:n_blocks]- kappa * nu * p[0:n_blocks]
    # p[(n_states-1)*n_blocks:n_states*n_blocks]
    rate_p_I = beta*alpha_s*Z*p[0:n_blocks] - gamma * p[n_blocks:2*n_blocks]
    rate = []
    for k in np.arange(n_blocks):
        rate=np.append(rate, rate_p_S[k])
    for k in np.arange(n_blocks):
        rate=np.append(rate, rate_p_I[k])  
    if death==0:
        rate_p_R = gamma * p[n_blocks:2*n_blocks] + kappa * nu * p[0:n_blocks]
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_p_R[k])   
    if death==1:
        #can be changed after CDC submission
        rate_p_R = rho * gamma * p[n_blocks:2*n_blocks] - kappa * p[(n_states-1)*n_blocks:n_states*n_blocks]
        rate_p_D = (1-rho) * gamma * p[n_blocks:2*n_blocks]
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_p_R[k]) 
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_p_D[k])             
    return rate


def cdc_solver_KFP(death, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, rho, n_states, n_blocks, p_0, t_grid, T):
    sol_p = solve_ivp(cdc_rate_ODE_p, [0,T], p_0, t_eval = t_grid, args = (death, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, rho, n_states, n_blocks))    
    p = sol_p.y
    return p






def cdc_rate_ODE_u(t, u, death, lambda_type, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, c_nu, n_blocks):
    Z  = inter_Z(t) 
    nu = inter_nu(t)
    alpha_s = inter_alpha_s(t)
    if lambda_type==2:
        lambda_s = lambda_s(t)
    rate_u_S = beta * alpha_s * Z * (u[0:n_blocks]-u[n_blocks:2*n_blocks]) - c_lambda*((lambda_s-alpha_s)**2) + kappa*nu*(u[0:n_blocks]-u[2*n_blocks:3*n_blocks]) - c_nu*((nu)**2)
                # ((beta**2)/(2*c_lambda)) * (Z**2) * (u[n_blocks:2*n_blocks]- u[0:n_blocks])**2)
    rate_u_I = -(gamma * (u[2*n_blocks:3*n_blocks]- u[n_blocks:2*n_blocks]) + c_inf)
    rate = []
    for k in np.arange(n_blocks):
        rate=np.append(rate, rate_u_S[k])
    for k in np.arange(n_blocks):
        rate=np.append(rate, rate_u_I[k]) 
    if death==0:
        # rate_u_R = -(kappa*(u[0:n_blocks] - u[2*n_blocks:3*n_blocks]))
        rate_u_R = 0 * alpha_s
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_u_R[k])        
    if death==1:   
#         change after CDC 2025 submission
        rate_u_R = -gamma*(kappa*(u[0:n_blocks] - u[2*n_blocks:3*n_blocks]))
        rate_u_D = - c_dead * np.ones((n_blocks))
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_u_R[k])  
        for k in np.arange(n_blocks):
            rate=np.append(rate, rate_u_D[k])  
    return rate


def cdc_solver_HJB(death, lambda_type, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, c_nu, n_blocks, u_T, t_grid, T):
    backward_t_grid = T-t_grid
    sol_u = solve_ivp(cdc_rate_ODE_u, [T, 0], u_T, t_eval = backward_t_grid, \
                  args = (death, lambda_type, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, c_nu, n_blocks)) 
    u = sol_u.y
    return np.flip(u,axis=1)








def simulateEQ_contact_rate_vaccination(n_blocks, n_states, Nt, lambda_s_in, lambda_i_in, lambda_r_in, \
                      graphon, beta, kappa, gamma, rho, c_lambda, c_inf, c_dead, c_nu, \
                      t_grid, T, p_0, u_T, n_print, exp_id, block_dens, lambda_type, lambda_duration, death, epsilon):
    #Algorithm for constant lambda
    if (lambda_type==0 or lambda_type==1):
        before_p, before_u, lambda_s, lambda_i, lambda_r  =  initializer(n_blocks, n_states, Nt, lambda_s_in, lambda_i_in, lambda_r_in, lambda_type, lambda_duration, p_0)
        Z = Z_calculator(n_blocks, block_dens, lambda_i, graphon, before_p, lambda_type)
        inter_Z = interp1d(t_grid, Z)
        alpha_s, alpha_i, alpha_r, nu = contact_rate_control_calc(lambda_s, lambda_i, lambda_r, beta, c_lambda, c_nu, Z, before_u, n_blocks, lambda_type, kappa, Nt)
        inter_alpha_s = interp1d(t_grid, alpha_s)
        inter_nu      = interp1d(t_grid, nu)
        after_p = cdc_solver_KFP(death, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, rho, n_states, n_blocks, np.reshape(p_0,(n_blocks*n_states)), t_grid, T)
        after_u = cdc_solver_HJB(death, lambda_type, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, c_nu, n_blocks, np.reshape(u_T,(n_blocks*n_states)), t_grid, T)
        convergence_p = la.norm(after_p - before_p)
        convergence_u = la.norm(after_u - before_u)
        i=0
        print("iter: ", i, "p conv: ", convergence_p, "u conv:", convergence_u)
        while ((la.norm(after_p - before_p) > epsilon) or (la.norm(after_u - before_u) > epsilon)):
            Z = Z_calculator(n_blocks, block_dens, lambda_i, graphon, after_p, lambda_type)
            inter_Z = interp1d(t_grid, Z)
            alpha_s, alpha_i, alpha_r, nu = contact_rate_control_calc(lambda_s, lambda_i, lambda_r, beta, c_lambda, c_nu, Z, after_u, n_blocks, lambda_type, kappa, Nt)
            inter_alpha_s = interp1d(t_grid, alpha_s)
            inter_nu      = interp1d(t_grid, nu)
            before_p = after_p.copy()
            before_u = after_u.copy()
            after_p = cdc_solver_KFP(death, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, rho, n_states, n_blocks, np.reshape(p_0,(n_blocks*n_states)), t_grid, T)
            after_u = cdc_solver_HJB(death, lambda_type, inter_alpha_s, inter_Z, inter_nu, beta, kappa, gamma, lambda_s, c_lambda, c_inf, c_dead, c_nu, n_blocks, np.reshape(u_T,(n_blocks*n_states)), t_grid, T)
            convergence_p = np.append(convergence_p, la.norm(after_p - before_p))
            convergence_u = np.append(convergence_u, la.norm(after_u - before_u))  
            i +=1
            if i % n_print == 0:
                print("iter: ", i, "p conv: ", convergence_p[-1], "u conv:", convergence_u[-1])
    #Algorithm for time dependent lambda: will change after CDC2025
    # if lambda_type==2:
    #     before_p, before_u, lambda_s, lambda_i, lambda_r  =  initializer(n_blocks, n_states, Nt, lambda_s_in, lambda_i_in, lambda_r_in, lambda_type, lambda_duration)
    #     inter_lambda_s = interp1d(t_grid, lambda_s, kind='previous')
    #     Z = Z_calculator(n_blocks, block_dens, lambda_i, graphon, before_p, lambda_type)
    #     inter_Z = interp1d(t_grid, Z)
    #     alpha_s, alpha_i, alpha_r = contact_rate_control_calc(lambda_s, lambda_i, lambda_r, beta, c_lambda, Z, before_u, n_blocks, lambda_type, kappa)
    #     inter_alpha_s = interp1d(t_grid, alpha_s)
    #     after_p = cdc_solver_KFP(death, inter_alpha_s, inter_Z, beta, kappa, gamma, rho, n_states, n_blocks, np.reshape(p_0,(n_blocks*n_states)), t_grid, T)
    #     after_u = cdc_solver_HJB(death, lambda_type, inter_Z, beta, kappa, gamma, inter_lambda_s, c_lambda, c_inf, c_dead, n_blocks, np.reshape(u_T,(n_blocks*n_states)), t_grid, T)
    #     convergence_p = la.norm(after_p - before_p)
    #     convergence_u = la.norm(after_u - before_u)
    #     i=0
    #     print("iter: ", i, "p conv: ", convergence_p, "u conv:", convergence_u)
    #     while ((la.norm(after_p - before_p) > epsilon) or (la.norm(after_u - before_u) > epsilon)):
    #         Z = Z_calculator(n_blocks, block_dens, lambda_i, graphon, after_p, lambda_type)   
    #         inter_Z = interp1d(t_grid, Z)
    #         alpha_s, alpha_i, alpha_r = contact_rate_control_calc(lambda_s, lambda_i, lambda_r, beta, c_lambda, Z, after_u, n_blocks, lambda_type, kappa)
    #         inter_alpha_s = interp1d(t_grid, alpha_s)
    #         before_p = after_p.copy()
    #         before_u = after_u.copy()
    #         after_p = cdc_solver_KFP(death, inter_alpha_s, inter_Z, beta, kappa, gamma, rho, n_states, n_blocks, np.reshape(p_0,(n_blocks*n_states)), t_grid, T)
    #         after_u = cdc_solver_HJB(death, lambda_type, inter_Z, beta, kappa, gamma, inter_lambda_s, c_lambda, c_inf, c_dead, n_blocks, np.reshape(u_T,(n_blocks*n_states)), t_grid, T)
    #         convergence_p = np.append(convergence_p, la.norm(after_p - before_p))
    #         convergence_u = np.append(convergence_u, la.norm(after_u - before_u))  
    #         i +=1
    #         if i % n_print == 0:
    #             print("iter: ", i, "p conv: ", convergence_p[-1], "u conv:", convergence_u[-1])      
    return (after_u, after_p, alpha_s, alpha_i, alpha_r, nu, Z, convergence_p, convergence_u)        





def density_plot(t_grid, convergence_u, convergence_p, after_p, alpha_s, alpha_i, Z, nblocks, title, T, type_plot, death, i_control):
    fig, ax = plt.subplots(math.ceil(nblocks/2),2, figsize=(10,4*(math.ceil(nblocks/2) +1)))
    # Density Plot
    i=0
    j=0
    for i in np.arange(math.ceil(nblocks/2)):
        for j in np.arange(2):
            if ((j==1) and (math.ceil(nblocks/2)!=nblocks/2) and (i==math.ceil(nblocks/2)-1)):
                break
            ax[i,j].plot(t_grid, after_p[2*i+j,:], label = "S", c='teal', linestyle="-")
            ax[i,j].plot(t_grid, after_p[nblocks + 2*i +j,:], label = "I", c='magenta', linestyle="-")
            ax[i,j].plot(t_grid, after_p[2*nblocks + 2*i+j,:], label = "R", c='olive', linestyle="-")
            if death == 1:
                ax[i,j].plot(t_grid, after_p[3*nblocks + 2*i+j,:], label = "D", c='darkorange', linestyle="-")
            ax[i,j].legend(fontsize="8")
            ax[i,j].set_title("Density: Block {}".format(2*i+j+1))
    plt.show()
    plt.savefig('%s.eps' % title, bbox_inches='tight', format='eps')


def control_plot(t_grid, convergence_u, convergence_p, after_p, alpha_s, alpha_i, Z, nblocks, title, T, type_plot, death, i_control):
    fig, ax = plt.subplots(figsize=(8, 4))
    if type_plot==0:
        color = iter(cm.rainbow(np.linspace(0, 1, nblocks)))
        for k in np.arange(nblocks):
            c = next(color)
            if i_control==1:
                ax.plot(t_grid, alpha_s[k, :], label="S, block {}".format(k+1), c=c, linestyle="-")
                ax.plot(t_grid, alpha_i[k, :], label="I, block {}".format(k+1), c=c, linestyle="-.")
            else:
                ax.plot(t_grid, alpha_s[k, :], label="S, block {}".format(k+1), c=c, linestyle="-")
            ax.legend(fontsize="8")
            ax.set_title("Controls")
        ax.set_xlim(0, T)
    if type_plot==1:
        ax.set_yscale('log')
        ax.plot(np.arange(np.size(convergence_u)), convergence_u, label="u conv.", c='magenta') # Convergence: when p^(k+1) is close to p^k
        ax.plot(np.arange(np.size(convergence_p)), convergence_p, label="p conv.", c='olive')
        ax.legend(fontsize="8")
        ax.set_title("Convergence")
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.show()