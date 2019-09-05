## This is a comprehensive case study that implements MCMC methods
## (Gibbs Sampler, M-H algorithm) in a Bayesian hierarchical modeling context.
## Sample code are given for Problem 2 part (d) and (e)
## Mathematical derivations of full posterior distribution and conditional posterior 
## distributions are provided in part (a) - (c) in the documentation
########################################################
##                Part (d) Python code:               ##
########################################################

import random as rd
import numpy as np
import matplotlib.pyplot as plt

## Estimated treatment effect and standard error
sigma = [14.9, 10.2, 16.3, 11.0, 9.4, 11.4, 10.4, 17.6]
y = [28.39, 7.94, -2.75, 6.82, -0.64, 0.63, 18.01, 12.16]
sigma = np.array(sigma)
y = np.array(y)
mu0 = rd.gauss(0, 100.0)      # Initial value of mu
tau_squ0 = 1/rd.gammavariate(0.01, 100.0)     # Initial value of tau_squared

#Pay attention that the parameter definitions of gamma density are different in R and Python
#So (0.01, 0.01) is transformed into (0.01, 100)
theta0 = [0.00]*8
theta0 = np.array(theta0)
for i in range(0,8):
    theta0[i] = rd.gauss((sigma[i]**2*mu0 + y[i]*tau_squ0)/(sigma[i]**2 + tau_squ0),
    np.sqrt((sigma[i]**2*tau_squ0)/(sigma[i]**2 + tau_squ0)))
# Initial values of theta_j's
# Define 2 vectors and 1 matrix to store posterior samples of mu, tau_squared and theta_j's
post_mu = [0.00]*2000
post_tau_squ = [0.00]*2000
post_mu = np.array(post_mu)
post_tau_squ = np.array(post_tau_squ)
post_theta = np.matrix([[0.00]*8]*2000)

# First iteration of Gibbs sampler
post_mu[0] = rd.gauss(10000*sum(theta0)/(80000 + tau_squ0), np.sqrt(10000*tau_squ0/(80000 + tau_squ0)))
post_tau_squ[0] = 1/rd.gammavariate(4.01, 1/(0.01 + sum((theta0 - post_mu[0])**2)/2))
for j in range(0,8):
     post_theta[0,j] = rd.gauss((sigma[j]**2*post_mu[0] + y[j]*post_tau_squ[0])/(sigma[j]**2 + post_tau_squ[0]),
     np.sqrt((sigma[j]**2*post_tau_squ[0])/(sigma[j]**2 + post_tau_squ[0])))
# Finish the rest of Gibbs sampler, 2000 draws in total
for i in range(1,2000):
    post_mu[i] = rd.gauss(10000*np.sum(post_theta[i-1,:])/(80000 + post_tau_squ[i-1]),
    np.sqrt(10000*post_tau_squ[i-1]/(80000 + post_tau_squ[i-1])))
    post_tau_squ[i] = 1/rd.gammavariate(4.01, 1/(0.01 + np.sum(np.array((post_theta[i-1,:]-post_mu[i]))**2)/2))
    for j in range(0,8):
           post_theta[i,j] = rd.gauss((sigma[j]**2*post_mu[i] + y[j]*post_tau_squ[i])/(sigma[j]**2 + post_tau_squ[i]),
           np.sqrt((sigma[j]**2*post_tau_squ[i])/(sigma[j]**2 + post_tau_squ[i])))

a = np.array([0.00]*8)    
# Define a vector to store the estimated probs of each theta_j being the best
post_est_best = np.matrix([[0.00]*8]*1000)
# Define a matrix to store indicators in each draw on which theta_j being the best
# Use the second half of draws from Gibbs sampler for more accuracy
for k in range(1000, 2000):
    b = list(np.array(post_theta[k,:]).reshape(-1,))
    post_est_best[k-1000, b.index(max(b))] = 1
for j in range(0,8):
     a[j] = (sum(post_est_best[:,j]))/1000
print(a)

# Define a matrix to store estimated probs of pairwise comparison between schools
post_est_comp = np.matrix([[0.00]*8]*8)
for k in range(1000,2000):
    for m in range(0,8):
        for n in range(m,8):
            if (post_theta[k,m] > post_theta[k,n]):
                post_est_comp[m,n] = post_est_comp[m,n] + 1
post_est_comp = post_est_comp/1000
print(post_est_comp)

########################################################
##                Part (e) Python code:               ##
########################################################
mu0 = rd.gauss(0,100)  #normal prior
tau_squ0 = 1/rd.gammavariate(1,1)  #Inverse Gamma prior
theta0 = [0.00]*8
theta0 = np.array(theta0)
for i in range(0,8):
    theta0[i] = rd.gauss((sigma[i]**2*mu0 + y[i]*tau_squ0)/(sigma[i]**2 + tau_squ0),
    np.sqrt((sigma[i]**2*tau_squ0)/(sigma[i]**2 + tau_squ0)))
post_mu = [0.00]*2000
post_tau_squ = [0.00]*2000
post_theta = np.matrix([[0.00]*8]*2000)

post_mu[0] = rd.gauss(10000*sum(theta0)/(80000 + tau_squ0), np.sqrt(10000*tau_squ0/(80000 + tau_squ0)))
post_tau_squ[0] = 1/rd.gammavariate(5, 1/(1 + sum((np.array(theta0 - post_mu[0])**2))/2))
for j in range(0,8):
    post_theta[0,j] = rd.gauss((sigma[j]**2*post_mu[0] + y[j]*post_tau_squ[0])/(sigma[j]**2 + post_tau_squ[0]),
    np.sqrt((sigma[j]**2*post_tau_squ[0])/(sigma[j]**2 + post_tau_squ[0])))

for i in range(1,2000):
    post_mu[i] = rd.gauss(10000*np.sum(post_theta[i-1,:])/(80000 + post_tau_squ[i-1]),
    np.sqrt(10000*post_tau_squ[i-1]/(80000 + post_tau_squ[i-1])))
    post_tau_squ[i] = 1/rd.gammavariate(5, 1/(1 + np.sum(np.array(post_theta[i-1,:] - post_mu[i])**2)/2))
    for j in range(0,8):
        post_theta[i,j] = rd.gauss((sigma[j]**2*post_mu[i] + y[j]*post_tau_squ[i])/(sigma[j]**2 + post_tau_squ[i]),
        np.sqrt((sigma[j]**2*post_tau_squ[i])/(sigma[j]**2 + post_tau_squ[i])))

a = [0.00]*8
post_est_best = np.matrix([[0.00]*8]*2000)
for k in range(1000,2000):
    bb = list(np.array(post_theta[k,:]).reshape(-1,))
    post_est_best[k-1000,bb.index(max(bb))]=1
for j in range(0,8):
    a[j] = (sum(post_est_best[:,j]))/1000
print(a[2]) ## The 3rd school being the best

## Plot the simulated Markov Chain
n, bins, patches = plt.hist(post_mu[1000:], 15, normed = 0, facecolor = "g", alpha = 0.75)
plt.xlabel('posterior mean'); plt.ylabel('Frequency')
plt.title('Histogram of the Markov Chain')
plt.axis([-2, 20, 0, 160])
plt.show()
# Everything for the code is the same as previous except change for conditional posterior of tau^2

########################################################
## Part (f) is similar to Part (d) and (e)   ##
## Just need to change hyper-parameters      ##
########################################################