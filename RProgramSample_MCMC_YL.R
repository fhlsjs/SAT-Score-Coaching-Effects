########################################################
##                Part (d)  R code:                   ##
########################################################
set.seed(1234)      # Set random seed 1234
sigma <- c(14.9, 10.2, 16.3, 11.0, 9.4, 11.4, 10.4, 17.6)
y <- c(28.39, 7.94, -2.75, 6.82, -0.64, 0.63, 18.01, 12.16)
mu0 <- rnorm(1, 0, 100)      # Initial value of mu
tau_squared0 <- 1/rgamma(1, 0.01, 0.01)     # Initial value of tau_squared
theta0 <- numeric(8)
for (i in 1:8){
  theta0[i]<-rnorm(1,(sigma[i]^2*mu0+y[i]*tau_squared0)/(sigma[i]^2+tau_squared0),sqrt((sigma[i]^2*tau_squared0)/(sigma[i]^2+tau_squared0)))
}       
# Initial values of theta_j's


post_mu <- numeric(2000)
post_tau_squared <- numeric(2000)
post_theta <- matrix(data = NA, nrow = 2000, ncol = 8)
# Define 2 vectors and 1 matrix to store posterior samples of 
# mu, tau_squared and theta_j's

# First iteration of Gibbs sampler
post_mu[1] <- rnorm(1, 10000*sum(theta0)/(80000+tau_squared0), sqrt(10000*tau_squared0/(80000+tau_squared0)))
post_tau_squared[1] <- 1/rgamma(1, 4.01, 0.01 + sum((theta0-post_mu[1])^2)/2)
for (j in 1:8){
  post_theta[1,j] <- rnorm(1, (sigma[j]^2*post_mu[1]+y[j]*post_tau_squared[1])/(sigma[j]^2+post_tau_squared[1]),sqrt((sigma[j]^2*post_tau_squared[1])/(sigma[j]^2+post_tau_squared[1])))
}  

# Finish the whole process of Gibbs sampler, 2000 draws in total
for (i in 2:2000){
  post_mu[i] <- rnorm(1,10000*sum(post_theta[i-1,])/(80000+post_tau_squared[i-1]),sqrt(10000*post_tau_squared[i-1]/(80000+post_tau_squared[i-1])))
  post_tau_squared[i] <- 1/rgamma(1,4.01,0.01+sum((post_theta[i-1,]-post_mu[i])^2)/2)
  for (j in 1:8){
    post_theta[i,j]<-rnorm(1,(sigma[j]^2*post_mu[i]+y[j]*post_tau_squared[i])/(sigma[j]^2+post_tau_squared[i]),sqrt((sigma[j]^2*post_tau_squared[i])/(sigma[j]^2+post_tau_squared[i])))
  }
} 


a <- numeric(8)        
# Define a vector to store the estimated probabilities 
# of each theta_j being the best
post_est_best <- matrix(data = 0, nrow = 1000, ncol = 8)
# Define a matrix to store indicators in each draw 
# which theta_j being the best
# Use the second half of draws from Gibbs sampler
for (k in 1001:2000){
  for (j in 1:8){
    if (post_theta[k,j]==max(post_theta[k,])) post_est_best[k-1000,j] <- 1
  }
}

for (j in 1:8){
  a[j] <- (sum(post_est_best[,j]))/1000
}
print(a)
post_est_compare <- matrix(data = 0, nrow = 8, ncol = 8)
# Define a matrix to store estimated probabilities of 
# pairwise comparison between schools
for (k in 1001:2000){
  for (m in 1:8){
    for (n in m:8){
      if (post_theta[k,m] > post_theta[k,n]) post_est_compare[m,n] <- post_est_compare[m,n] + 1
    }
  }
}
post_est_compare <- post_est_compare/1000
post_est_compare




########################################################
##                Part (e)  R code:                   ##
########################################################
set.seed(1234)
sigma <- c(14.9, 10.2, 16.3, 11.0, 9.4, 11.4, 10.4, 17.6)
y <- c(28.39, 7.94, -2.75, 6.82, -0.64, 0.63, 18.01, 12.16)
mu0 <- rnorm(1, 0, 100)  #normal prior
tau_squared0 <- 1/rgamma(1, 1, 1)  #Inverse Gamma prior
theta0 <- numeric(8)
for (i in 1:8){
  theta0[i] <- rnorm(1,(sigma[i]^2*mu0+y[i]*tau_squared0)/(sigma[i]^2+tau_squared0),sqrt((sigma[i]^2*tau_squared0)/(sigma[i]^2+tau_squared0)))
}

post_mu <- numeric(2000)
post_tau_squared <- numeric(2000)
post_theta <- matrix(data = NA, nrow = 2000, ncol = 8)

post_mu[1] <- rnorm(1, 10000*sum(theta0)/(80000+tau_squared0), sqrt(10000*tau_squared0/(80000+tau_squared0)))
post_tau_squared[1] <- 1/rgamma(1, 5, 1 + sum((theta0-post_mu[1])^2)/2)
for (j in 1:8){
  post_theta[1,j] <- rnorm(1, (sigma[j]^2*post_mu[1]+y[j]*post_tau_squared[1])/(sigma[j]^2+post_tau_squared[1]), sqrt((sigma[j]^2*post_tau_squared[1])/(sigma[j]^2+post_tau_squared[1])))
}


for (i in 2:2000){
  post_mu[i] <- rnorm(1, 10000*sum(post_theta[i-1,])/(80000+post_tau_squared[i-1]),sqrt(10000*post_tau_squared[i-1]/(80000+post_tau_squared[i-1])))
  post_tau_squared[i] <- 1/rgamma(1, 5, 1 + sum((post_theta[i-1,]-post_mu[i])^2)/2)
  for (j in 1:8){
    post_theta[i,j] <- rnorm(1,(sigma[j]^2*post_mu[i]+y[j]*post_tau_squared[i])/(sigma[j]^2+post_tau_squared[i]),sqrt((sigma[j]^2*post_tau_squared[i])/(sigma[j]^2+post_tau_squared[i])))
  }
}


a <- numeric(8)
post_est_best <- matrix(data = 0, nrow = 1000, ncol = 8)
for (k in 1001:2000){
  for (j in 1:8){
    if (post_theta[k,j]==max(post_theta[k,])) post_est_best[k-1000,j] <- 1
  }
}


for (j in 1:8){
  a[j] <- (sum(post_est_best[,j]))/1000
}
print(a[3])
# Everything for the code is the same as previous 
# except change for conditional posterior of tau^2




########################################################
##                Part (f)  R code:                   ##
########################################################
set.seed(1234)
post.theta <- matrix(data = 0, nrow = 1000, ncol = 8)
for (j in 1:8){
  post.theta[,j] <- rnorm(1000,y[j],sigma[j])
}


a <- numeric(8)
post.est.best <- matrix(data = 0, nrow = 1000, ncol = 8)
for (k in 1:1000){
  for (j in 1:8){
    if (post.theta[k,j]==max(post.theta[k,])) post.est.best[k,j] <- 1
  }
}


for (j in 1:8){
  a[j] <- (sum(post.est.best[,j]))/1000
}
print(a)

post.est.compare <- matrix(data = 0, nrow = 8, ncol = 8)
for (k in 1:1000){
  for (m in 1:8){
    for (n in m:8){
      if (post.theta[k,m]>post.theta[k,n]) post.est.compare[m,n] <- post.est.compare[m,n]+1
    }
  }
}
post.est.compare <- post.est.compare/1000
print(post.est.compare)

