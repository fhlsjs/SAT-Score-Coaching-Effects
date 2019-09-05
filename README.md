# SAT-Score-Coaching-Effects
This is a comprehensive data science project that implements Bayesian hierarchical modeling with MCMC.

The whole repository is outlined as follows:
1. The Problem 2 in pdf file is the project description. It is a well known dataset from Bayesian Inference, Rubin (1981). There are 8 schools with their estimated treatment effects as well as the standard errors. We assume a Normal and Inverse Gamma conjugate prior for the mean and variance of treatment effect terms respectively, with hyperparameters to be changed and impacts to be investigated.

2. The docx file is the full report of this project. Parts (a) through (c) are mathematical derivations of posterior full probability density and posterior conditional probability density of mean and variance (given the other) respectively. We derive the posterior conditional distribution to implement a Gibbs sampler. For each part, complete derivations, results, conclusions and careful justifications are provided with necessary code.

3. There are full implementation in R and python scripts attached. Technically, we generate posterior samples from MCMC, and make inference based on the second half of iterations (e.g. last 5000 out of 10000) when the Markov chain tends to converge. In summary, our goals of this study include estimating the posterior probability of: 1) each school being the best; 2) one school being better than the other for all pairwise comparisons; and 3) the impact of changing hyperparameters. In the last part of this project, we compare the classical ANOVA approach and the Bayesian approach, and reach the following conclusions: 1) Bayesian inference can incorporate knowledge of the model parameters without data collection via the choice of prior distribution (and hyperparameters), while conclusions of classical approach are fully driven by data; 2) Impact of the choice of prior distribution can be very important in Bayesian inference. We need to be careful about the practical meaning of a certain prior, and make sure that the prior does reflect our knowledge closely.
