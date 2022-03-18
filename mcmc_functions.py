# Dependencies

from scipy.stats import norm
import numpy as np

# Functions

def calc_prior(x, stdev):
    return norm.logpdf(x, loc=0, scale=stdev)

def new_cand(prev, stdev_prop):
    return np.random.normal(loc=prev, scale=stdev_prop)

def calc_posteriors(params, x, y, stdevs):
    # read the proposed new values
    prop_a = params[0]
    prop_b = params[1]
    prop_e = params[2]

    #calculate predictions and likelihood
    pred = prop_a * x + prop_b
    single_likelihoods = norm.logpdf(y, loc=pred, scale = prop_e)
    likelihood = np.sum(single_likelihoods)

    # calculate the posteriors
    posterior_a = calc_prior(prop_a, stdevs[0]) + likelihood
    posterior_b = calc_prior(prop_b, stdevs[1]) + likelihood
    posterior_e = calc_prior(prop_e, stdevs[2]) + likelihood

    return likelihood, posterior_a, posterior_b, posterior_e

def accept_new(posterior_prev, posterior_new, prev, new, stdev_prop):
    r = np.exp(posterior_new + norm.logpdf(prev, loc=new, scale=stdev_prop) - posterior_prev - norm.logpdf(new, loc=prev, scale=stdev_prop))
    # print(f'r: ' + str(r))
    if np.random.random() < r:
        return True
    else:
        return False

# - generate samples of a,b conditional on the data with your self-written MCMC

def metropolis_hastings(list_initial_samples, iter, data, stdevs, stdev_prop):
    num_accepted_a = 0
    num_accepted_b = 0
    num_accepted_e = 0
    
    samples = np.zeros((iter, 7))
    samples[0] = np.array(list_initial_samples)
    
    for i in range(1,iter):
        # new candidates
        new_a = new_cand(samples[i-1,0], stdev_prop)
        new_b = new_cand(samples[i-1,1], stdev_prop)
        new_e = new_cand(samples[i-1,2], stdev_prop)
        # print(f'New a: {new_a}')
        # print(f'New b: {new_b}')

        # calculate posterior of a and b (and e)
        likelihood, posterior_new_a, posterior_new_b, posterior_new_e = calc_posteriors([new_a, new_b, new_e], data[0], data[1], stdevs)
        # print(f'Posterior a: {posterior_a}')
        # print(f'Posterior b: {posterior_b}')

        # write same as previous
        samples[i] = samples[i-1]
        samples[i,3] = likelihood

        # check new a
        if accept_new(samples[i-1,4], posterior_new_a, samples[i-1,0], new_a, stdev_prop):
            samples[i,0] = new_a
            samples[i,4] = posterior_new_a
            num_accepted_a += 1

        # check new b
        if accept_new(samples[i-1,5], posterior_new_b, samples[i-1,1], new_b, stdev_prop):
            samples[i,1] = new_b
            samples[i,5] = posterior_new_b
            num_accepted_b += 1

        # check new e
        if accept_new(samples[i-1,6], posterior_new_e, samples[i-1,2], new_e, stdev_prop):
            samples[i,2] = new_e
            samples[i,6] = posterior_new_e
            num_accepted_e += 1

    return samples, num_accepted_a, num_accepted_b, num_accepted_e