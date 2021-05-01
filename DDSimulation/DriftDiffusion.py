import numpy as np
np.random.seed(134)

class PureDriftDiffusion(object):
    # default params follow Bogacz 2006
    def __init__(
        self,
        time,# given change in time
        drift_rate, # Drift rate, aka "A" A > 0 if threshodl, A < 0 if caution, wo/ loss
        start, # Starting point
        diffusion_rate, # Diffusion rate
        thresholds, # Thresholds
        steps, # number of iterations for drift diffusion
        trials, # number of trials to run drift diffusion
        bias, # A value +/- that lowers a particular threshold, , not implemented
        nondecision_component, # Non decision component, not implemented
        seeds, # not implemented
        **kwargs,
    ):
        #super().__init(**kwargs)
        self.dt = time
        self.mu = drift_rate
        self.x_0 = start
        self.sigma = diffusion_rate
        self.z = thresholds #was thresholds
        self.steps = steps
        self.trials= trials
        self.bias = bias
        self.T_er = nondecision_component
        self.seeds = seeds

    def call(self):
        results, decision_indices = [], []
        for trial in range(self.steps):
            dx = 0
            drift = []
            while True:
                drift.append(dx) #shift from start
                dx += (self.mu*self.dt)+np.asscalar(np.random.normal(loc=0, scale=np.sqrt((self.sigma**2)*self.dt), size=1))
                if (dx < self.z[0]) or (dx > self.z[1]):
                    decision_indices.append(len(drift))
                    break
                # drift.append(dx) #shift from start was here for some reason, but got here so moved above
            results.append(drift)
        longest_drift = max(decision_indices)
        # results = [elt for elt in results if elt != []]
        results = [(drift+[self.z[1] for val in range(longest_drift+1-len(drift))]) if (drift[-1] > 0) else (drift+[self.z[0] for val in range(longest_drift+1-len(drift))]) for drift in results]
        # for above check if it's longest_drift+1-len(drift) or longest_drift-len(drift), for both or (1st case) only one
        return results, decision_indices

    # which z value?
    def decision_time(self): # DT
        exp1 = (self.z[1]*self.mu)/(self.sigma**2)
        exp2 = ((2*self.z[1])/self.mu)
        exp3 = (2*self.x_0*self.mu)/(self.sigma**2)
        exp4 = (self.x_0/self.mu)
        part1 = (self.z[1]/self.mu)*np.tanh(exp1)
        part2 = exp2*((1-np.exp(exp3))/(np.exp(2*exp1)-np.exp(-2*exp1)))
        out = part1 + part2 - exp4
        return out


    def error_rate(self): # ER
        exp1 = (2*self.z[1]*self.mu)/(self.sigma**2)
        exp2 = (2*self.z[1]*self.x_0)/(self.sigma**2)
        part1 = 1 / (1 + np.exp(exp1))
        part2 = ((1-np.exp(exp2))/(np.exp(exp1)-np.exp(-exp1)))
        out = part1 - part2
        return out

    def reaction_time(self):
        exp1 = (self.z[1]*self.mu)/(self.sigma**2)
        exp2 = ((2*self.z[1])/self.mu)
        exp3 = (2*self.x_0*self.mu)/(self.sigma**2)
        exp4 = (self.x_0/self.mu)
        part1 = (self.z[1]/self.mu)*np.tanh(exp1)
        part2 = exp2*((1-np.exp(exp3))/(np.exp(2*exp1)-np.exp(-2*exp1)))
        out = part1 + part2 - exp4 + self.T_er
        return out


# MISTAKES WHERE SOME USEFUL SYNTAX IS PRESENT
# def call(self):
#     '''Runs drift diffusion over a certain number of trials.
#     '''
#     results = []
#     for trial in range(self.trials):
#         all_dx = [(self.mu*self.dt)+np.random.normal(loc=0, scale=np.sqrt((self.sigma**2)*self.dt), size=1) for step in range(250)]
#         all_drift = np.cumsum(a=np.asfarray(all_dx))
#         drift = all_drift[:np.argmax(all_drift >= self.z[1])]
#         drift = np.append(arr=drift[:np.argmax(drift <= self.z[0])], values=[[self.z[0]]]) if (np.min(drift) < -1) else np.append(arr=drift, values=[[self.z[1]]])
#         results.append(drift)
#     return results

#INSIDE call() function
# def stop_at_threshold(drift, thresholds):
#     clean = []
#     for x in drift:
#         if x <= thresholds[0] or x >= thresholds[1]:
#             break
#         else:
#             clean.append(x)
#     print(clean)
#     #thresholds[0] if (clean[-1] < 0) else thresholds[1]
#     clean += [0 for x in range(drift.size-len(clean))]
#     return np.asarray(clean)

#INSIDE call() for loop
#drift = np.asarray([x for x in drift if (self.z[0] <= x <=self.z[1])])
#print(drift)
#else (self.z[0] if (x < 0) else (self.z[1]))
#drift = stop_at_threshold(drift, self.z)
#drift = drift[drift <= self.z[1]]
#drift = drift_all[drift_all >= self.z[0]]
#drift = np.fromiter((x for x in drift_all if (x <= self.z[1]) and (x >= self.z[0])), dtype=drift_all.dtype)
#drift = np.append(arr=drift, values=[[self.z[1]]])
#drift = np.append(arr=drift, values=[np.full(shape=all_drift.size-len(drift), fill_value=1)])

# UNDERNEATH 'return results'
#results.append((np.linspace(start=0, stop=3.5, num=len(y_vals)), drift))
# time_steps = np.random.normal(loc=0, scale=sigma.self, size=steps.self)
# dx = [self.drift_rate*dt*time_steps[dt] for dt in range(steps.self)] #changes in accumulated x
# y_vals = [_ + dx[_] for _ in range(steps.self)]
# return (steps.self, y_vals)
