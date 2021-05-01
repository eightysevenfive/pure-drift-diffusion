#import utils
import argparse
from random import randrange
import numpy as np
from DriftDiffusion import PureDriftDiffusion
import matplotlib.pyplot as plt
import os


def main():


    title = "Pure and Extended Drift Diffusion"
    parser = argparse.ArgumentParser(description=title,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Equation Options
    equa_opts = parser.add_argument_group('Equation Options')
    equa_opts.add_argument("-t", "--time", type=float, required=False, default=0.01, help="The value of dt in the drift diffusion equation.")
    equa_opts.add_argument("-dr", "--drift_rate", type=float, default=1.0, help="The value of the drift rate in the drift diffusion equation.")
    equa_opts.add_argument("-s", "--start", type=float, default=0.0, help="The value of the start time in the drift diffusion equation.")
    equa_opts.add_argument("-dfr", "--diff_rate", type=float, default=1.0, help="The value of sigma (aka c) in the drift diffusion equation.")
    equa_opts.add_argument('-thres', '--threshold', type=float, default=1.0, help='The absolute value of the threshold for drift diffusion')
    equa_opts.add_argument('-bi', '--bias', type=float, default=0.0)
    equa_opts.add_argument('-ndc', '--nondecomp', type=float, default=0.0)

    # Fitting Options
    fit_opts = parser.add_argument_group('Fitting Options')
    fit_opts.add_argument('-sd', '--seeds', type=int, default=134, help='Enter a seed value for the experiment')
    fit_opts.add_argument('-st', '--steps', type=int, default=10000, help='The number of steps that drift diffusion simulates')
    fit_opts.add_argument('-tri', '--trials', type=int, default=4, help='The number of trials to plot')

    # Plotting Options
    plot_opts = parser.add_argument_group('Plotting Options')
    args = parser.parse_args()

    # interval = (0, 2.5) # perhaps, (0, steps*time / 60)

    interval = args.steps
    model = PureDriftDiffusion(
        args.time,
        args.drift_rate,
        args.start,
        args.diff_rate,
        (-args.threshold,args.threshold),
        args.steps,
        args.trials,
        args.bias,
        args.nondecomp,
        args.seeds,
    )
    results, decision_indices = model.call()
    print(f"DECISION TIME: {model.decision_time()}\nERROR RATE: {model.error_rate()}\nREACTION TIME: {model.reaction_time()}")

    # highest = 0
    # for result in results:
    #     x = list(filter(lambda x: x != 1.0, result))
    #     if x[-1] >= highest:
    #         highest = x[-1]
    # time_interval = highest+(highest/5)
    time_interval = ((max(decision_indices) * args.time))


    # PLOT SOME RANDOM DRIFTS
    fig, ax1 = plt.subplots()
    x1, x2, y1, y2 = ax1.axis()
    ax1.axis((0, time_interval+0.1+args.start, -args.threshold-1, args.threshold+1)) #change to  -args.threshold-1, args.threshold+1 if not using subplots
    selected_results = [results[randrange(len(results))] for trial in range(args.trials)]
    for vals in selected_results:
        fill_val = args.threshold if (vals[-1] > 0) else -args.threshold
        ax1.plot(
            np.linspace(start=args.start, stop=time_interval+args.start, num=len(vals)),
            vals,
            linewidth=0.70,
        )
    tmfont = {'fontname': "Times New Roman"}
    ax1.axhline(y=args.threshold, color='black', linewidth=0.85)
    ax1.axhline(y=0, color='black', alpha=0.5, linewidth=0.75, linestyle='--')
    ax1.axhline(y=-args.threshold, color='black', linewidth=0.85)
    ax1.axvline(x=args.start, color='black', alpha=0.5, linewidth=0.75, linestyle='--' )
    plt.text(args.start/2, 0.01, 'x_0', color='black', fontsize=8, **tmfont)
    plt.xlabel('Time [s]', fontsize=14, **tmfont)
    plt.ylabel('Accumulation', fontsize=14, **tmfont)
    plt.title('Pure Drift Diffusion', fontsize=16, **tmfont)

    # PLOT DISTRIBUTION CURVES
    bins = time_interval*10
    #ax2 = plt.subplot(212)

    #ax2 = ax1.twiny() #fig.add_subplot()
    #ax2.axis((args.start, time_interval+0.1, args.threshold, args.threshold+1))
    # bins = max(decision_indices)
    # ax2.hist(bins=)


    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# # Execution Options
# exec_opts = parser.add_argument_group('Execution Options')
# # exec_opts.add_argument("-choice", "--exec_choice", required=True, choices=('Manual', 'Automatically'),
# #     help="Choice to manually input model values or automatic extract them from a given dataset.")
# args = parser.parse_args()
# if args.exec_choice == 'Automatically':
#     exec_opts.add_argument("-data", "--datapath", type=str, required=True, help="Enter the path to the dataset")
#     exec_opts.add_argument('-model', '--modeltype', type=str, required=True, choices=('DD', 'EDD'),
#         help="Choice to use Drift Diffusion (DD) or Extended Drift Diffusion (EDD)")
# else:
#     pass

# PLOT DRIFTS
# max_trial_len = max([len(vals) for vals in results])+(steps//1000)
# for vals in results:
#     fill_val = thresholds[1] if (vals[-1] > 0) else thresholds[0]
#     vals = np.append(
#         arr=vals,
#         values=[np.full(shape=max_trial_len-len(vals), fill_value=fill_val)]
#     )
#     plt.plot(
#         np.linspace(start=interval[0], stop=interval[1]+1, num=max_trial_len),
#         vals,
#         linewidth=0.70,
#     )
    #plt.axvline(x=vals[v_line_index-1], color='black', linestyle='--', linewidth=0.5)


# PLOT DRIFT RATE
# plt.plot(
#     np.linspace(start=interval[0], stop=interval[1]+1, num=results[0].size),
#     np.cumsum(a=np.full(shape=results[0].size, fill_value=drift_rate))
# )


# mu = 1 # Drift rate, aka "A" A > 0 if threshodl, A < 0 if caution
# A = 1
# x_0 = 0 # Start point
# sigma = 1 # Diffusion rate
# dt = 0.01 # given change in time
# #dW = 0.1 # Gaussian noise increment
# z = (-1, 1) # Thresholds
# steps = 100000 # number of iterations for drift diffusion
# interval = (0, 2.5) # time in minutes
# trials = 5 # number of trials to run drift diffusion
# bias = 0.3 # A value +/- that lowers a particular threshold
# T_er = 0 # Non decision component
#
#
# # choice sigma/c = 1, dt=0.01, A/mu = 1, z=1
# np.random.seed(134)
# results = []
# for trial in range(5):
#     dx_s = [(mu*dt)+np.random.normal(loc=0, scale=np.sqrt((sigma**2)*dt), size=1) for step in range(steps)]
#     y_vals = np.cumsum(a=np.asfarray(dx_s))
#     y_vals = [_ for _ in y_vals if _ <= 1.0]
#     y_vals.insert(0, x_0)
#     results.append((np.linspace(start=0, stop=3.5, num=len(y_vals)),y_vals))



#y_vals = [_ for _ in y_vals if _ <= 1.0]
#print(y_vals)
# for iter in range(steps-1):
#     if A > 0:
#         #x = (mu*dt*x + np.random.normal(loc=0, scale=np.sqrt((sigma**2)*dt), size=(1,)))
#         dx = (mu*dt + np.random.normal(loc=0, scale=np.sqrt((sigma**2)*dt), size=1))
#         y_vals.append(dx)
#     else:
#         dx = (-mu*dt + np.random.normal(loc=0, scale=np.sqrt((sigma**2)*dt), size=(1,)))
#         y_vals.append(dx)


# x1, x2, y1, y2 = plt.axis()
# plt.axis((time[0],time[1], z[0], z[1]))
# #x_vals = [(x*time[1])/steps for x in x_vals[1:]]
# #x_vals.insert(0, x_0)
#
# print(y_vals)
# #plt.plot(y_vals, np.linspace(start=0, stop=3.5, num=len(y_vals)))
# for result in results:
#     x, y = result
#     plt.plot(x,y)
#
# # time_steps = np.random.normal(loc=0, scale=sigma, size=steps)
# # dx = [drift_rate*dt*0.001 + sigma*time_steps[dt] for dt in range(steps)] #changes in accumulated x
# # y_vals = [drift_rate*_ + dx[_] for _ in range(steps)]
# # plt.plot(list(range(steps)), y_vals)
#
# #plt.axvline(x1+3, color='black', linewidth=0.4)
# #plt.plot(list(range(50)), steps, color='black', linewidth=0.4)
# plt.show()
