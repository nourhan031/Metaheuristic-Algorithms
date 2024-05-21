import pandas as pd
import random
import matplotlib.pyplot as plt

def input_data(Path):
    """
    Takes the path of the Excel file of the instances.
    Returns a dict of jobs number as Key and weight, processing time (hours) and due date (hours) as values.
    """
    return pd.read_excel(Path, names=['Job', 'weight', "processing_time", "due_date"],
                         index_col=0).to_dict('index')

print(input_data('Instance_30.xlsx'))

# C_i: completion time
# d_i: due date
# T_i = Tardiness of the job (Delay time)

# T_i = max{C_i - d_i, 0}
# if C_i < d_i: T = 0 (no delay)

# The objective is to order N jobs in a way that MIN the tot weighted delay
# i.e. MIN sum(W_i * T_i) -> the higher the job's weight, the higher its delay

def objective_fun(instance_dict, sol, show = False):
    '''Takes a set of scheduled jobs, dict (input data)
    Return the objective function value of the solution
    '''
    dict = instance_dict
    start_time = 0
    obj_fun = 0

    for i in sol:
        C_i = start_time + dict[i]["processing_time"]  # Completion time
        T_i = max(C_i - dict[i]["due_date"], 0)  # tardiness is the max of completion time of i minus due date of i and 0 -> if completion<due therefore tardiness=0(no delay)
        obj_fun += dict[i]["weight"] * T_i  # obj func is the sum of the product of the weight and the tardiness for each job
        start_time = C_i  # start time for the next job is to be the completion time of the current job


# def visualize_solution(instance_dict, sol):
#     '''Visualize the start and completion times of each job in the solution'''
#     start_time = 0
#     for i in sol:
#         plt.barh(i, dict[i]["processing_time"], left=start_time, color='blue')
#         start_time += dict[i]["processing_time"]
#
#     plt.xlabel('Time')
#     plt.ylabel('Jobs')
#     plt.title('Job Schedule')
#     plt.show()
#
# visualize_solution(instance_dict=input_data('Instance_30.xlsx'), )
