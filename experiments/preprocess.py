import pandas as pd

if __name__ == '__main__':
    dyn_wait = pd.read_csv('experiments/fixed/dyn_wait.csv')
    dyn_brake = pd.read_csv('experiments/fixed/dyn_brake.csv')
    dyn_speed = pd.read_csv('experiments/fixed/dyn_speed.csv')
    dyn_queue = pd.read_csv('experiments/fixed/dyn_queue.csv')
    dyn_emission = pd.read_csv('experiments/fixed/dyn_emission.csv')
    dyn_pressure = pd.read_csv('experiments/fixed/dyn_pressure.csv')

    stat_wait = pd.read_csv('experiments/fixed/stat_wait.csv')
    stat_brake = pd.read_csv('experiments/fixed/stat_brake.csv')
    stat_speed = pd.read_csv('experiments/fixed/stat_speed.csv')
    stat_queue = pd.read_csv('experiments/fixed/stat_queue.csv')
    stat_emission = pd.read_csv('experiments/fixed/stat_emission.csv')
    stat_pressure = pd.read_csv('experiments/fixed/stat_pressure.csv')

    dyn_rw = dyn_wait['reward']
    dyn_rb = dyn_brake['reward']
    dyn_rs = dyn_speed['reward']
    dyn_rq = dyn_queue['reward']
    dyn_re = dyn_emission['reward']
    dyn_rp = dyn_pressure['reward']

    stat_rw = stat_wait['reward']
    stat_rb = stat_brake['reward']
    stat_rs = stat_speed['reward']
    stat_rq = stat_queue['reward']
    stat_re = stat_emission['reward']
    stat_rp = stat_pressure['reward']

    print("dynamic scenarios")
    for r, label in zip([dyn_rw, dyn_rb, dyn_rs, dyn_rq, dyn_re, dyn_rp], ["wait", "brake", "speed", "queue", "emission", "pressure"]):
        print(f"{label} range: {r.max() - r.min()}")

    print("\nstatic scenarios")
    for r, label in zip([stat_rw, stat_rb, stat_rs, stat_rq, stat_re, stat_rp], ["wait", "brake", "speed", "queue", "emission", "pressure"]):
        print(f"{label} range: {r.max() - r.min()}")
