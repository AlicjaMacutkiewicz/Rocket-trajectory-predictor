import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rocketpy import Flight , Accelerometer

# @BRIEF
# cretes string based on rp.solution_array, without np.float type signature
# @ARGUMENTS: 
#       data -> rp.solution_array instance
# @RETURN
#       msg -> string representation of given array
def rp_solution_arr_str(data):
    msg = '['
    for itr in data:
        msg += f"{str(itr)},"
    msg += ']'
    return msg


#helper function for 'run_single_simulation'
def get_best_acceleration(real_vals, suffix, all_accels_df, thresholds):
    cond = [np.abs(real_vals) < t for t in thresholds]
    choices = [all_accels_df[f"LSM9DS1_acc_{g}g_{suffix}"] for g in [2, 4, 8]]
    return np.select(cond, choices, default=all_accels_df[f"LSM9DS1_acc_16g_{suffix}"])

def run_single_simulation(i, rocket, environment, flight):
    current_flight = Flight(
            heading=flight.heading,
            environment=environment,
            rocket=rocket,
            rail_length=flight.rail_length
            )
    #for parquet data saving and packing (comment out lines below)
    # |
    file_name = f"output/flight_{i}.out"
    with open(file_name, 'w+') as file:
            for sample in current_flight.solution:
                file.write(rp_solution_arr_str(sample) + '\n')
    # |
    accel_data = []

    for sensor_tuple in rocket.sensors:
        sensor = sensor_tuple.component
        if isinstance(sensor , Accelerometer):
            cols = ["Time", f"{sensor.name}_X", f"{sensor.name}_Y" , f"{sensor.name}_Z"]
            frame = pd.DataFrame(sensor.measured_data , columns=cols)
            frame.set_index("Time", inplace=True)

            accel_data.append({
                "df": frame,
                    "range": sensor.measurement_range,
                    "name": sensor.name
                })
    if accel_data:
        all_accels_df = pd.concat([item["df"] for item in accel_data], axis=1)
        czasy = all_accels_df.index.values
        
        real_x= np.array([current_flight.ax(t) for t in czasy])
        real_y = np.array([current_flight.ay(t) for t in czasy])
        real_z = np.array([current_flight.az(t) for t in czasy])

        thresholds = [19.613, 39.227, 78.453]
        
        all_accels_df["Best_Acc_X"] = get_best_acceleration(real_x, "X", all_accels_df, thresholds)
        all_accels_df["Best_Acc_Y"] = get_best_acceleration(real_y, "Y", all_accels_df, thresholds)
        all_accels_df["Best_Acc_Z"] = get_best_acceleration(real_z, "Z", all_accels_df, thresholds)
        
        final_df = all_accels_df[["Best_Acc_X", "Best_Acc_Y", "Best_Acc_Z"]].copy()
        final_df.to_csv(f"output/flight_{i}_best_sensors.csv", index_label="Time")
        final_df['flight_id'] = i 

        #for parquet data saving and packing 
        # | 
        # return final_df
        # |
