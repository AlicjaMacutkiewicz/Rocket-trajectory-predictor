import numpy as np
from RK4Sim import rk4_t

thrust_data = np.loadtxt(
    "../../../source_model/R7_SIMLE/R7_OUTPUT/thrust_source.csv", delimiter=","
)
thrust_dict = dict(zip(thrust_data[:, 0], thrust_data[:, 1], strict=False))

R_EARTH = 6371000.0
initial_position = np.array([0.0, 0.0, R_EARTH])
launch_direction = np.array([0.0, 0.0, 1.0])
total_flight_time = 64
rocket_dry_mass = 52.806
fuel_mass = 12.0
isp = 321.258976922505

trajectory = rk4_t(
    initial_position,
    rocket_dry_mass,
    fuel_mass,
    launch_direction,
    total_flight_time,
    thrust_dict,
    isp,
)

print(f"{'time (s)'} {'altitude (m)'} {'velocity (m/s)'}")

for t, pos, vel in trajectory[::50]:
    altitude = np.linalg.norm(pos) - R_EARTH
    velocity_mag = np.linalg.norm(vel)

    print(f"{t:<10.2f} {altitude:<15.2f} {velocity_mag:<15.2f}")

final_t, final_pos, final_vel = trajectory[-1]
final_alt = np.linalg.norm(final_pos) - R_EARTH
