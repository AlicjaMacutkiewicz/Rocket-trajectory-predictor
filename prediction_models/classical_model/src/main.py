import json

import numpy as np
from RK4Sim import rk4_t


def load_simulation_config(parameter_json_path):
    with open(parameter_json_path, encoding="utf-8") as file:
        model_data = json.load(file)

    rocket_data = model_data.get("rocket", {})
    motor_data = model_data.get("motors", {})
    stored_results = model_data.get("stored_results", {})

    rocket_dry_mass = rocket_data.get("mass", 50.876)
    total_flight_time = stored_results.get("flight_time", 64.0)

    fuel_mass = 13.04  # takie jest niby na manifeście, chociaż 10-12 było wcześniej chyba
    isp = motor_data.get("isp", 204.26)  # to wyszło em z staticów
    # ewenatualnie isp = 321,258976922505 to wyszło pawłowi

    return rocket_dry_mass, fuel_mass, isp, total_flight_time


def main():
    with open("paths.json", encoding="utf-8") as file:
        paths = json.load(file)

    thrust_path = paths["thrust_source"]
    parameter_path = paths["parameters"]
    rocket_dry_mass, fuel_mass, isp, total_flight_time = load_simulation_config(parameter_path)

    thrust_data = np.loadtxt(thrust_path, delimiter=",")
    thrust_dict = dict(zip(thrust_data[:, 0], thrust_data[:, 1], strict=False))

    R_EARTH = 6371000.0
    initial_position = np.array([0.0, 0.0, R_EARTH])
    launch_direction = np.array([0.0, 0.0, 1.0])

    trajectory = rk4_t(
        start_position=initial_position,
        rocket_mass=rocket_dry_mass,
        fuel_mass=fuel_mass,
        angle=launch_direction,
        time=total_flight_time,
        thrust=thrust_dict,
        isp=isp,
        sampling_rate=500,
    )

    print(f"{'time (s)':<10} {'altitude (m)':<15} {'velocity (m/s)':<15}")

    for t, pos, vel, _acc in trajectory:
        altitude = np.linalg.norm(pos) - R_EARTH
        velocity_mag = np.linalg.norm(vel)

        print(f"{t:<10.4f} {altitude:<15.4f} {velocity_mag:<15.4f}")


if __name__ == "__main__":
    main()
