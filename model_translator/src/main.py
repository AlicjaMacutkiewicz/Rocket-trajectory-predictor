import os
import matplotlib.pyplot as plt
import numpy as np
import json
from enum import IntEnum
from rocketpy import Environment, SolidMotor, Rocket, Flight as rk
from rocketpy.simulation import flight






class LogLevel(IntEnum):
    INFO = 1
    WARNING = 2
    ERROR = 3
C_CURRENT_LOG_LEVEL = LogLevel.INFO

def print_error(msg):
    if C_CURRENT_LOG_LEVEL>= LogLevel.ERROR:
       print("ERROR: " + msg + '\n')
def print_info(msg):
    if C_CURRENT_LOG_LEVEL>= LogLevel.INFO:
        print("INFO: " + msg + '\n')
def print_warning(msg):
    if C_CURRENT_LOG_LEVEL>= LogLevel.WARNING:
        print("WARNING: " + msg + '\n')


# @BRIEF
# Initializes rocket with data found in give JSON file
# @ARGUMENTS: 
#     path_to_file -> STRING representing path to JSON file
# @RETURN
#     rocket -> Initialized rocketpy::Rocket class 


def init_from_JSON(path_to_file):
    with open(path_to_file, 'r', encoding='utf-8')as file:
        data= json.load(file)
    print_info(f"Reading from {path_to_file}")

    name = data['id']['rocket_name']

    print_info(f"Loading model: {name}")

    motor_data = data["motors"] 
     
    print_info("loading motor")
    motor = SolidMotor(
        thrust_source=motor_data["thrust_source"],
        dry_mass=motor_data["dry_mass"],
        dry_inertia=motor_data["dry_inertia"],
        nozzle_radius=motor_data["nozzle_radius"],
        grain_number=motor_data["grain_number"],
        grain_density=motor_data["grain_density"],
        grain_outer_radius=motor_data["grain_outer_radius"],
        grain_initial_inner_radius=motor_data["grain_initial_inner_radius"],
        grain_initial_height=motor_data["grain_initial_height"],
        grain_separation=motor_data["grain_separation"],
        grains_center_of_mass_position=motor_data["grains_center_of_mass_position"],
        center_of_dry_mass_position=motor_data["center_of_dry_mass_position"],
        nozzle_position=motor_data["nozzle_position"],
        burn_time=3.0, 
        throat_radius=motor_data["throat_radius"],
        coordinate_system_orientation=motor_data["coordinate_system_orientation"]
            ) 
    rocket_data = data["rocket"]
    print_info("loading rocket")
    rocket = Rocket(
        radius=rocket_data["radius"],
        mass=rocket_data["mass"],
        inertia=tuple(rocket_data["inertia"]),
        power_off_drag=rocket_data["drag_curve"],
        power_on_drag=["drag_curve"],
        center_of_mass_without_motor=rocket_data["center_of_mass_without_propellant"],
        coordinate_system_orientation=rocket_data["coordinate_system_orientation"]
                )
    rocket.add_motor(motor, position=motor_data["position"])
    print_info("loading nose")
    nose_data = data["nosecones"]
    rocket.add_nose(
            length=nose_data["lenght"],
            kind=nose_data["kind"],
            position=nose_data["position"]
            )
    print_info("loading fins")
    rocket.add_trapezoidal_fins(
            n=3,
            root_chord=0.12,
            tip_chord=0.06,
            span=0.10,
            position=1.0, 
            sweep_length=0.05
            )

    parachute_data = data["parachutes"]["0"]
    print_info("loading parachute")
    rocket.add_parachute(
            name=parachute_data["name"],
            cd_s=parachute_data["cds"],
            trigger=parachute_data["deploy_event"],
            lag=parachute_data["deploy_delay"]
            )
    return rocket

