import numpy as np

# liczy kolejna pozycje po czasie dt
def euler_next(dt, mass, position, angle, velocity, thrust):
    gravity = np.array([0.0, 0.0, -9.81])

    new_acceleration = (thrust * angle) / max(mass, 1e-8) + gravity

    new_velocity = velocity + new_acceleration * dt
    new_position = position + new_velocity * dt

    return new_position, new_velocity, new_acceleration

# liczy pozycja(t)
def euler_t(start_position, start_mass, angle, time, thrust):
    t = 0
    dt = 0.02
    position = start_position
    velocity = np.array([0, 0, 0])
    acceleration = np.array([0, 0, 0])

    times = np.array(list(thrust.keys()))
    values = np.array(list(thrust.values()))

    while t < time:
        current_thrust = 0
        if t < times[-1]:
            current_thrust = np.interp(t, times, values)

        position, velocity, acceleration = euler_next(dt, start_mass, position, angle, velocity, current_thrust)
        t += dt
    return position, velocity, acceleration