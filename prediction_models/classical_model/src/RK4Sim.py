import numpy as np

def rk4_next(dt, mass, position, angle, velocity, thrust):
    gravity = np.array([0.0, 0.0, -9.81])

    def acceleration(pos, vel):
        return (thrust * angle) / max(mass, 1e-8) + gravity

    k1_v = acceleration(position, velocity) * dt
    k1_x = velocity * dt

    k2_v = acceleration(position + 0.5 * k1_x, velocity + 0.5 * k1_v) * dt
    k2_x = (velocity + 0.5 * k1_v) * dt

    k3_v = acceleration(position + 0.5 * k2_x, velocity + 0.5 * k2_v) * dt
    k3_x = (velocity + 0.5 * k2_v) * dt

    k4_v = acceleration(position + k3_x, velocity + k3_v) * dt
    k4_x = (velocity + k3_v) * dt

    new_velocity = velocity + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    new_position = position + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
    new_acceleration = acceleration(new_position, new_velocity)

    return new_position, new_velocity, new_acceleration

def rk4_t(start_position, start_mass, angle, time, thrust):
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

        position, velocity, acceleration = rk4_next(dt, start_mass, position, angle, velocity, current_thrust)

        t += dt

    return position, velocity, acceleration