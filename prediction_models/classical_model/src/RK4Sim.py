import numpy as np

G = 6.67430e-11
G0 = 9.80665


# liczy kolejna pozycje po czasie dt
def rk4_next(dt, mass, position, angle, velocity, thrust):
    def acceleration(pos):
        earth_mass = 5.972e24
        gravity = -G * earth_mass * pos / max(np.linalg.norm(pos) ** 3, 1e-8)
        return (thrust * angle) / max(mass, 1e-8) + gravity

    k1_a = acceleration(position)
    k1_v = k1_a * dt
    k1_x = velocity * dt

    k2_a = acceleration(position + 0.5 * k1_x)
    k2_v = k2_a * dt
    k2_x = (velocity + 0.5 * k1_v) * dt

    k3_a = acceleration(position + 0.5 * k2_x)
    k3_v = k3_a * dt
    k3_x = (velocity + 0.5 * k2_v) * dt

    k4_a = acceleration(position + k3_x)
    k4_v = k4_a * dt
    k4_x = (velocity + k3_v) * dt

    new_acceleration = (k1_a + 2 * k2_a + 2 * k3_a + k4_a) / 6
    new_velocity = velocity + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
    new_position = position + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6

    return new_position, new_velocity, new_acceleration


# liczy pozycja(t)
def rk4_t(start_position, rocket_mass, fuel_mass, angle, time, thrust, isp):
    t = 0
    dt = 0.02
    position = start_position
    velocity = np.array([0, 0, 0])
    acceleration = np.array([0, 0, 0])

    trajectory = []

    times = np.array(list(thrust.keys()))
    values = np.array(list(thrust.values()))

    while t < time:
        current_thrust = 0
        if t < times[-1]:
            current_thrust = np.interp(t, times, values)

            mdot = current_thrust / (isp * G0)
            fuel_mass -= mdot * dt
            fuel_mass = max(fuel_mass, 0.0)

        position, velocity, acceleration = rk4_next(
            dt, fuel_mass + rocket_mass, position, angle, velocity, current_thrust
        )
        trajectory.append((t, np.copy(position), np.copy(velocity), np.copy(acceleration)))
        t += dt

    return trajectory


def rk4_accelerations_at_times(
    target_times,
    start_position,
    rocket_mass,
    fuel_mass,
    angle,
    thrust,
    isp,
):
    target_times = np.asarray(target_times, dtype=np.float32)
    original_shape = target_times.shape
    flat_times = target_times.reshape(-1)

    if flat_times.size == 0:
        return np.empty((*original_shape, 3), dtype=np.float32)

    trajectory = rk4_t(
        start_position=start_position,
        rocket_mass=rocket_mass,
        fuel_mass=fuel_mass,
        angle=angle,
        time=float(np.max(flat_times)) + 0.02,
        thrust=thrust,
        isp=isp,
    )

    trajectory_times = np.array([row[0] for row in trajectory], dtype=np.float32)
    trajectory_accelerations = np.array([row[3] for row in trajectory], dtype=np.float32)

    accelerations = np.stack(
        [
            np.interp(flat_times, trajectory_times, trajectory_accelerations[:, axis])
            for axis in range(3)
        ],
        axis=-1,
    )

    return accelerations.reshape((*original_shape, 3)).astype(np.float32)
