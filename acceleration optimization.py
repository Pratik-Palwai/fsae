import numpy
import matplotlib.pyplot as plt
from time import time

START_TIME = time()

figure = plt.figure()
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.subplots_adjust(left=0.053, bottom=0.069, right=0.985, top=0.9, wspace=0.18, hspace=0.383)

nrows = 2
ncols = 3

output_plot = figure.add_subplot(nrows, ncols, 1)
force_plot = figure.add_subplot(nrows, ncols, 4)

velocity_plot = figure.add_subplot(nrows, ncols, 2)
optimal_ratio_velocity_plot = figure.add_subplot(nrows, ncols, 5)

position_plot = figure.add_subplot(nrows, ncols, 3)
optimal_ratio_position_plot = figure.add_subplot(nrows, ncols, 6)

M_CAR = 500 # mass of car and driver [kg]
REAR_DIST = 0.75 # proportion of car's weight on rear axle
ETA = 0.75 # overall powertrain efficiency
K_STATIC = 1.25 # static friction coefficient for tyres
R_WHEEL = 0.2286 # radius of tyre [m]
A_FRONT = 0.124461 # frontal area [m^2]
A_WING = 0.857528 # airfoil area [m^2]
RHO = 1.225 # density of air at sea level [kg / m^3]
C_D = 1.01 # drag coefficient
C_L = 1.75 # lift (downforce) coefficient
C_R = 0.02 # rolling resistance coefficient
N_BASE_VELOCITY = 4.41414 # optimal powertrain sprocket ratio for velocity
N_BASE_POSITION = 4.29293 # optimal powertrain sprocket ratio for position

def motorTorque(omega):
    """Returns motor torque [Nm] given motor velocity [rad/s]"""
    rpm = omega * 60 / (2 * numpy.pi)

    constant = 352 * numpy.heaviside(4500 - rpm, 4500)
    curve = numpy.heaviside(rpm - 4500, 4500) * ((0.00000721212 * (rpm ** 2)) + (-0.15842 * rpm) + 918.02797)
    nominal = constant + curve

    return ETA * nominal

def vehicleVelocity(omega, sprocket_ratio):
    """Returns vehicle velocity [m/s] given motor velocity [rad/s] and powertrain sprocket ratio"""
    omega_axle = omega / sprocket_ratio
    return omega_axle * R_WHEEL

def motorVelocity(vehicle_velocity, sprocket_ratio):
    """Returns motor velocity [rad/s] given vehicle velocity [m/s] and powertrain sprocket ratio"""
    omega_axle = vehicle_velocity / R_WHEEL
    return omega_axle * sprocket_ratio

def externalForces(omega, sprocket_ratio):
    """Returns external forces (drag, rolling resistance) [N] given motor velocity [rad/s] and powertrain sprocket ratio"""
    v_x = vehicleVelocity(omega, sprocket_ratio)

    f_gravity = 9.81 * M_CAR
    f_lift = 0.5 * RHO * C_L * (v_x ** 2) * A_WING
    f_normal = f_gravity + f_lift
    f_rolling_resistance = 4 * C_R * f_normal

    f_drag = 0.5 * RHO * C_D * (v_x ** 2) * A_FRONT

    return f_rolling_resistance + f_drag

def wheelForce(omega, sprocket_ratio):
    """Returns force from wheels [N] given motor velocity [rad/s] and powertrain sprocket ratio. Accounts for static friction"""
    torque_axle = motorTorque(omega) * sprocket_ratio
    nominal_tractive = torque_axle / R_WHEEL
    
    v_x = vehicleVelocity(omega, sprocket_ratio)
    f_gravity = 9.81 * M_CAR
    f_lift = 0.5 * RHO * C_L * (v_x ** 2) * A_WING
    f_normal = f_gravity + f_lift
    friction_max = K_STATIC * f_normal * REAR_DIST

    final = numpy.minimum(friction_max, nominal_tractive)
    return final

def deltaV(omega_i, d_t, sprocket_ratio):
    """Returns final motor velocity [rad/s] given initial motor velocity [rad/s], time differential [sec], and powertrain sprocket ratio"""
    v_i = vehicleVelocity(omega_i, sprocket_ratio)
    back_force = externalForces(omega_i, sprocket_ratio)
    driving_force = wheelForce(omega_i, sprocket_ratio)

    a_car = (1 / M_CAR) * (driving_force - back_force)
    v_f = v_i + (a_car * d_t)

    return motorVelocity(v_f, sprocket_ratio)

OMEGA_MIN = 0 # minimum motor velocity [rad/s]
OMEGA_MAX = 950 # maximum motor velocity [rad/s]
V_TARGET = 26.8224 # target vehicle velocity [m/s]
OMEGA_TARGET = motorVelocity(V_TARGET, N_BASE_VELOCITY) # motor speed at target vehicle velocity [rad/s]
POSITION_TARGET = 75 # acceleration event distance [m]
D_T = 0.001 # time differential for integration solving [sec]
SAMPLE_SIZE = 100

OMEGA_SERIES = numpy.linspace(OMEGA_MIN, OMEGA_MAX, SAMPLE_SIZE)
TIME_SERIES_VELOCITY = numpy.arange(0, 10, D_T)
TIME_SERIES_POSITION = numpy.arange(0, 5, D_T)
SPROCKET_RATIO_SERIES = numpy.linspace(3, 7, SAMPLE_SIZE)

motor_torque_series = motorTorque(OMEGA_SERIES)

output_plot.plot(OMEGA_SERIES, motor_torque_series, label="Motor torque", color="blue")
output_plot.axvline(OMEGA_TARGET, label="Target motor velocity", color="black")
output_plot.set_xlabel("Motor velocity [rad/s]")
output_plot.set_ylabel("Motor torque [Nm]")
output_plot.set_title("Powertrain outputs")
output_plot.legend(loc="lower left")

external_force_series = externalForces(OMEGA_SERIES, N_BASE_VELOCITY)
wheel_force_series = wheelForce(OMEGA_SERIES, N_BASE_VELOCITY)
net_force_series = wheel_force_series - external_force_series

force_plot.plot(OMEGA_SERIES, external_force_series, label="Resistive forces", color="blue")
force_plot.plot(OMEGA_SERIES, wheel_force_series, label="Force from tyres", color="red")
force_plot.plot(OMEGA_SERIES, net_force_series, label="Net force", color="orange")
force_plot.axvline(OMEGA_TARGET, label="Target motor velocity", color="black")
force_plot.set_xlabel("Motor velocity [rad/s]")
force_plot.set_ylabel("Force [N]")
force_plot.set_title("Forces acting on vehicle")
force_plot.legend(loc="upper right")

velocity_time_base = 10.00
velocity_times = []

for r in SPROCKET_RATIO_SERIES:
    omega_final = motorVelocity(V_TARGET, r)
    omega = 0

    for t in TIME_SERIES_VELOCITY:
        omega = deltaV(omega, D_T, r)
        if omega >= omega_final:
            velocity_times.append(t)
            break

velocity_plot.plot(SPROCKET_RATIO_SERIES, velocity_times, label="Time to reach target velocity", color="blue")
velocity_plot.axvline(N_BASE_VELOCITY, label="Optimal sprocket ratio", color="black")
velocity_plot.set_xlabel("Sprocket ratio")
velocity_plot.set_ylabel("Time [sec]")
velocity_plot.set_title("Time to reach target velocity vs sprocket ratio")
velocity_plot.legend(loc="upper right")

print("Optimal sprocket ratio for velocity:", SPROCKET_RATIO_SERIES[velocity_times.index(min(velocity_times))])
print("Time to accelerate to", V_TARGET, "[m/s]:", min(velocity_times), "[sec] \n")

optimal_ratio_velocity_series = []
omega = 0
for t in TIME_SERIES_VELOCITY:
    omega = deltaV(omega, D_T, N_BASE_VELOCITY)
    velocity = vehicleVelocity(omega, N_BASE_VELOCITY)
    optimal_ratio_velocity_series.append(velocity)

optimal_ratio_velocity_plot.plot(TIME_SERIES_VELOCITY, optimal_ratio_velocity_series, label="Vehicle speed", color="blue")
optimal_ratio_velocity_plot.axhline(V_TARGET, label="Target velocity", color="black")
optimal_ratio_velocity_plot.set_xlabel("Time [sec]")
optimal_ratio_velocity_plot.set_ylabel("Vehicle speed [m/s]")
optimal_ratio_velocity_plot.set_title("Vehicle speed at optimal velocity sprocket ratio: " + str(N_BASE_VELOCITY))
optimal_ratio_velocity_plot.legend(loc="lower right")

position_time_base = 10.00
position_times = []

for r in SPROCKET_RATIO_SERIES:
    omega_final = motorVelocity(V_TARGET, r)
    omega = 0
    position = 0

    for t in TIME_SERIES_VELOCITY:
        omega = deltaV(omega, D_T, r)
        velocity = vehicleVelocity(omega, r)
        position = position + (velocity * D_T)

        if position > POSITION_TARGET:
            position_times.append(t)
            break

position_plot.plot(SPROCKET_RATIO_SERIES, position_times, label="Event times", color="blue")
position_plot.axvline(N_BASE_POSITION, label="Optimal sprocket ratio", color="black")
position_plot.set_xlabel("Sprocket ratio")
position_plot.set_ylabel("Time [sec]")
position_plot.set_title("Time to reach target position vs sprocket ratio")
position_plot.legend(loc="lower right")

print("Optimal sprocket ratio for position:", SPROCKET_RATIO_SERIES[position_times.index(min(position_times))])
print("Time to reach", POSITION_TARGET, "[m]:", min(position_times), "[sec]")

optimal_ratio_position_series = []
position = 0
omega = 0
for t in TIME_SERIES_POSITION:
    omega = deltaV(omega, D_T, N_BASE_POSITION)
    velocity = vehicleVelocity(omega, N_BASE_POSITION)
    position = position + (velocity * D_T)
    optimal_ratio_position_series.append(position)

optimal_ratio_position_plot.plot(TIME_SERIES_POSITION, optimal_ratio_position_series, label="Vehicle position", color="blue")
optimal_ratio_position_plot.axhline(POSITION_TARGET, label="Target position", color="black")
optimal_ratio_position_plot.set_xlabel("Time [sec]")
optimal_ratio_position_plot.set_ylabel("Position [m]")
optimal_ratio_position_plot.set_title("Vehicle position at optimal position sprocket ratio: " + str(N_BASE_POSITION))
optimal_ratio_position_plot.legend(loc="lower right")

END_TIME = time()
elapsed_time = END_TIME - START_TIME
print("\nSimulation time:", elapsed_time, "[sec]")

plt.suptitle("UCI FSAE EV powertrain simulations")
plt.show()