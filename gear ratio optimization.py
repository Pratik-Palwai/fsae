import numpy
import matplotlib.pyplot as plt

figure = plt.figure()
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.subplots_adjust(left=0.053, bottom=0.069, right=0.985, top=0.9, wspace=0.18, hspace=0.383)

nrows = 2
ncols = 2
torque_plot = figure.add_subplot(nrows, ncols, 1)
accels_plot = figure.add_subplot(nrows, ncols, 2)
force_plot = figure.add_subplot(nrows, ncols, 3)
optimal_plot = figure.add_subplot(nrows, ncols, 4)

M_CAR = 500 # mass of car and driver [kg]
R_WHEEL = 0.2286 # radius of tyre [m]
A_FRONT = 1.5 # frontal area [m^2]
A_WING = 4.0 # airfoil area [m^2]
RHO = 1.225 # density of air at sea level [kg / m^3]
C_D = 1.01 # drag coefficient
C_L = 1.75 # lift (downforce) coefficient
C_R = 0.015 # rolling resistance coefficient
N_BASE = 5.0 # powertrain sprocket ratio
ETA = 0.73 # overall powertrain efficiency
K_STATIC = 1.25 # static friction coefficient for tyres

def motorTorque(omega):
    """Returns motor torque [Nm] given motor velocity [rad/s]"""
    rpm = omega * 60 / (2 * numpy.pi)

    constant = 352 * numpy.heaviside(4500 - rpm, 4500)
    curve = numpy.heaviside(rpm - 4500, 4500) * ((0.00000721212 * (rpm ** 2)) + (-0.15842 * rpm) + 918.02797)
    nominal = ETA * (constant + curve)

    return nominal

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
    nominal = torque_axle / R_WHEEL
    
    v_x = vehicleVelocity(omega, sprocket_ratio)
    f_gravity = 9.81 * M_CAR
    f_lift = 0.5 * RHO * C_L * (v_x ** 2) * A_WING
    f_normal = f_gravity + f_lift
    friction_max = K_STATIC * f_normal

    final = numpy.minimum(friction_max, nominal)
    return final

def deltaTime(omega_i, d_t, sprocket_ratio):
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
OMEGA_TARGET = motorVelocity(V_TARGET, N_BASE) # motor speed at target vehicle velocity [rad/s]
D_T = 0.001 # time differential for integration solving [sec]

omega_series = numpy.linspace(OMEGA_MIN, OMEGA_MAX, 100)
time_series = numpy.arange(0, 10, D_T)

motor_torque_series = motorTorque(omega_series)

torque_plot.plot(omega_series, motor_torque_series, label="Motor torque", color="blue")
torque_plot.axvline(OMEGA_TARGET, label="Target motor velocity", color="violet")
torque_plot.set_xlabel("Motor velocity [rad/s]")
torque_plot.set_ylabel("Motor torque [Nm]")
torque_plot.set_title("Powertrain torque curve")
torque_plot.legend(loc="lower left")

sprocket_ratio_series = numpy.linspace(3, 8, 100)
accel_time_base = 10.00
accel_times = []

for g in sprocket_ratio_series:
    omega_final = motorVelocity(V_TARGET, g)
    omega = 0
    for t in time_series:
        omega = deltaTime(omega, D_T, g)
        if omega >= omega_final:
            accel_times.append(t)
            if t <= accel_time_base:
                N_BASE = g
                accel_time_base = t
            break

accels_plot.plot(sprocket_ratio_series, accel_times, label="Time to reach target velocity", color="blue")
accels_plot.axvline(N_BASE, label="Optimal sprocket ratio", color="violet")
accels_plot.set_xlabel("Powertrain sprocket ratio")
accels_plot.set_ylabel("Time [sec]")
accels_plot.set_title("Acceleration times vs powertrain sprocket ratio")
accels_plot.legend(loc="upper right")
ext_force_series = externalForces(omega_series, N_BASE)
wheel_force_series = wheelForce(omega_series, N_BASE)
net_force_series = wheel_force_series - ext_force_series

force_plot.plot(omega_series, ext_force_series, label="Resistive forces", color="blue")
force_plot.plot(omega_series, wheel_force_series, label="Force from tyres", color="red")
force_plot.plot(omega_series, net_force_series, label="Net force", color="orange")
force_plot.axvline(OMEGA_TARGET, label="Target motor velocity", color="violet")
force_plot.set_xlabel("Motor velocity [rad/s]")
force_plot.set_ylabel("Force [N]")
force_plot.set_title("Forces acting on vehicle")
force_plot.legend(loc="upper right")

optimal_run = []
omega = 0
for t in time_series:
    omega = deltaTime(omega, D_T, N_BASE)
    optimal_run.append(vehicleVelocity(omega, N_BASE))

optimal_plot.plot(time_series, optimal_run, label="Vehicle speed", color="blue")
optimal_plot.axhline(V_TARGET, label="Target velocity", color="violet")
optimal_plot.set_xlabel("Time [sec]")
optimal_plot.set_ylabel("Vehicle speed [m/s]")
optimal_plot.set_title("Vehicle speed at optimal sprocket ratio: " + str(N_BASE))
optimal_plot.legend(loc="lower right")

plt.suptitle("UCI FSAE EV powertrain simulations")
plt.tight_layout()
plt.show()
