import numpy as np
import matplotlib.pyplot as plt
import sys

class Vec3d:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        return
    
    def __str__(self):
        return f'x: {self.x} y: {self.y} z: {self.z}'

    def __repr__(self):
        return f'x: {self.x} y: {self.y} z: {self.z}'

def beta_to_pwm(beta):

    Popt = [ 1.59587557e-01,  1.58019463e+03, -2.04235057e-03,  4.34184765e-01,
    3.74410204e-02,  2.18558980e-01]

    
    offset = Popt[1]
    gain = Popt[2]
    outer_gain = Popt[3]
    lr = Popt[4]
    lf = Popt[5]

    L = lr + lf

    steering_angle = np.arctan(L/lr * np.tan(beta))
    u = np.tan(steering_angle / outer_gain) / gain + offset

    return u

def beta_to_angle(beta):

    Popt = [ 1.59587557e-01,  1.58019463e+03, -2.04235057e-03,  4.34184765e-01,
    3.74410204e-02,  2.18558980e-01]

    
    offset = Popt[1]
    gain = Popt[2]
    outer_gain = Popt[3]
    lr = Popt[4]
    lf = Popt[5]

    L = lr + lf

    steering_angle = np.arctan(L/lr * np.tan(beta))

    return steering_angle

def anlge_to_pwm(steering_angle):

    Popt = [ 1.59587557e-01,  1.58019463e+03, -2.04235057e-03,  4.34184765e-01,
    3.74410204e-02,  2.18558980e-01]

    
    offset = Popt[1]
    gain = Popt[2]
    outer_gain = Popt[3]
    lr = Popt[4]
    lf = Popt[5]

    L = lr + lf

    u = np.tan(steering_angle / outer_gain) / gain + offset

    return u


folder = 'validation-data/'
files = [folder + 'id_data.npz'] + [folder + 'id_data (%d).npz'%n for n in range(1,7)]

states = []
t0 = 0

#### data ####
t = []
u_a = []
u_y = []
imu_a = []
cam_a = []
vive_v_linear = []
vive_v_angular = []
##############


for filename in files:
    data = np.load(filename, allow_pickle = True)
    t.extend(data['t'] + t0)
    u_a.extend(data['u_a'])
    u_y.extend(data['u_y'])
    imu_a.extend(data['imu_a'])
    cam_a.extend(data['cam_a'])
    vive_v_linear.extend(data['vive_vlinear'])
    vive_v_angular.extend(data['vive_vangular'])
    t0 = t[-1] + 0.01
    


n = len(t)
t = np.array(t)
v1 = np.array([v.x for v in vive_v_linear])
v2 = np.array([v.y for v in vive_v_linear])
v3 = np.array([v.z for v in vive_v_linear])
w1 = np.array([w.x for w in vive_v_angular])
w2 = np.array([w.y for w in vive_v_angular])
w3 = np.array([w.z for w in vive_v_angular])
a1 = np.array([a.x for a in imu_a])
a2 = np.array([a.y for a in imu_a])
a3 = np.array([a.z for a in imu_a])

u = np.array([anlge_to_pwm(i) for i in u_y])

t0 = t[0]
tf = t[-1]

u[u==None]=1500


import casadi as ca

N_opt = 1500

v  = ca.interpolant('v',  'linear', [t], np.sqrt(v1**2+v2**2))
v1 = ca.interpolant('v1', 'linear', [t], v1)
v2 = ca.interpolant('v2', 'linear', [t], v2)
w3 = ca.interpolant('w3', 'linear', [t], w3)
u  = ca.interpolant('u', 'linear', [t], u)
u_cmd  = ca.interpolant('u', 'linear', [t], u_y)


delay = ca.MX.sym('delay')
offset = ca.MX.sym('offset')
gain = ca.MX.sym('gain')
outer_gain = ca.MX.sym('gain2')
lr = ca.MX.sym('lr')
lf = ca.MX.sym('lf')
L = lr + lf

params = ca.vertcat(delay, offset, gain, outer_gain, lr, lf)

t = ca.MX.sym('t')


def smooth_abs(x):
    return ca.sqrt(x**2 + 1e-6**2)
    
def smooth_sign(x):
    return x / smooth_abs(x)

steering_angle = outer_gain*(ca.arctan((u(t-delay) - offset) * gain))
beta = ca.arctan(lr/L*ca.tan(steering_angle))
beta_func = ca.Function('beta',[t, params], [beta])

w3_model =  v(t) * ca.cos(beta) / L * ca.tan(steering_angle)
w3_func  = ca.Function('w3_fit',[t, params], [w3_model])

v1_model = v(t) * ca.cos(beta)
v2_model = v(t) * ca.sin(beta)
v1_func  = ca.Function('v1_fit',[t,params], [v1_model])
v2_func  = ca.Function('v2_fit',[t,params], [v2_model])



error_model = (w3_func(t,params) - w3(t))**2 \
            + (v2_func(t,params) - v2(t))**2

error_func = ca.Function('error', [t, params], [error_model])


opti = ca.Opti()
P = opti.variable(params.size()[0])
opti.set_initial(P, [0.03,1500, 0.00, 1.00, 0.1, 0.1])
J = 0 
for tc in np.linspace(t0,tf,N_opt):
    J += error_func(tc, P)
fJ = ca.Function('J',[P],[J])

opti.minimize(J)

opti.subject_to(P[0] >= 0)
opti.subject_to(P[0] <= 0.2)
opti.subject_to(P[4] >= 0.01)
opti.subject_to(P[5] >= 0.01)
opti.subject_to(P[4] + P[5] == 0.256)

opti.solver('ipopt')
sol = opti.solve()

Popt = sol.value(P)


tspan = np.linspace(t0, tf, N_opt)
v1_meas = v1(tspan)
v2_meas = v2(tspan)
w3_meas = w3(tspan)
v_meas = v(tspan)

beta_fit = np.array(beta_func(tspan[None], Popt)).squeeze()

v1_fit = np.array(v1_func(tspan[None], Popt)).squeeze()
v2_fit = np.array(v2_func(tspan[None], Popt)).squeeze()
w3_fit = np.array(w3_func(tspan[None], Popt)).squeeze()
e_sol = np.array(error_func(tspan[None], Popt)).squeeze()
u_sol = np.array(u(tspan[None])).squeeze()

print(Popt)
print(fJ(Popt))
print('Cumulative error: ', sum(e_sol))

u_meas = beta_to_angle(beta_fit)
u_cmd = np.array(u_cmd(tspan[None])).squeeze()


plt.subplot(2,1,1)
plt.plot(tspan, u_cmd, label='cmd')
plt.plot(tspan, u_meas, label='meas')
plt.legend()
plt.subplot(2,1,2)
plt.plot(tspan, np.abs(u_cmd-u_meas), label='error')
plt.legend()
plt.show()
