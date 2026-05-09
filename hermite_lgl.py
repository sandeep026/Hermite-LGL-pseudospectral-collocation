import casadi as cs
from lgltools import LGL
import numpy as np
import matplotlib.pyplot as plt

# dynamics
x   =cs.SX.sym('x',2,1)
u   =cs.SX.sym('u')
f   =cs.vertcat(x[1],-x[1]+u)
xdot=cs.Function('xdot',[x,u],[f])
obj =cs.Function('obj',[x,u],[u**2])

# number of intervals
m=1000
# number of states and controls
nx=2
nu=1

#polynomial order must be odd
N=3 #hermite simpson
if N%2==0:
    raise ValueError(f'Even polynomial not allowed n={N}')
# nodes at which the polynomial passes through
n1=int((N+1)/2)
# collocation points
n2=int((N-1)/2)
lgl=LGL(N)
# lgl points
tau=lgl.tau.T
# polynomial passes at these points
tau_n1=tau[:,0::2]
# evaluate dynamics at these points as defect constraints
tau_n2=tau[:,1::2]

# intervals
t0=0
tf=2
t=t0+((tf-t0)*cs.linspace(0,m,m+1)/m).T

ta=cs.MX.sym('ta',1)   
td=cs.MX.sym('td',1)
f1=cs.Function('f1',[ta,td],[td*tau+ta]).map(m,'thread')
tr=t[:,1:]
tl=t[:,0:-1]
td=0.5*(tr-tl)
ta=0.5*(tr+tl)
t_seg=f1(ta,td)
dt_seg=tr-tl

# decision variables
nlp=cs.Opti()
x_seg_n1  =nlp.variable(nx,int(m*n1)) # overlap of states and control at each segment

u_c  =nlp.variable(nu,int(m))
us=cs.MX.sym('us',nu,1)
repeat=cs.Function('repeat',[us],[us@cs.DM.ones(1,n1)]).map(m,'thread')
u_seg_n1 =repeat(u_c)
repeat=cs.Function('repeat',[us],[us@cs.DM.ones(1,n2)]).map(m,'thread')
u_seg_n2 =repeat(u_c)
repeat=cs.Function('repeat',[us],[us@cs.DM.ones(1,N)]).map(m,'thread')
u_seg=repeat(u_c)


# polynomial
tau_s       =cs.MX.sym('tau_s')
a_i         =cs.MX.sym('a_i',nx,N+1)
tau_coeff   = cs.hcat([tau_s**j for j in range(N+1)])
poly        =a_i@tau_coeff.T
poly_der    =cs.jacobian(poly,tau_s)
X_a         =cs.Function('X_a',[a_i, tau_s],[poly])
X_a_der     =cs.Function('X_a_der',[a_i, tau_s],[poly_der])

xn1s=cs.MX.sym('xn1s',nx,n1)
un1s=cs.MX.sym('un1s',nu,n1)
dts=cs.MX.sym('dts',1)

taucoe=cs.Function('taucoe',[tau_s],[tau_coeff.T])
der_coeff=cs.jacobian(tau_coeff.T,tau_s)
dercoe=cs.Function('dercoe',[tau_s],[der_coeff])
coeff=cs.horzcat(xn1s,0.5*xdot.map(n1,'thread')(xn1s,un1s)*(dts))@cs.inv(cs.horzcat(taucoe(tau_n1),dercoe(tau_n1)))    

cc=cs.Function('cc',[dts,xn1s,un1s],[cs.simplify(coeff)])

taucoe=cs.Function('taucoe',[tau_s],[tau_coeff.T])
taucoeN=taucoe.map(N,'thread')
taucoen2=taucoe.map(n2,'thread')

der_coeff=cs.jacobian(tau_coeff.T,tau_s)
dercoe=cs.Function('dercoe',[tau_s],[der_coeff])
dercoen2=dercoe.map(n2,'thread')

tau_n2_s=cs.MX.sym('tau_n1_s',1,n2)
tau_N_s=cs.MX.sym('tau_n2_s',1,N)

x_seg_n2_f=cs.Function('xs',[dts,xn1s,un1s],[coeff@taucoen2(tau_n2)])
x_seg_n2_der_f=cs.Function('xsder',[dts,xn1s,un1s],[coeff@dercoen2(tau_n2)])
x_seg_N_f=cs.Function('xs',[dts,xn1s,un1s],[coeff@taucoeN(tau)])

x_seg_n2=x_seg_n2_f.map(m,'thread',6)(dt_seg,x_seg_n1,u_seg_n1)
x_seg_der_n2=x_seg_n2_der_f.map(m,'thread',6)(dt_seg,x_seg_n1,u_seg_n1)
x_seg=x_seg_N_f.map(m,'thread',6)(dt_seg,x_seg_n1,u_seg_n1)
xn  =cs.MX.sym('xn',nx,N)
un  =cs.MX.sym('un',nu,N)
dts =cs.MX.sym('dts',1)
quad=cs.Function('quad',[xn,un,dts],[0.5*dts*cs.dot(lgl.wi.T,obj.map(N,'thread')(xn,un))]).map(m,'thread',6)
quad=cs.sum2(quad(x_seg,u_seg,dt_seg))
xn2  =cs.MX.sym('xn',nx,n2)
un2  =cs.MX.sym('un',nu,n2)
fn2=cs.Function('fn2',[xn2,un2,dts],[0.5*dts*xdot.map(n2,'unroll')(xn2,un2)])
defect=cs.vec(x_seg_der_n2-fn2.map(m,'thread',6)(x_seg_n2,u_seg_n2,dt_seg))
event=cs.vertcat(x_seg[:,0],cs.DM([1,-2.694528]).T@x_seg[:,-1]+1.155356)

# End indices of segments except the last one
left = [(i + 1) * n1 - 1 for i in range(m - 1)]
# Start indices of segments except the first one
right = [(i + 1) * n1 for i in range(m - 1)]
contin=x_seg_n1[:,left]-x_seg_n1[:,right]

nlp.minimize(quad)
nlp.subject_to(event==0)
nlp.subject_to(defect==0)
nlp.subject_to(contin==0)
nlp.solver('ipopt')
nlp.set_initial(x_seg_n1,cs.DM.rand(x_seg_n1.shape))
nlp.set_initial(u_c,cs.DM.rand(u_c.shape))
sol=nlp.solve()

print(cs.n_nodes(cs.gradient(nlp.f,nlp.x)),cs.n_nodes(cs.jacobian(nlp.g,nlp.x)))


X_s=sol.value(x_seg)
U_s=sol.value(u_seg)
t_s=sol.value(t_seg)
plt.plot(t_s.flatten(),X_s[0].flatten(),t_s.flatten(),X_s[1].flatten())
plt.figure()
plt.plot(t_s.flatten(),U_s.flatten())
