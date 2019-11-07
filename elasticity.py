"""
Tutorial program for ME/EMA 605. Modified from the corresponding FEnicCS tutorial demo program.

Linear elasticity problem in 2D/3D.

   div(sigma) + f = 0 
   
The model is used to simulate a 2D/3D elastic beam clamped at
its left end and deformed under its own weight (non-zero forcing function f).
"""

#include FEniCS headers
from __future__ import print_function
from fenics import *
from ufl import nabla_div, nabla_grad, div

#Problem parameters variables
L = 1;   #Beam length
W = 0.2; #Beam width and Height 
mu = 1; #Lame parameters
lambda_=1.25; #Lame parameters
rho = 1.0e-2; #density  
delta = W/L
g = 1.0; #gravity constant for forcing function

#Create mesh
#The points represent the diagonal ends of the box, followed by the number of elments along each dimension
mesh = RectangleMesh(Point(0, 0), Point(L, W), 10, 3) #2D mesh
#mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 10, 3, 3) #3D mesh

#Define function space.
#This creates a H1 function space from which we can construct the solution (trail) and test function spaces.
V = VectorFunctionSpace(mesh, 'P', 1)

# Define needed u (trial space), w (test space) function spaces
u = TrialFunction(V)
w = TestFunction(V)

# Define boundary condition
tol = 1E-14 #tolerance
# Define function to identify the left boundary that is clamped
def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol
# Create the boundary condition
bc = DirichletBC(V, Constant((0, 0)), clamped_boundary) #2D
#bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary) #3D

# Define expressions for strain and stress
#strain
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
#stress (St.Venant Kirchhoff model)
def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# Define other needed constants
d = u.geometric_dimension()  # space dimension
f = Constant((0, -rho*g))    #forcing function in 2D
#f = Constant((0, 0, -rho*g)) #forcing function in 3D
T = Constant((0, 0))         #traction (Neumann B.C) in 2D
#T = Constant((0, 0, 0))      #traction (Neumann B.C) in 3D

#Define the weak form of the problem
a = inner(sigma(u), epsilon(w))*dx #LHS of the weak form
L = dot(f, w)*dx + dot(T, w)*ds    #RHS of the weak form

# Compute solution by solving the weak formulation
u = Function(V)
solve(a == L, u, bc)

# Compute stress for plotting (This is a post-processing step)
s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)  # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s)) # Von Mises stress
V = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, V)

# Compute magnitude of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)

# Save solution to file in VTK format (Can be visualized with Paraview or Visit)
File('displacement.pvd') << u
File('von_mises.pvd') << von_Mises
File('magnitude.pvd') << u_magnitude

