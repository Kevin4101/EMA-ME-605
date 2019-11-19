"""
Condition Set 1
"""

#include FEniCS headers
from __future__ import print_function
from fenics import *
from ufl import nabla_div, nabla_grad, div

#Problem parameters variables
L = 1;   #Beam length
W = ([0,0.03]x[0,0.08]); #Beam width and Height 
dx = W/L
g = 1.0; #gravity constant for forcing function

#Create mesh
#The points represent the diagonal ends of the box, followed by the number of elments along each dimension
mesh = RectangleMesh(Point(0, 0), Point(L, W), 30, 80) #2D mesh
#mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 10, 3, 3) #3D mesh

#Define function space.
#This creates a H1 function space from which we can construct the solution (trail) and test function spaces.
V = FunctionSpace(mesh, 'P', 1)

# Define needed u (trial space), w (test space) function spaces
u = TrialFunction(V)
w = TestFunction(V)

# Define boundary condition
g1 = 300
g2 = 310

# Define function to identify bottom boundary
def bot_boundary(y=0,on_boundary):
   return on_boundary
# Define function to identify top boundary
def top_boundary(y=0.08,on_boundary):
   return on_boundary

# Create the boundary condition
bc1 = DirichletBC(V, g1, bot_boundary) #2D
bc2 = DirichletBC(V, g2, top_boundary) #2D
#bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary) #3D

# Define initial value
u_n = interpolate(g1,V)

# Define other needed constants
d = u.geometric_dimension()  # space dimension
f = Constant((0, -rho*g))    #forcing function in 2D
#f = Constant((0, 0, -rho*g)) #forcing function in 3D
T = Constant((0, 0))         #traction (Neumann B.C) in 2D
#T = Constant((0, 0, 0))      #traction (Neumann B.C) in 3D

#Define the weak form of the problem
F = u*v*dx - u_n*v*dx

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

