import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import optimize
import numpy as np

from aeropy.geometry.airfoil import CST
from aeropy.CST_2D import calculate_arc_length, dxi_u

def calculate_c(length_0, A, TE_displacement, N1, N2):
    """Equations in the New_CST.pdf. Calculates the upper chord in order for
       the cruise and landing airfoils ot have the same length."""
    
    def integrand(psi, Au, delta_xi, N1, N2):
        return np.sqrt(1 + dxi_u(psi, Au, delta_xi, N1, N2)**2)
    
    def f(current_c):
        """Function dependent of c_C and that outputs c_C.""" 
        current_length, err = quad(integrand, 0, 1, args=(A, TE_displacement/current_c, N1, N2))
        A0 = -TE_displacement/(length_0/current_length)
        A[0] = A0
        print(current_c, current_length)
        return length_0/current_length

    current_c = optimize.fixed_point(f, length_0, maxiter=100)
    #In case the calculated chord is really close to the original, but the
    #algorithm was not able to make them equal
    if abs(length_0 - current_c) < 1e-7:
        return length_0
    #The output is an array so it needs the extra [0]
    #Must redefine A as the returned A after calling calculate_c()
    return current_c, A
    
# Class coefficients (fixed)
N1 = 1.
N2 = 1.

# Initial length
length_0 = 1.

# Delta x
TE_displacement = 0.1

# Shape coefficients
A = [-TE_displacement, 0.2]

# new chord
current_chord = calculate_c(length_0, A, TE_displacement, N1, N2)
#current_chord = 1
# non-dimensional y coordinates
y = np.linspace(0,current_chord)

# non-dimensional x coordinates
x = CST(y, current_chord, TE_displacement, Au=A, N1 = N1, N2 = N2)


dxi = dxi_u(psi = np.array(y)/current_chord, Au = A, 
            delta_xi = TE_displacement/current_chord,
            N1 = N1, N2 = N2)

# Plotting
plt.plot(x, y, label='A = '+ str(A[0]))
plt.plot(dxi, y, label='dxi')
plt.axis('equal')
plt.legend()
plt.grid()
plt.xlabel(r'$x$', fontsize = 14.)
plt.ylabel('$y$', fontsize = 14.)
plt.show()


length_0 = 1.
for TE_displacement in np.linspace(0,0.4,3):
    for A_i in np.linspace(-0.4,0,3):
        # Shape coefficients
        A = [-TE_displacement,A_i]

        # non-dimensional y coordinates
        psi = np.linspace(0,length_0,1000)

        # non-dimensional x coordinates
        xi = CST(psi, length_0, TE_displacement, Au=A, N1 = N1, N2 = N2)

        # Plotting
        plt.plot(xi, psi, label='A = '+ str(A_i) + r', $\Delta \xi$ = ' + str(TE_displacement))
plt.axis('equal')
plt.legend()
plt.xlabel(r'$\xi$', fontsize = 14.)
plt.ylabel('$\psi$', fontsize = 14.)
plt.grid()
plt.show()

# Plotting change in x and y domain
chord = []
for TE_displacement in np.linspace(0,0.4,3):
    for A_i in np.linspace(-0.4,0.,3):
        chord_i = calculate_c(length_0, [-TE_displacement,A_i], TE_displacement, N1, N2)
        chord.append(chord_i)
        
         # non-dimensional y coordinates
        psi = np.linspace(0,chord_i,1000)

        # non-dimensional x coordinates
        xi = CST(psi, chord_i, TE_displacement, Au=[-TE_displacement,A_i], N1 = N1, N2 = N2)
        
        # Plotting
        plt.plot(xi, psi, label='A = '+ str(A_i) + r', $\Delta \xi$ = ' + str(TE_displacement))   
plt.axis('equal')
plt.legend()
plt.xlabel('$x$', fontsize = 14.)
plt.ylabel('$y$', fontsize = 14.)
plt.grid()
plt.show()
