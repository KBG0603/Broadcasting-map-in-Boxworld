##############################################################################
# Joint Broadcasting Map Characterization
##############################################################################




import sympy as sp

import numpy as np


##############
# Tensor
#############

def tensor(*args):

   M = 1;
   for j in range(len(args)):
       if type(args[j]) is tuple:
           for k in range(args[j][1]):
               M = np.kron(M,args[j][0])
       else:
            M = np.kron(M,args[j])
   return M
 


##############################################################################
# Evaluation of effect on  a state
##############################################################################

def eval_effect(effect, state):
    """
    <effect,state>
    """
    return sp.expand((effect.T*state)[0])




def to_sp(x):
    return sp.Matrix(x)


##############################################################################
# Generic partial trace
##############################################################################

def PTrX(rho, sys, dim):
    """
    GPT partial trace by contraction with the unit effect.

    Parameters
    ----------
    rho : SymPy column vector

    sys : list or tuple
        Subsystems to trace out (1-indexed).

    dim : list or tuple
        Local dimensions, e.g. [3,3,3,3]

    Returns
    -------
    SymPy column vector
    """
    
    if isinstance(sys, int):
        sys = [sys]

    assert len(dim) >= max(sys), \
        "Subsystem index exceeds number of systems."
    
    ops = []

    for i, d in enumerate(dim, start=1):

        if i in sys:
            ops.append(Tr.T)
        else:
            ops.append(sp.eye(d))

    return sp.expand(
        to_sp(tensor(*ops)) * rho
    )



##############################################################################
# Basis vectors (standard orthogonal basis vector in a 3-dimensional space)
##############################################################################

e1 = sp.Matrix([1,0,0])
e2 = sp.Matrix([0,1,0])
e3 = sp.Matrix([0,0,1])

basis = [e1,e2,e3]


# Define box-world states
rho00 = sp.Matrix([0, 0, 1])
rho10 = sp.Matrix([1, 0, 1])
rho01 = sp.Matrix([0, 1, 1])
rho11 = sp.Matrix([1, 1, 1])

# Convenient basis (basis leading to irreducible subrepresenation of the diheddral group)
omega = sp.Rational(1, 2) * (rho00 + rho11)
v = rho01 - rho10
u = rho00 - rho11

# Effects
Tr = sp.Matrix([0, 0, 1])
ex = sp.Matrix([1, 0, 0])
exbar = Tr - ex

# Single-system maps (from the solution of the single-system problem)
def N1(s):
    return sp.expand(to_sp(tensor(s, omega)))

def N2(s):
    return sp.expand(to_sp(tensor(omega, s)))

def N3(s):
    return sp.expand(eval_effect(Tr, s) * to_sp(tensor(omega, omega)))

def N4(s):
    return sp.expand(eval_effect(Tr, s) * (to_sp(tensor(v, v)) + to_sp(tensor(u, u))))

N = [N1, N2, N3, N4]

# Adjacent and diagonal classical bases
adj_states  = [rho10, rho00]
adj_effects = [ex, exbar]

diag_states  = [rho10, rho01]
diag_effects = [ex, exbar]




######### 2 input 4 output broadcasting ######################


##############################################################################
# Two-system dephasing
##############################################################################

def DD(X, statesA, effectsA, statesB, effectsB):

    out = sp.zeros(9,1)

    for stA, effA in zip(statesA, effectsA):

        for stB, effB in zip(statesB, effectsB):

            eff = to_sp(
                tensor(effA, effB)
            )

            st = to_sp(
                tensor(stA, stB)
            )

            out += eval_effect(eff, X) * st

    return sp.expand(out)




##############################################################################
# Two-system classical broadcaster
##############################################################################

def BB_cl(X, statesA, effectsA, statesB, effectsB):

    out = sp.zeros(81,1)

    for stA, effA in zip(statesA, effectsA):

        for stB, effB in zip(statesB, effectsB):

            eff = to_sp(
                tensor(effA, effB)
            )

            st = to_sp(
                tensor(
                    stA, stA,
                    stB, stB
                )
            )

            out += eval_effect(eff, X) * st

    return sp.expand(out)








##############################################################################
# Four-system dephasing
##############################################################################

def DDDD(X, statesA, effectsA, statesB, effectsB):

    out = sp.zeros(81,1)

    for stA1, effA1 in zip(statesA, effectsA):
        for stA2, effA2 in zip(statesA, effectsA):
            for stB1, effB1 in zip(statesB, effectsB):
                for stB2, effB2 in zip(statesB, effectsB):

                    eff = to_sp(
                        tensor(
                            effA1,
                            effA2,
                            effB1,
                            effB2
                        )
                    )

                    st = to_sp(
                        tensor(
                            stA1,
                            stA2,
                            stB1,
                            stB2
                        )
                    )

                    out += eval_effect(eff, X) * st

    return sp.expand(out)




    
    
##############################################################################
# Unknown coefficients (10 variables)
##############################################################################

a = {}

for i in range(4):
    for j in range(i,4):

        a[(i,j)] = sp.symbols(f"a{i+1}{j+1}")
    
    
    
 ##############################################################################
# Two-input broadcaster
##############################################################################

def B(rho,sigma):

    out = sp.zeros(81,1)

    for i in range(4):
        for j in range(i,4):

            coeff = a[(i,j)]

            if i==j:

                out += coeff * to_sp(
                    tensor(
                        N[i](rho),
                        N[j](sigma)
                    )
                )

            else:

                out += coeff * (
                    to_sp(
                        tensor(
                            N[i](rho),
                            N[j](sigma)
                        )
                    )
                    +
                    to_sp(
                        tensor(
                            N[j](rho),
                            N[i](sigma)
                        )
                    )
                )

    return sp.expand(out)   
    


##############################################################################
# Basis for A⊗B
##############################################################################

basis2 = []

for x in basis:
    for y in basis:

        basis2.append(
            to_sp(
                tensor(x,y)
            )
        )    
        


##############################################################################
# Broadcasting constraint
##############################################################################




eqs = []

for X in basis2:

    #
    # Recover rho,sigma from basis element
    #

    rho   = PTrX(X, 2, [3,3])
    sigma = PTrX(X, 1, [3,3])

    #
    # Broadcast map
    #

    Y = B(rho, sigma)

    
        
        
## Classical consistency: adjacent adjacent 
       
for rho in adj_states:
    for sigma in adj_states:

        Y = B(rho, sigma)

        lhs = DDDD(
            Y,
            adj_states, adj_effects,
            adj_states, adj_effects
        )

        rhs = BB_cl(
            to_sp(tensor(rho, sigma)),
            adj_states, adj_effects,
            adj_states, adj_effects
        )

        for k in range(81):
            eqs.append(
                sp.expand(lhs[k] - rhs[k])
            )     


# ##############################################################################
# # Classical consistency: adjacent + diagonal
# ##############################################################################

# for rho in adj_states:
#     for sigma in diag_states:

#         Y = B(rho, sigma)

#         lhs = DDDD(
#             Y,
#             adj_states,  adj_effects,
#             diag_states, diag_effects
#         )

#         rhs = BB_cl(
#             to_sp(
#                 tensor(rho, sigma)
#             ),
#             adj_states,  adj_effects,
#             diag_states, diag_effects
#         )

#         for k in range(81):

#             eqs.append(
#                 sp.expand(lhs[k] - rhs[k])
#             )



# ##############################################################################
# # Classical consistency: diagonal + adjacent
# ##############################################################################

# for rho in diag_states:
#     for sigma in adj_states:

#         Y = B(rho, sigma)

#         lhs = DDDD(
#             Y,
#             diag_states, diag_effects,
#             adj_states,  adj_effects
#         )

#         rhs = BB_cl(
#             to_sp(
#                 tensor(rho, sigma)
#             ),
#             diag_states, diag_effects,
#             adj_states,  adj_effects
#         )

#         for k in range(81):
#             eqs.append(
#                 sp.expand(lhs[k] - rhs[k])
#             )

# # ##############################################################################
# # # Classical consistency: diagonal + diagonal
# # ##############################################################################


# for rho in diag_states:
#     for sigma in diag_states:

#         Y = B(rho, sigma)

#         lhs = DDDD(
#             Y,
#             diag_states, diag_effects,
#             diag_states, diag_effects
#         )

#         rhs = BB_cl(
#             to_sp(
#                 tensor(rho, sigma)
#             ),
#             diag_states, diag_effects,
#             diag_states, diag_effects
#         )

#         for k in range(81):
#             eqs.append(
#                 sp.expand(lhs[k] - rhs[k])
#             )










##############################################################################
# Identify independent equations from the original system
##############################################################################
variables = list(a.values())

#
# Convert equations to A*x = b
#
A, b = sp.linear_eq_to_matrix(
    eqs,
    variables
)


RREF_A, pivots_A = A.rref()

print("\nPivot rows / independent original equations:")

#
# Find pivot rows by looking at the transpose.
# The pivot columns of A.T correspond to independent rows of A.
#
_, independent_row_indices = A.T.rref()

print("Independent row indices:", independent_row_indices)

print("\n" + "="*80)
print("INDEPENDENT ORIGINAL EQUATIONS")
print("="*80)

for n, idx in enumerate(independent_row_indices, start=1):

    lhs = sp.expand(
        sum(
            A[idx, j] * variables[j]
            for j in range(len(variables))
        )
    )

    rhs = b[idx]

    print(f"\nEq {n}:")
    sp.pprint(sp.Eq(lhs, rhs))



##############################################################################
# Solve
##############################################################################

sol = sp.solve(
    eqs,
    variables,
    dict=True
)

print("\nSolution:")
print(sol)
