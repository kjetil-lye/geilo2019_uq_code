import sympy

def p_exact():
    h_0, x_0, v_0, alpha, g = sympy.symbols('h x v a g')

    return h_0, x_0, v_0, alpha, g, sympy.cos(alpha)*v_0*(v_0*sympy.sin(alpha)+ sympy.sqrt((v_0*sympy.sin(alpha))**2+2*g*h_0))/g +x_0

def integral_mean_p(x_0, v_0, alpha, g):
    h, x, v, a, g_, p = p_exact()
    return sympy.integrate(p, (h, 0, 1)).subs([(x, x_0), (v, v_0), (a, alpha), (g_, g)])


def integral_variance_p(x_0, v_0, alpha, g):
    h, x, v, a, g_, p = p_exact()
    mean = integral_mean_p(x_0, v_0, alpha, g)
    return sympy.integrate((p-mean)**2, (h, 0, 1)).subs([(x, x_0), (v, v_0), (a, alpha), (g_, g)])
