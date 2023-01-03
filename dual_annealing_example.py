import numpy as np
from scipy.optimize import dual_annealing

N = 201
x = np.linspace(-1,1,N)

def objetivo(parametros):  # parametros = f(h, vel)
    observados = 6 + np.exp(-7*x) + 0.1*np.random.randn(len(x)) # curva de dispersion f(lambda_curve0, c_curve0) x-> lambda_curve0 
    teoricos = parametros[0] + np.exp(-parametros[1]*x)        # theoretical_dispersion_curve(c_t, lambda_t)
    return np.sqrt(np.sum((teoricos-observados)**2/len(teoricos)))

if __name__ == '__main__':
    a0 = 3
    b0 = 4
    lw = [a0-10, b0-10]
    up = [a0+10, b0+10]

    ret = dual_annealing(objetivo, bounds=list(zip(lw, up)))
    print(ret)