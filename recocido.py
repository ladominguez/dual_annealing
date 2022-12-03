import masw
import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams

filename    = 'SampleData.dat'
HeaderLines = 7
fs          = 1000 # Hz
N           = 24
x1          = 10   # m
dx          = 1    # m
direction   = 'forward';
Amasw       = masw.masw(filename, 1/fs, fs, N, dx, x1, direction, header=6 )

lambda_curve0 = np.array([28.44, 25.30909091, 22, 18.92307692, 17.05714286, 14.96, 13.2, 11.50588235, 10.06666667, 8.52631579, 7.5, 6.74285714, 6.27272727,
                      5.68695652, 5.2, 4.848, 4.52307692, 4.13333333, 3.9, 3.72413793, 3.52, 3.36774194, 3.225, 3.09090909, 2.96470588, 2.84571429,
                      2.73333333, 2.62702703, 2.49473684, 2.4, 2.31, 2.25365854, 2.2, 2.12093023, 2.07272727, 2, 1.93043478, 1.86382979, 1.825, 1.7877551])
c_curve0 = np.array([237, 232, 220, 205, 199, 187, 176, 163, 151, 135, 125, 118, 115, 109, 104, 101, 98, 93, 91, 90, 88, 87, 86, 85, 84, 83, 82, 81, 79,
                 78, 77, 77, 77, 76, 76, 75, 74, 73, 73, 73])
c_test_min = 50 #m/s
c_test_max = 500 #m/+s
delta_c_test = 0.5 #m/s
c_test = np.arange(c_test_min, c_test_max + 1, delta_c_test) #m/s
c_test = c_test[:-1]

#Parámetros de la capa
n = 6
alpha = [1440, 1440, 1440, 1440, 1440, 1440, 1440] #m/s
h = [1, 1, 2, 2, 4, 5, np.Inf] #m
beta = [75, 90, 150, 180, 240, 290, 290] #m/s
rho = [1850, 1850, 1850, 1850, 1850, 1850, 1850] #kg/m^3

up_low_boundary = 'yes'
FigWidth = 8 #cm
FigHeight = 10 #cm
FigFontSize = 8 #pt

c_t, lambda_t = Amasw.theoretical_dispersion_curve(c_test, h, alpha, beta, rho, n, lambda_curve0)

fig, ax = plt.subplots()
obs, = ax.plot(c_curve0, lambda_curve0, 'ko-', markersize = 3, markerfacecolor = 'k')
obs.set_label('Exp.')
calc, = ax.plot(c_t, lambda_t,'r+--', markersize = 10, markerfacecolor = 'r')
calc.set_label('Theor.')
ax.legend(loc = 'lower left', fontsize = FigFontSize)

#Etiquetas de eje y límites de eje
ax.set_xlabel('Velocidad de la onda de Rayleigh [m/s]', fontsize = FigFontSize, fontweight = 'normal')
ax.set_ylabel('Longitud de onda [m]', fontsize = FigFontSize, fontweight = 'normal')
    

#Tamaño de la figura
ax.grid()
rcParams['figure.figsize'] = 2, 2
fig.set_figheight(FigHeight)
fig.set_figwidth(FigWidth)
plt.gca().invert_yaxis()

def error(c_t, lambda_t):
    e = np.sqrt(np.sum((c_curve0 - c_t)**2) / len(c_curve0)) #Error de la Velocidad de fase
    #e2 = np.sqrt(np.sum((lambda_curve0 - lambda_t)**2) / lambda_curve0) #Error de la Longitud de Onda
    print(e)
    return e
    
# algoritmo de recocido simulado
def recocido_simulado(n_iteraciones, limites, n_pasos, temp, c_test, h, alpha, beta, rho, n):
    #funcion teórica inicial
    c_t, lambda_t = Amasw.theoretical_dispersion_curve(c_test, h, alpha, beta, rho, n, lambda_curve0)
     
    # evaluar el punto inicial para la Velocidad de Onda
    mejor_eval = error(c_t, lambda_t)
     
    vel_esp = np.array(h + beta)
    # solución de trabajo actual
    curr, curr_eval = vel_esp, mejor_eval
    marcador = list()

    # ejecutar el algoritmo
    for i in range(n_iteraciones):
        # Da un paso
        candidato = curr + np.random.randn(len(limites)) * n_pasos
        #evalua en el modelo
        c_t, lambda_t = Amasw.theoretical_dispersion_curve(c_test, candidato[0:7], alpha, candidato[7:14], rho, n, lambda_curve0)
        # evaluar punto candidato
        candidato_eval = error(c_t, lambda_t)

        # comprobar si hay una nueva mejor solución
        if candidato_eval < mejor_eval:
            # almacenar nuevo mejor punto
            mejor, mejor_eval = candidato, candidato_eval
            # realizar un seguimiento de las puntuaciones
            marcador.append(mejor_eval)
            
        # diferencia entre evaluación de puntos de candidato y actual
        diff = candidato_eval - curr_eval
        # calcular la temperatura para el punto actual
        t = temp / float(i + 1)
        # calcular el criterio de aceptación de la metrópoli
        metropolis = np.exp(-diff / t)
        
        # comprobar si debemos mantener el nuevo punto
        if diff < 0 or np.random.rand() < metropolis:
            # almacenar el nuevo punto actual
            curr, curr_eval = candidato, candidato_eval

    return mejor, mejor_eval, marcador

#Aplicando el método de inversión por recocido simulado

c_test_min = 50 #m/s
c_test_max = 500 #m/+s
delta_c_test = 0.5 #m/s
c_test = np.arange(c_test_min, c_test_max + 1, delta_c_test) #m/s
c_test = c_test[:-1]

#Parámetros de la capa
n = 6
alpha = [1440, 1440, 1440, 1440, 1440, 1440, 1440] #m/s
h = [1, 1, 2, 2, 4, 5, np.Inf] #m
beta = [75, 90, 150, 180, 240, 290, 290] #m/s
rho = [1850, 1850, 1850, 1850, 1850, 1850, 1850] #kg/m^3

#------------------------------------------------------------------
#-----------------Empieza el Recocido Simulado---------------------
#------------------------------------------------------------------
        
# sembrar el generador de números pseudoaleatorios
np.random.seed(1)
# definir el rango de entrada
limites = np.asarray([[-5.0, 5.0]])
# definir las iteraciones totales
n_iteraciones = 50
# definir el tamaño de paso máximo
n_pasos = 0.1
# temperatura inicial
temp = 10

# realizar la búsqueda de recocido simulado para la velocidad de onda de corte
mejor, puntuaje, puntuaje2 = recocido_simulado(n_iteraciones, limites, n_pasos, temp, c_test, h, alpha, beta, rho, n)
h = mejor[0:7] #Nuevo espesor de capa
beta = mejor[7:14] #Nueva velocidad de la onda de corte'

#Última evaluación de la curva teórica
c_t, lambda_t = Amasw.theoretical_dispersion_curve(c_test, h, alpha, beta, rho, n, lambda_curve0)
        
#------------------------------------------------------------------
#-----------------Termina el Recocido Simulado---------------------
#------------------------------------------------------------------

#Visualizar los resultados
FigWidth = 8 #cm
FigHeight = 10 #cm
FigFontSize = 8 #pt

fig, ax = plt.subplots()
obs, = ax.plot(c_curve0, lambda_curve0, 'ko-', markersize = 3, markerfacecolor = 'k')
obs.set_label('Exp.')
calc, = ax.plot(c_t, lambda_t,'r+--', markersize = 10, markerfacecolor = 'r')
calc.set_label('Theor.')
ax.legend(loc = 'lower left', fontsize = FigFontSize)

#Etiquetas de eje y límites de eje
ax.set_xlabel('Velocidad de la onda de Rayleigh [m/s]', fontsize = FigFontSize, fontweight = 'normal')
ax.set_ylabel('Longitud de onda [m]', fontsize = FigFontSize, fontweight = 'normal')

#Tamaño de la figura
ax.grid()
rcParams['figure.figsize'] = 2, 2
fig.set_figheight(FigHeight)
fig.set_figwidth(FigWidth)
plt.gca().invert_yaxis()


#Calcule el vector de profundidad z
z = np.zeros(n+1);
for i in range(n):
    z[i+1] = np.sum(h[0:i])

#Graficar el perfil de velocidad de la onda de corte
fig2, ax2 = plt.subplots()
for i in range(n):
    ax2.plot(np.array([beta[i], beta[i]]), np.array([z[i], z[i+1]]), 'k', markersize = 3, markerfacecolor = 'k')
    ax2.plot(np.array([beta[i], beta[i+1]]), np.array([z[i+1], z[i+1]]), 'k', markersize = 3, markerfacecolor = 'k')
ax2.plot(np.array([beta[n], beta[n]]), np.array([z[n], z[n]+5]), 'k', markersize = 3, markerfacecolor = 'k')

#Establecer los ejes y los límites de los ejes
ax2.set_xlabel('Velocidad de onda de corte [m/s]', fontsize = FigFontSize)
ax2.set_ylabel('Espesor [m]', fontsize = FigFontSize)

#Tamaño de la figura
ax.grid()
rcParams['figure.figsize'] = 2, 2
fig2.set_figheight(FigHeight)
fig2.set_figwidth(FigWidth)