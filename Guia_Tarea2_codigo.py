"""
Guía de aprendizaje – Tarea 2: Miscelánea de ejercicios
Código Python consolidado de los ejercicios 1 al 20.

Requisitos sugeridos:
    pip install numpy matplotlib

Ejecuta el archivo desde una carpeta con permisos de escritura.
Las gráficas se muestran con Matplotlib; puedes guardarlas con
plt.savefig(...) antes de plt.show() si deseas almacenarlas.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Para ejecución reproducible y visualización ordenada.
np.set_printoptions(precision=4, suppress=True)


# ================================================================
# EJERCICIO 1: Operaciones con matrices
# ================================================================
import numpy as np

A = np.array([[2, 3], [1, 4]])
B = np.array([[1, 2], [3, 1]])
print("A+B =\n", A + B)
print("2A-3B =\n", 2*A - 3*B)
print("AB =\n", A @ B)
det_A = np.linalg.det(A)
print("det(A) =", det_A)
print("Invertible:", not np.isclose(det_A, 0))
if not np.isclose(det_A, 0):
    print("A^(-1) =\n", np.linalg.inv(A))


# ================================================================
# EJERCICIO 2: Sistema de ecuaciones, notación matricial y regla de Cramer
# ================================================================
import numpy as np

M = np.array([[2, 1, -1], [3, -2, 1], [1, 4, -3]], dtype=float)
b = np.array([5, -4, 3], dtype=float)
print("det(M) =", np.linalg.det(M))
print("rangos:", np.linalg.matrix_rank(M), np.linalg.matrix_rank(np.c_[M, b]))
if np.isclose(np.linalg.det(M), 0):
    if np.linalg.matrix_rank(M) < np.linalg.matrix_rank(np.c_[M, b]):
        print("Sistema incompatible: no tiene solución")
    else:
        print("Sistema compatible indeterminado")
else:
    print(np.linalg.solve(M, b))


# ================================================================
# EJERCICIO 3: Valores y vectores propios
# ================================================================
import numpy as np

C = np.array([[4, 1], [2, 3]], dtype=float)
valores, vectores = np.linalg.eig(C)
for valor, vector in zip(valores, vectores.T):
    print("valor propio:", valor)
    print("vector propio:", vector)


# ================================================================
# EJERCICIO 4: Función lineal
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: 2*x + 3
for x in [2, -1, 0]:
    print(f"f({x}) = {f(x)}")
x = np.linspace(-3, 3, 200)
plt.plot(x, f(x), label="f(x)=2x+3")
plt.scatter([2, -1, 0], [f(2), f(-1), f(0)], color="red")
plt.axhline(0, color="black", linewidth=.7); plt.axvline(0, color="black", linewidth=.7)
plt.xlabel("x"); plt.ylabel("f(x)"); plt.legend(); plt.grid(); plt.show()


# ================================================================
# EJERCICIO 5: Función cuadrática
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

g = lambda x: x**2 - 4*x + 3
raices = np.roots([1, -4, 3])
xv = -(-4)/(2*1); yv = g(xv)
print("Raíces:", raices)
print("Cortes: eje x", [(r, 0) for r in raices], "; eje y", (0, g(0)))
print("Vértice:", (xv, yv), "; abre hacia arriba")
x = np.linspace(-1, 5, 300)
plt.plot(x, g(x), label="g(x)=x²−4x+3")
plt.scatter(raices, [0, 0], color="red"); plt.scatter([xv], [yv], color="purple")
plt.axhline(0, color="black", linewidth=.7); plt.axvline(0, color="black", linewidth=.7)
plt.legend(); plt.grid(); plt.show()


# ================================================================
# EJERCICIO 6: Ecuación no lineal
# ================================================================
import numpy as np

coeficientes = [1, -6, 11, -6]
soluciones = np.roots(coeficientes)
print("Soluciones:", np.sort(soluciones))


# ================================================================
# EJERCICIO 7: Medidas de tendencia central
# ================================================================
import numpy as np
from collections import Counter

edades = np.array([25, 30, 29, 32, 25, 40, 28, 33, 35, 29])
frecuencias = Counter(edades)
moda = [x for x, f in frecuencias.items() if f == max(frecuencias.values())]
print("Media:", edades.mean())
print("Mediana:", np.median(edades))
print("Moda:", moda)


# ================================================================
# EJERCICIO 8: Medidas de dispersión
# ================================================================
import numpy as np

edades = np.array([25, 30, 29, 32, 25, 40, 28, 33, 35, 29])
print("Rango:", np.ptp(edades))
print("Varianza poblacional:", np.var(edades))
print("Desviación estándar poblacional:", np.std(edades))
print("Varianza muestral:", np.var(edades, ddof=1))
print("Desviación estándar muestral:", np.std(edades, ddof=1))


# ================================================================
# EJERCICIO 9: Visualización de ventas mensuales
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

ventas = np.array([200, 250, 300, 350, 400, 450])
meses = ["M1", "M2", "M3", "M4", "M5", "M6"]
plt.bar(meses, ventas); plt.xlabel("Mes"); plt.ylabel("Ventas")
plt.title("Ventas mensuales"); plt.show()
plt.hist(ventas, bins=[175, 275, 375, 475], edgecolor="black")
plt.xlabel("Ventas"); plt.ylabel("Frecuencia"); plt.title("Histograma"); plt.show()


# ================================================================
# EJERCICIO 10: Clasificación de clientes
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

clientes = {"A": (500, 200), "B": (300, 100), "C": (700, 400)}
X = np.array(list(clientes.values()), dtype=float)
# Clasificación simple por gasto total: usamos la mediana como umbral.
# Si el gasto coincide con la mediana, se clasifica como gasto alto.
umbral = np.median(X.sum(axis=1))
grupos = {nombre: ("bajo" if sum(p) < umbral else "alto")
          for nombre, p in clientes.items()}
print("Umbral (mediana):", umbral, "Grupos:", grupos)
for nombre, (a, o) in clientes.items():
    plt.scatter(a, o, s=90); plt.annotate(nombre, (a, o))
plt.xlabel("Alimentación"); plt.ylabel("Ocio"); plt.grid(); plt.show()


# ================================================================
# EJERCICIO 11: Optimización de inventarios
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

A = np.linspace(0, 100, 400)
B_presupuesto = (5000 - 50*A) / 30
B_capacidad = 150 - A
plt.plot(A, B_presupuesto, label="50A+30B=5000")
plt.plot(A, B_capacidad, label="A+B=150")
plt.fill_between(A, 0, np.minimum(B_presupuesto, B_capacidad), alpha=.25)
plt.xlim(0, 105); plt.ylim(0, 160)
plt.xlabel("Unidades de A"); plt.ylabel("Unidades de B")
plt.legend(); plt.grid(); plt.show()
print("Vértices: (0,0), (0,150), (25,125), (100,0)")


# ================================================================
# EJERCICIO 12: Media móvil de tres días
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

visitas = np.array([500, 520, 550, 530, 600, 620, 610, 650, 680, 700])
media_movil = np.convolve(visitas, np.ones(3)/3, mode="valid")
print("Media móvil:", np.round(media_movil, 2))
dias = np.arange(1, 11)
plt.plot(dias, visitas, "o-", label="Original")
plt.plot(dias[2:], media_movil, "s-", label="MA(3)")
plt.xlabel("Día"); plt.ylabel("Visitas"); plt.legend(); plt.grid(); plt.show()


# ================================================================
# EJERCICIO 13: Segmentación con K-means
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1000,500], [1500,800], [800,300], [1200,700], [500,200]], float)
centroides = np.array([[650,300], [1350,750]], float)
for _ in range(20):
    etiquetas = np.argmin(((X[:, None, :] - centroides[None, :, :])**2).sum(axis=2), axis=1)
    nuevos = np.array([X[etiquetas == k].mean(axis=0) for k in range(2)])
    if np.allclose(nuevos, centroides): break
    centroides = nuevos
print("Etiquetas:", etiquetas + 1)
print("Centroides:\n", centroides)
for k in range(2):
    plt.scatter(X[etiquetas == k, 0], X[etiquetas == k, 1], label=f"Grupo {k+1}")
plt.scatter(centroides[:,0], centroides[:,1], marker="X", c="black", s=140)
plt.xlabel("Categoría 1"); plt.ylabel("Categoría 2"); plt.legend(); plt.grid(); plt.show()


# ================================================================
# EJERCICIO 14: Regresión lineal
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

proyectos = np.array([1,2,3,4,5,6])
ingresos = np.array([50,60,70,80,90,100])
pendiente, intercepto = np.polyfit(proyectos, ingresos, 1)
print(f"Ecuación: y = {pendiente:.2f}x + {intercepto:.2f}")
print("Predicción para 8 proyectos:", pendiente*8 + intercepto)
plt.scatter(proyectos, ingresos)
plt.plot(proyectos, pendiente*proyectos + intercepto, label="Recta ajustada")
plt.xlabel("Proyectos"); plt.ylabel("Ingresos (miles USD)"); plt.legend(); plt.grid(); plt.show()


# ================================================================
# EJERCICIO 15: Correlación de Pearson
# ================================================================
import numpy as np

inversion = np.array([10,20,30,40,50])
ventas = np.array([100,150,200,250,300])
r = np.corrcoef(inversion, ventas)[0, 1]
print("Coeficiente de Pearson:", r)


# ================================================================
# EJERCICIO 16: Inventario por categorías y coeficiente de variación
# ================================================================
import numpy as np

categorias = {
    "Electrónicos": [50,60,70,80,90],
    "Ropa": [100,90,110,120,100],
    "Alimentos": [200,250,300,220,270]
}
for nombre, datos in categorias.items():
    x = np.array(datos, dtype=float)
    media, desviacion = x.mean(), x.std()
    cv = 100 * desviacion / media
    print(nombre, "media=", media, "desv.=", desviacion, "CV%=", cv)


# ================================================================
# EJERCICIO 17: Calificaciones
# ================================================================
import numpy as np

notas = np.array([85,90,88,92,87,91,89,93], dtype=float)
print("Media:", notas.mean())
print("Varianza poblacional:", notas.var())
print("Desviación estándar:", notas.std())


# ================================================================
# EJERCICIO 18: Comparación de dos equipos
# ================================================================
import numpy as np

equipos = {"A": [75,80,85,70,90], "B": [78,82,76,85,89]}
for nombre, datos in equipos.items():
    x = np.array(datos, dtype=float)
    print(nombre, "media=", x.mean(), "varianza=", x.var(), "desv.=", x.std())


# ================================================================
# EJERCICIO 19: Consumo de agua: medidas muestrales
# ================================================================
import numpy as np

agua = np.array([12,15,14,10,13,11], dtype=float)
print("Media:", agua.mean())
print("Varianza muestral:", agua.var(ddof=1))
print("Desviación estándar muestral:", agua.std(ddof=1))


# ================================================================
# EJERCICIO 20: Valores de una recta
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

x = np.array([-2,-1,0,1,2,3])
y = 2*x + 3
print(list(zip(x, y)))
print("Pendiente:", 2, "Intersección con eje y:", 3)
plt.plot(x, y, "o-", label="y=2x+3")
plt.axhline(0, color="black", linewidth=.7); plt.axvline(0, color="black", linewidth=.7)
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.grid(); plt.show()
