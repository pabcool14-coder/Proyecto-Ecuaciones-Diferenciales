import sys

import numpy as np
PASO = 0.1


def paso_heun(f, t, estado, h):
    # Heun predice el siguiente valor y promedia la pendiente inicial y la del punto estimado.
    pendiente_inicial = f(t, estado)
    prediccion = estado + h * pendiente_inicial
    pendiente_final = f(t + h, prediccion)
    return estado + h * (pendiente_inicial + pendiente_final) / 2


def paso_rk4(f, t, estado, h):
    # RK4 combina cuatro pendientes y da el doble de peso a las dos del punto medio.
    k1 = f(t, estado)
    k2 = f(t + h / 2, estado + h * k1 / 2)
    k3 = f(t + h / 2, estado + h * k2 / 2)
    k4 = f(t + h, estado + h * k3)
    return estado + h * (k1 + 2*k2 + 2*k3 + k4) / 6


def resolver(f, puntos, inicial, metodo):
    # Parte de la condición inicial y guarda el resultado de cada paso.
    valores = np.empty((len(puntos), len(inicial)))
    valores[0] = inicial
    for i in range(len(puntos) - 1):
        h = puntos[i + 1] - puntos[i]
        valores[i + 1] = metodo(f, puntos[i], valores[i], h)
    return valores


# Ecuaciones de primer orden y sus soluciones analíticas.
def ed1(t, estado):
    y = estado[0]
    return np.array([(y**3 - 2*t*y) / t**2])


def exacta1(t):
    return [np.sqrt(5*t / (3*t**5 + 2))]


def ed2(x, estado):
    y = estado[0]
    return np.array([1 + (x - 1) * np.exp(-y)])


def exacta2(x):
    return [np.log(np.exp(x) - x)]


# En las ecuaciones de segundo orden, v representa la derivada de y. Se calculan y y v en cada paso.
def ed3(x, estado):
    y, v = estado
    return np.array([v, (x*np.exp(-1.5*x) - 12*v - 9*y) / 4])


def exacta3(x):
    polinomio = 1 + 1.5*x + x**3/24
    factor = np.exp(-1.5*x)
    y = factor * polinomio
    v = factor * (1.5 + x**2/8 - 1.5*polinomio)
    return [y, v]


def ed4(x, estado):
    y, v = estado
    return np.array([v, (np.log(x) - x**2*v - 4*x*y) / x**3])


def exacta4(x):
    s = np.log(x)
    a, b = 23/25, 97/50
    y = a*np.cos(2*s) + b*np.sin(2*s) + (5*s + 2)/(25*x)
    v = (-2*a*np.sin(2*s) + 2*b*np.cos(2*s))/x + (3 - 5*s)/(25*x**2)
    return [y, v]


# Cada sistema lineal define las derivadas de x y de y, junto con una función para su solución exacta.
def lineal1(t, estado):
    x, y = estado
    return np.array([-2*x + y, x - 2*y])


def exacta_lineal1(t):
    a, b = np.exp(-t), np.exp(-3*t)
    return [(a + b)/2, (a - b)/2]


def lineal2(t, estado):
    x, y = estado
    return np.array([-2*x, 2*x - 3*y])


def exacta_lineal2(t):
    x = np.exp(-2*t)
    return [x, 2*(x - np.exp(-3*t))]


# Sistema de Van der Pol con μ = 1. La primera derivada de x es y.
def van_der_pol(t, estado):
    x, y = estado
    return np.array([y, (1 - x**2)*y - x])


# Cada ejercicio tiene su ecuación, intervalo, condiciones iniciales y variables.
PROBLEMAS = [
    ("ED 1 · Bernoulli", ed1, exacta1, (1, 3), [1], "t", ["y"]),
    ("ED 2 · Sustitución", ed2, exacta2, (0, 2), [0], "x", ["y"]),
    ("ED 3 · Ecuación forzada", ed3, exacta3, (0, 2), [1, 0], "x", ["y", "y'"]),
    ("ED 4 · Cauchy–Euler", ed4, exacta4, (1, 3), [1, 4], "x", ["y", "y'"]),
    ("Sistema lineal 1", lineal1, exacta_lineal1, (0, 3), [1, 0], "t", ["x", "y"]),
    ("Sistema lineal 2", lineal2, exacta_lineal2, (0, 3), [1, 0], "t", ["x", "y"]),
    ("Van der Pol", van_der_pol, None, (0, 15), [2, 0], "t", ["x", "y"]),
]
METODOS = [("Heun", paso_heun), ("RK4", paso_rk4)]


def comparar_exacta(problema, puntos):
    nombre, f, exacta, intervalo, inicial, eje, variables = problema
    # Calcula la solución exacta y las dos aproximaciones en los mismos puntos.
    referencia = np.array([exacta(t) for t in puntos])
    heun = resolver(f, puntos, inicial, paso_heun)
    rk4 = resolver(f, puntos, inicial, paso_rk4)
    columnas, datos = [eje], [puntos]
    resumen = []
    for j, variable in enumerate(variables):
        real = referencia[:, j]
        error_heun = np.abs(heun[:, j] - real)
        error_rk4 = np.abs(rk4[:, j] - real)
        columnas.extend([f"{variable} exacta", f"{variable} Heun", f"{variable} RK4",
                         f"error Heun {variable}", f"error RK4 {variable}"])
        datos.extend([real, heun[:, j], rk4[:, j], error_heun, error_rk4])

        print(f"  {variable} al final: exacta {real[-1]:.8f} | "
              f"Heun {heun[-1, j]:.8f} | RK4 {rk4[-1, j]:.8f}")
        for metodo, error in [("Heun", error_heun), ("RK4", error_rk4)]:
            # Calcula el RMSE y el error máximo a partir del primer paso.
            rmse = np.sqrt(np.mean(error[1:]**2))
            maximo = np.max(error[1:])
            resumen.append((variable, metodo, rmse, maximo))
            print(f"    {metodo:4}  RMSE = {rmse:.4e}   error máximo = {maximo:.4e}")

    return columnas, np.column_stack(datos), resumen


def comparar_van_der_pol(problema, puntos):
    nombre, f, _, (inicio, fin), inicial, eje, variables = problema
    pasos = len(puntos) - 1
    h = (fin - inicio) / pasos
    columnas, datos = [eje], [puntos]
    resumen = []
    for nombre_metodo, metodo in METODOS:
        soluciones = []
        # Resuelve el sistema con tres tamaños de paso: h, h/2 y h/4.
        for factor in [1, 2, 4]:
            malla = np.linspace(inicio, fin, pasos*factor + 1)
            valores = resolver(f, malla, inicial, metodo)
            # Conserva los valores que corresponden a los mismos tiempos.
            soluciones.append(valores[::factor])
        for j, variable in enumerate(variables):
            for factor, valores in zip([1, 2, 4], soluciones):
                etiqueta = f"h = {h/factor:g}"
                columnas.append(f"{nombre_metodo} {variable} {etiqueta}")
                datos.append(valores[:, j])
            print(f"  {nombre_metodo}, {variable}({fin:g}) con h/4 = {soluciones[2][-1, j]:.8f}")
            # Compara h con h/2 y luego h/2 con h/4 para cada variable.
            for i in range(2):
                mayor, menor = h / 2**i, h / 2**(i + 1)
                diferencia = np.abs(soluciones[i][:, j] - soluciones[i + 1][:, j])
                columnas.append(f"diferencia {nombre_metodo} {variable} {mayor:g} vs {menor:g}")
                datos.append(diferencia)
                rmse = np.sqrt(np.mean(diferencia[1:]**2))
                maximo = np.max(diferencia[1:])
                resumen.append((nombre_metodo, variable, mayor, menor, rmse, maximo))
                print(f"    pasos {mayor:g} y {menor:g}: diferencia máxima = {maximo:.4e}")
    return columnas, np.column_stack(datos), resumen


def main():
    # Muestra los siete ejercicios en la consola y abre la ventana con la opción elegida.
    print("Proyecto de Ecuaciones Diferenciales · Heun y RK4")
    for numero, problema in enumerate(PROBLEMAS, start=1):
        print(f"{numero}. {problema[0]}")
    opcion = input("Elige un problema (1–7) o escribe 0 para ejecutar todos: ").strip()
    if opcion not in ("0", "1", "2", "3", "4", "5", "6", "7"):
        print("Opción inválida. Ejecuta de nuevo y elige un número del 0 al 7.")
        return
    if not np.isfinite(PASO) or PASO <= 0:
        print("Elige un PASO positivo y finito al inicio del archivo.")
        return
    abrir_ventana(int(opcion))


ECUACIONES = [
    "t² y′ + 2t y − y³ = 0",
    "y′ = (x + eʸ − 1)e⁻ʸ",
    "4y″ + 12y′ + 9y = x exp(−3x/2)",
    "x³ y″ + x² y′ + 4xy = ln(x)",
    "x′ = −2x + y       y′ = x − 2y",
    "x′ = −2x                y′ = 2x − 3y",
    "x′ = y                    y′ = (1 − x²)y − x       (μ = 1)",
]


def dibujar_curvas(lienzo, puntos, series, eje, unidad):
    # Ajusta la gráfica al tamaño de la ventana y dibuja las curvas en una misma escala.
    lienzo.delete("all")
    ancho, alto = lienzo.winfo_width(), lienzo.winfo_height()
    if ancho < 150 or alto < 95:
        return
    izquierda, derecha, arriba, abajo = 76, ancho - 33, 28, alto - 46
    minimo = min(np.min(y) for _, y in series)
    maximo = max(np.max(y) for _, y in series)
    margen = (maximo - minimo)*0.06 if maximo > minimo else max(abs(maximo)*0.05, 0.01)
    minimo, maximo = minimo - margen, maximo + margen
    x = izquierda + (puntos - puntos[0])/(puntos[-1] - puntos[0])*(derecha - izquierda)
    for i in range(5):
        fraccion = i/4
        py = abajo - fraccion*(abajo - arriba)
        px = izquierda + fraccion*(derecha - izquierda)
        lienzo.create_line(izquierda, py, derecha, py, fill="#dae3e9")
        lienzo.create_text(izquierda-8, py, anchor="e", text=f"{minimo + fraccion*(maximo-minimo):.3g}")
        lienzo.create_text(px, abajo+12, text=f"{puntos[0] + fraccion*(puntos[-1]-puntos[0]):.4g}")
    lienzo.create_text(izquierda, 10, text=unidad, anchor="w")
    lienzo.create_text((izquierda+derecha)/2, alto-9, text=eje)
    colores = ["#163d51", "#c55a18", "#087b84"]
    trazos = [(), (7, 4), (2, 3)]
    for i, (_, valores) in enumerate(series):
        y = abajo - (valores-minimo)/(maximo-minimo)*(abajo-arriba)
        coordenadas = np.column_stack((x, y)).ravel().tolist()
        lienzo.create_line(coordenadas, fill=colores[i], dash=trazos[i], width=2)


def abrir_ventana(opcion=1):
    import tkinter as tk
    from tkinter import ttk, messagebox

    raiz = tk.Tk()
    raiz.title("Ecuaciones diferenciales")
    raiz.geometry(f"{min(1020, raiz.winfo_screenwidth()-40)}x{min(710, raiz.winfo_screenheight()-85)}")
    raiz.minsize(720, 500)
    estilo = ttk.Style(raiz)
    estilo.theme_use("clam")
    estilo.configure("Treeview", rowheight=25, background="white", fieldbackground="white")
    estilo.configure("Treeview.Heading", background="#e4edf1", foreground="#163d51")
    estilo.configure("Principal.TButton", background="#126875", foreground="white", padding=(14, 6))
    estilo.map("Principal.TButton", background=[("active", "#105563")])
    base = ttk.Frame(raiz, padding=14)
    base.pack(fill="both", expand=True)
    ttk.Label(base, text="Heun y RK4", font=("TkDefaultFont", 17, "bold"),
              foreground="#163d51").pack(anchor="w")

    # Controles para elegir el ejercicio y el tamaño de paso.
    controles = ttk.Frame(base)
    controles.pack(fill="x", pady=(12, 7))
    ttk.Label(controles, text="Problema").pack(side="left")
    lista = ttk.Combobox(controles, values=[p[0] for p in PROBLEMAS], width=25, state="readonly")
    lista.pack(side="left", padx=(6, 18))
    lista.current(max(0, opcion - 1))
    ttk.Label(controles, text="Paso h").pack(side="left")
    paso = tk.StringVar(value=str(PASO))
    entrada = ttk.Entry(controles, textvariable=paso, width=9)
    entrada.pack(side="left", padx=6)
    boton = ttk.Button(controles, text="Calcular", style="Principal.TButton")
    boton.pack(side="left", padx=5)
    todos = ttk.Button(controles, text="Calcular los 7")
    todos.pack(side="left", padx=3)

    enunciado = tk.StringVar()
    ttk.Label(base, textvariable=enunciado, foreground="#18465a",
              font=("TkDefaultFont", 11), wraplength=760).pack(anchor="w", pady=(1, 9))
    resultados = ttk.Frame(base)
    resultados.pack(fill="both", expand=True)
    resultados.columnconfigure(0, weight=1, uniform="panel")
    resultados.columnconfigure(1, weight=1, uniform="panel")
    resultados.rowconfigure(0, weight=1)
    detalles = ttk.LabelFrame(resultados, text="Tabla", padding=5)
    grafica = ttk.LabelFrame(resultados, text="Gráfica", padding=5)
    detalles.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
    grafica.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

    def crear_tabla(padre, alto):
        caja = ttk.Frame(padre)
        caja.pack(fill="both", expand=True)
        arbol = ttk.Treeview(caja, show="headings", height=alto)
        vertical = ttk.Scrollbar(caja, orient="vertical", command=arbol.yview)
        horizontal = ttk.Scrollbar(caja, orient="horizontal", command=arbol.xview)
        arbol.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        arbol.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        caja.rowconfigure(0, weight=1)
        caja.columnconfigure(0, weight=1)
        return arbol

    tabla = crear_tabla(detalles, 6)
    filtros = ttk.Frame(grafica)
    filtros.pack(fill="x")
    ttk.Label(filtros, text="Mostrar").pack(side="left")
    vista = ttk.Combobox(filtros, state="readonly", width=30)
    vista.pack(side="left", padx=6)
    leyenda = ttk.Label(grafica)
    leyenda.pack(anchor="w", pady=(4, 0))
    lienzo = tk.Canvas(grafica, width=220, height=145, bg="white", highlightthickness=0)
    lienzo.pack(fill="both", expand=True, pady=(7, 0))

    titulo_resumen = ttk.Label(base, text="Errores", font=("TkDefaultFont", 11, "bold"))
    titulo_resumen.pack(anchor="w", pady=(9, 3))
    caja_resumen = ttk.Frame(base)
    caja_resumen.pack(fill="x")
    resumen_tabla = crear_tabla(caja_resumen, 2)
    pie = ttk.Frame(base)
    pie.pack(fill="x", pady=(8, 0))
    estado = tk.StringVar()
    ttk.Label(pie, textvariable=estado, wraplength=530).pack(side="left")
    calculados = {}
    datos_grafica = []
    puntos_grafica = []
    posicion = tk.IntVar(value=0)
    registros = []
    navegador = ttk.Frame(detalles)
    navegador.pack(fill="x")
    pagina_texto = ttk.Label(navegador)
    pagina_texto.pack(side="left")

    def llenar(tabla_destino, cabeceras, filas):
        tabla_destino.delete(*tabla_destino.get_children())
        ids = [str(i) for i in range(len(cabeceras))]
        tabla_destino.configure(columns=ids)
        for clave, texto in zip(ids, cabeceras):
            tabla_destino.heading(clave, text=texto)
            tabla_destino.column(clave, width=max(94, len(texto)*8+20), anchor="e")
        for fila in filas:
            tabla_destino.insert("", "end", values=fila)

    def mostrar_pagina(desplazamiento=0):
        # Muestra hasta 100 filas por página.
        inicio = posicion.get() + desplazamiento
        posicion.set(inicio)
        tabla.delete(*tabla.get_children())
        for fila in registros[inicio:inicio+100]:
            tabla.insert("", "end", values=[f"{v:.9g}" for v in fila])
        total = len(registros)
        pagina_texto.configure(text=f"Filas {inicio+1}–{min(total,inicio+100)} de {total}" if total else "")
        anterior.configure(state="normal" if inicio else "disabled")
        siguiente.configure(state="normal" if inicio+100 < total else "disabled")
    siguiente = ttk.Button(navegador, text="Siguiente", command=lambda: mostrar_pagina(100))
    siguiente.pack(side="right")
    anterior = ttk.Button(navegador, text="Anterior", command=lambda: mostrar_pagina(-100))
    anterior.pack(side="right")

    def actualizar_grafica(evento=None):
        indice = vista.current()
        if indice >= 0 and datos_grafica:
            nombre, series, unidad = datos_grafica[indice]
            leyenda.configure(text="   ".join(f"{marca} {etiqueta}" for marca, (etiqueta, _) in
                                             zip(["━", "┄", "···"], series)))
            dibujar_curvas(lienzo, puntos_grafica, series, PROBLEMAS[lista.current()][5], unidad)

    def limpiar(*args):
        nonlocal registros
        i = lista.current()
        _, _, _, (a, b), inicial, eje, variables = PROBLEMAS[i]
        condiciones = ",   ".join(f"{v}({a}) = {valor}" for v, valor in zip(variables, inicial))
        enunciado.set(f"{ECUACIONES[i]}\n{condiciones}       ·       {a} ≤ {eje} ≤ {b}")
        datos_grafica.clear()
        registros = []
        posicion.set(0)
        mostrar_pagina()
        tabla.configure(columns=())
        resumen_tabla.delete(*resumen_tabla.get_children())
        resumen_tabla.configure(columns=())
        titulo_resumen.configure(text="Convergencia" if i == 6 else "Errores")
        vista.configure(values=[])
        vista.set("")
        leyenda.configure(text="")
        lienzo.delete("all")
        estado.set("")

    def calcular(evento=None):
        nonlocal registros, puntos_grafica
        i = lista.current()
        problema = PROBLEMAS[i]
        _, _, exacta, (a, b), _, eje, variables = problema
        try:
            h = float(paso.get().strip().replace(",", "."))
            if not np.isfinite(h) or h <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Revisa el paso", "Ingresa un paso mayor que 0.", parent=raiz)
            return
        if h < (b-a)/20000:
            messagebox.showerror("Paso mínimo", f"Usa h ≥ {(b-a)/20000:g}.", parent=raiz)
            return
        # Divide el intervalo en pasos iguales hasta llegar al punto final.
        n = max(1, int(np.ceil((b-a)/h)))
        puntos = np.linspace(a, b, n+1)
        raiz.configure(cursor="watch")
        raiz.update_idletasks()
        try:
            # Usa la solución exacta cuando está disponible. Para Van der Pol compara los tres pasos.
            comparar = comparar_exacta if exacta is not None else comparar_van_der_pol
            if i not in calculados:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    columnas, datos, resumen = comparar(problema, puntos)
                calculados[i] = columnas, datos, resumen
            columnas, datos, resumen = calculados[i]
        except FloatingPointError:
            messagebox.showerror("Revisa el paso", "Prueba con un paso menor.", parent=raiz)
            return
        finally:
            raiz.configure(cursor="")
        indice_vista = vista.current()
        registros, puntos_grafica = datos, puntos
        llenar(tabla, columnas, [])
        posicion.set(0)
        mostrar_pagina()
        cabeceras = ["Variable", "Método", "RMSE", "Error máximo"]
        if exacta is None:
            cabeceras = ["Método", "Variable", "Paso mayor", "Paso menor", "RMSE diferencia", "Dif. máxima"]
        filas = []
        for fila in resumen:
            filas.append([*fila[:2], *(f"{v:.7g}" for v in fila[2:])])
        resumen_tabla.configure(height=min(4, len(filas)))
        llenar(resumen_tabla, cabeceras, filas)
        datos_grafica.clear()
        k = 1
        # Prepara las gráficas de cada variable y sus errores.
        if exacta is not None:
            for v in variables:
                series = [(metodo, datos[:, k+j]) for j, metodo in enumerate(["Exacta", "Heun", "RK4"])]
                datos_grafica.append((f"Solución · {v}", series, v))
                errores = [("Heun", datos[:, k+3]), ("RK4", datos[:, k+4])]
                datos_grafica.append((f"Error absoluto · {v}", errores, "Error absoluto"))
                k += 5
        else:
            # Agrupa las curvas de Van der Pol por método y variable, junto con sus diferencias.
            for metodo, _ in METODOS:
                for v in variables:
                    series = [(f"h = {(b-a)/(n*factor):g}", datos[:, k+j]) for j, factor in enumerate([1, 2, 4])]
                    datos_grafica.append((f"{metodo} · {v}", series, v))
                    diferencias = [("h y h/2", datos[:, k+3]), ("h/2 y h/4", datos[:, k+4])]
                    datos_grafica.append((f"Diferencias · {metodo} · {v}", diferencias, "Diferencia absoluta"))
                    k += 5
        vista.configure(values=[opcion[0] for opcion in datos_grafica])
        vista.current(indice_vista if 0 <= indice_vista < len(datos_grafica) else 0)
        actualizar_grafica()
        estado.set(f"{n} pasos  ·  h = {(b-a)/n:.8g}")

        return True

    def calcular_todos():
        # Calcula los siete ejercicios y deja el seleccionado para el final.
        elegido = lista.current()
        orden = [i for i in range(len(PROBLEMAS)) if i != elegido] + [elegido]
        for i in orden:
            lista.current(i)
            limpiar()
            estado.set(f"Calculando {PROBLEMAS[i][0]}…")
            raiz.update_idletasks()
            if not calcular():
                return
        estado.set("7 problemas calculados.")

    todos.configure(command=calcular_todos)
    boton.configure(command=calcular)
    entrada.bind("<Return>", calcular)
    def elegir_problema(evento):
        limpiar()
        calcular()

    def cambiar_paso(*args):
        calculados.clear()
        if len(puntos_grafica) > 1:
            h_visible = puntos_grafica[1] - puntos_grafica[0]
            estado.set(f"Gráfica con h = {h_visible:.8g} · Pulsa Calcular para actualizar.")

    lista.bind("<<ComboboxSelected>>", elegir_problema)
    paso.trace_add("write", cambiar_paso)
    vista.bind("<<ComboboxSelected>>", actualizar_grafica)
    lienzo.bind("<Configure>", actualizar_grafica)
    limpiar()
    raiz.after_idle(calcular_todos if opcion == 0 else calcular)
    raiz.mainloop()


if __name__ == "__main__":
    if "--consola" in sys.argv:
        main()
    else:
        abrir_ventana()
