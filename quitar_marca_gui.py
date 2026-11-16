#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quitar_marca_gui.py - Interfaz grafica para quitar_marca.py

Uso: doble clic. Arrastra (o elige) una foto o video, dibuja con el mouse un
rectangulo sobre la marca y presiona "Quitar marca". La pestana
"Que detecto" muestra exactamente que pixeles se van a retocar.

Requiere quitar_marca.py en la misma carpeta.
Opcional: pip install tkinterdnd2  (habilita arrastrar archivos a la ventana)
"""

import argparse
import base64
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

import quitar_marca as qm

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAY_ARRASTRE = True
except Exception:
    HAY_ARRASTRE = False

ANCHO_VISTA = 820
ALTO_VISTA = 460
ANCHO_DETALLE = 560
CONTEXTO = 30  # pixeles de entorno que se muestran alrededor de la zona


def a_foto_tk(imagen_bgr):
    """Convierte una imagen de OpenCV en un PhotoImage de Tk (sin Pillow)."""
    ok, buffer = cv2.imencode(".png", imagen_bgr)
    if not ok:
        raise IOError("No se pudo preparar la imagen para mostrarla.")
    return tk.PhotoImage(data=base64.b64encode(buffer.tobytes()))


def es_video(ruta):
    return os.path.splitext(ruta)[1].lower() in qm.EXT_VIDEO


def es_soportado(ruta):
    ext = os.path.splitext(ruta)[1].lower()
    return ext in qm.EXT_VIDEO or ext in qm.EXT_IMAGEN


def leer_primer_cuadro(ruta):
    if es_video(ruta):
        captura = cv2.VideoCapture(ruta)
        ok, cuadro = captura.read()
        captura.release()
        return cuadro if ok else None
    return qm.leer_imagen(ruta)


class Aplicacion(object):

    def __init__(self, raiz):
        self.raiz = raiz
        self.ruta = None
        self.cuadro = None        # primer cuadro a resolucion original
        self.escala = 1.0         # vista / original
        self.zona = None          # (x, y, w, h) en pixeles del original
        self.mascara = None       # mascara detectada para self.zona
        self.clave_mascara = None
        self.ocupado = False
        self.cola = queue.Queue()
        self.fotos = {}           # referencias para que Tk no borre las imagenes
        self.inicio_arrastre = None

        raiz.title("Quitar marca de agua")
        raiz.minsize(880, 680)

        self.expandir = tk.IntVar(value=3)
        self.grano = tk.BooleanVar(value=True)
        self.aislar = tk.BooleanVar(value=True)
        self.estado = tk.StringVar(value="Elige o arrastra una foto o video para empezar.")

        self.pestanas = ttk.Notebook(raiz)
        self.pestanas.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        self.tab_limpiar = ttk.Frame(self.pestanas, padding=8)
        self.tab_detecto = ttk.Frame(self.pestanas, padding=8)
        self.pestanas.add(self.tab_limpiar, text="  Limpiar  ")
        self.pestanas.add(self.tab_detecto, text="  Qué detectó  ")
        self.pestanas.bind("<<NotebookTabChanged>>", self.al_cambiar_pestana)

        self.armar_tab_limpiar()
        self.armar_tab_detecto()

        pie = ttk.Frame(raiz, padding=(8, 6))
        pie.pack(fill="x")
        self.barra = ttk.Progressbar(pie, mode="determinate", maximum=100)
        self.barra.pack(fill="x")
        ttk.Label(pie, textvariable=self.estado, anchor="w").pack(fill="x", pady=(4, 0))

        if HAY_ARRASTRE:
            raiz.drop_target_register(DND_FILES)
            raiz.dnd_bind("<<Drop>>", self.al_soltar)

        raiz.after(100, self.revisar_cola)

    # ------------------------------------------------------- interfaz

    def armar_tab_limpiar(self):
        marco = self.tab_limpiar
        arriba = ttk.Frame(marco)
        arriba.pack(fill="x")
        ttk.Button(arriba, text="Elegir archivo...", command=self.elegir).pack(side="left")
        self.etiqueta_archivo = ttk.Label(arriba, text="(ningún archivo)")
        self.etiqueta_archivo.pack(side="left", padx=10)

        self.lienzo = tk.Canvas(marco, width=ANCHO_VISTA, height=ALTO_VISTA,
                                bg="#1e1e1e", highlightthickness=1,
                                highlightbackground="#555555", cursor="crosshair")
        self.lienzo.pack(pady=8)
        self.lienzo.bind("<ButtonPress-1>", self.al_presionar)
        self.lienzo.bind("<B1-Motion>", self.al_mover)
        self.lienzo.bind("<ButtonRelease-1>", self.al_soltar_mouse)
        self.dibujar_bienvenida()

        opciones = ttk.Frame(marco)
        opciones.pack(fill="x")
        ttk.Checkbutton(opciones, text="Retocar solo la marca (videos)",
                        variable=self.aislar, command=self.invalidar_mascara).pack(side="left")
        ttk.Checkbutton(opciones, text="Agregar grano al relleno",
                        variable=self.grano).pack(side="left", padx=14)
        ttk.Label(opciones, text="Margen extra (px):").pack(side="left")
        ttk.Spinbox(opciones, from_=0, to=15, width=4, textvariable=self.expandir,
                    command=self.invalidar_mascara).pack(side="left", padx=4)

        botones = ttk.Frame(marco)
        botones.pack(fill="x", pady=(10, 0))
        self.boton_ver = ttk.Button(botones, text="Ver qué detectó", command=self.ver_deteccion)
        self.boton_ver.pack(side="left")
        self.boton_quitar = ttk.Button(botones, text="Quitar marca", command=self.quitar)
        self.boton_quitar.pack(side="right")

    def armar_tab_detecto(self):
        marco = self.tab_detecto
        self.resumen = ttk.Label(marco, anchor="w", justify="left", wraplength=820,
                                 text="Carga un archivo y marca la zona para ver aquí qué se detectó.")
        self.resumen.pack(fill="x")
        self.paneles = {}
        titulos = [
            ("original", "Original"),
            ("detectado", "Lo que detectó (en rojo: lo único que se va a retocar)"),
            ("resultado", "Vista previa del resultado"),
        ]
        for clave, titulo in titulos:
            ttk.Label(marco, text=titulo, font=("", 9, "bold")).pack(anchor="w", pady=(10, 2))
            etiqueta = ttk.Label(marco)
            etiqueta.pack(anchor="w")
            self.paneles[clave] = etiqueta

    def dibujar_bienvenida(self):
        self.lienzo.delete("all")
        texto = "Haz clic en «Elegir archivo...»"
        if HAY_ARRASTRE:
            texto = "Arrastra aquí una foto o video\no haz clic en «Elegir archivo...»"
        self.lienzo.create_text(ANCHO_VISTA // 2, ALTO_VISTA // 2, text=texto,
                                fill="#bbbbbb", font=("", 14), justify="center")

    # --------------------------------------------------------- carga

    def elegir(self):
        if self.ocupado:
            return
        patrones = " ".join("*" + e for e in sorted(qm.EXT_IMAGEN | qm.EXT_VIDEO))
        ruta = filedialog.askopenfilename(
            title="Elige una foto o video",
            filetypes=[("Fotos y videos", patrones), ("Todos los archivos", "*.*")],
        )
        if ruta:
            self.cargar(ruta)

    def al_soltar(self, evento):
        if self.ocupado:
            return
        rutas = self.raiz.tk.splitlist(evento.data)
        if rutas:
            self.cargar(rutas[0])

    def cargar(self, ruta):
        if not es_soportado(ruta):
            messagebox.showwarning("Formato no soportado",
                                   "Ese tipo de archivo no es una foto ni un video soportado.")
            return
        cuadro = leer_primer_cuadro(ruta)
        if cuadro is None:
            messagebox.showerror("No se pudo abrir", "No se pudo leer el archivo:\n" + ruta)
            return
        self.ruta = ruta
        self.cuadro = cuadro
        self.zona = None
        self.invalidar_mascara()
        alto, ancho = cuadro.shape[:2]
        self.escala = min(ANCHO_VISTA / float(ancho), ALTO_VISTA / float(alto), 1.0)
        vista = cuadro
        if self.escala < 1.0:
            vista = cv2.resize(cuadro, (int(ancho * self.escala), int(alto * self.escala)),
                               interpolation=cv2.INTER_AREA)
        self.fotos["vista"] = a_foto_tk(vista)
        self.lienzo.delete("all")
        self.lienzo.config(width=vista.shape[1], height=vista.shape[0])
        self.lienzo.create_image(0, 0, image=self.fotos["vista"], anchor="nw")
        self.etiqueta_archivo.config(text=os.path.basename(ruta))
        self.barra["value"] = 0
        self.estado.set("Dibuja con el mouse un rectángulo sobre la marca de agua.")
        self.pestanas.select(self.tab_limpiar)

    # ---------------------------------------------- seleccion de zona

    def al_presionar(self, evento):
        if self.cuadro is None or self.ocupado:
            return
        self.inicio_arrastre = (evento.x, evento.y)
        self.lienzo.delete("zona")

    def al_mover(self, evento):
        if self.inicio_arrastre is None:
            return
        x0, y0 = self.inicio_arrastre
        self.lienzo.delete("zona")
        self.lienzo.create_rectangle(x0, y0, evento.x, evento.y, outline="#ff3030",
                                     width=2, tags="zona")

    def al_soltar_mouse(self, evento):
        if self.inicio_arrastre is None:
            return
        x0, y0 = self.inicio_arrastre
        self.inicio_arrastre = None
        self.fijar_zona_vista(x0, y0, evento.x, evento.y)

    def fijar_zona_vista(self, x0, y0, x1, y1):
        """Recibe un rectangulo en coordenadas de la vista y lo pasa al original."""
        alto, ancho = self.cuadro.shape[:2]
        izq, der = sorted((x0, x1))
        arr, aba = sorted((y0, y1))
        x = int(izq / self.escala)
        y = int(arr / self.escala)
        w = int((der - izq) / self.escala)
        h = int((aba - arr) / self.escala)
        if w < 6 or h < 6:
            self.zona = None
            self.lienzo.delete("zona")
            self.estado.set("La zona es demasiado pequeña. Dibuja un rectángulo sobre la marca.")
            return
        self.zona = qm.ajustar_zona((x, y, w, h), ancho, alto)
        self.invalidar_mascara()
        zx, zy, zw, zh = self.zona
        self.lienzo.delete("zona")
        self.lienzo.create_rectangle(zx * self.escala, zy * self.escala,
                                     (zx + zw) * self.escala, (zy + zh) * self.escala,
                                     outline="#ff3030", width=2, tags="zona")
        self.estado.set("Zona marcada (%d x %d px). Presiona «Ver qué detectó» o «Quitar marca»."
                        % (zw, zh))

    def invalidar_mascara(self):
        self.mascara = None
        self.clave_mascara = None

    # ------------------------------------------------------ deteccion

    def clave_actual(self):
        return (self.ruta, self.zona, int(self.expandir.get()), bool(self.aislar.get()))

    def listo_para_trabajar(self):
        if self.ocupado:
            return False
        if self.cuadro is None:
            messagebox.showinfo("Falta el archivo", "Primero elige una foto o video.")
            return False
        if self.zona is None:
            messagebox.showinfo("Falta la zona",
                                "Dibuja con el mouse un rectángulo sobre la marca de agua.")
            return False
        return True

    def al_cambiar_pestana(self, evento):
        en_detecto = self.pestanas.select() == str(self.tab_detecto)
        if not en_detecto or self.ocupado:
            return
        if self.cuadro is None or self.zona is None:
            return
        if self.clave_mascara != self.clave_actual():
            self.ver_deteccion()

    def ver_deteccion(self):
        if not self.listo_para_trabajar():
            return
        self.bloquear(True, "Analizando el video para aislar la marca...")
        self.barra.config(mode="indeterminate")
        self.barra.start(12)
        clave = self.clave_actual()
        hilo = threading.Thread(target=self.trabajo_detectar, args=(clave,), daemon=True)
        hilo.start()

    def calcular_mascara(self, clave):
        ruta, zona, expandir, aislar = clave
        if aislar and es_video(ruta):
            return qm.detectar_mascara_video(ruta, zona, expandir=expandir)
        return None

    def trabajo_detectar(self, clave):
        try:
            mascara = self.calcular_mascara(clave)
            self.cola.put(("detectado", clave, mascara))
        except BaseException as error:
            self.cola.put(("error", str(error) or repr(error)))

    def mostrar_deteccion(self, clave, mascara):
        self.mascara = mascara
        self.clave_mascara = clave
        ruta, zona, expandir, aislar = clave
        x, y, w, h = zona
        alto, ancho = self.cuadro.shape[:2]
        x0, y0 = max(0, x - CONTEXTO), max(0, y - CONTEXTO)
        x1, y1 = min(ancho, x + w + CONTEXTO), min(alto, y + h + CONTEXTO)

        completa = np.zeros((alto, ancho), dtype=np.uint8)
        if mascara is None:
            completa[y:y + h, x:x + w] = 255
        else:
            completa[y:y + h, x:x + w] = mascara

        original = self.cuadro[y0:y1, x0:x1].copy()
        detectado = original.copy()
        dentro = completa[y0:y1, x0:x1] > 0
        rojo = np.zeros_like(detectado)
        rojo[:, :] = (0, 0, 255)
        detectado[dentro] = (0.35 * detectado[dentro] + 0.65 * rojo[dentro]).astype(np.uint8)
        cv2.rectangle(detectado, (x - x0, y - y0), (x - x0 + w - 1, y - y0 + h - 1),
                      (0, 255, 255), 1)
        limpio = qm.limpiar_cuadro(self.cuadro.copy(), zona, 5, "telea",
                                   mascara, bool(self.grano.get()))
        resultado = limpio[y0:y1, x0:x1]

        factor = min(ANCHO_DETALLE / float(original.shape[1]), 150.0 / float(original.shape[0]), 4.0)
        factor = max(factor, 0.2)
        imagenes = {"original": original, "detectado": detectado, "resultado": resultado}
        for nombre in imagenes:
            imagen = imagenes[nombre]
            tamano = (max(1, int(imagen.shape[1] * factor)), max(1, int(imagen.shape[0] * factor)))
            interp = cv2.INTER_NEAREST if factor > 1.5 else cv2.INTER_AREA
            self.fotos[nombre] = a_foto_tk(cv2.resize(imagen, tamano, interpolation=interp))
            self.paneles[nombre].config(image=self.fotos[nombre])

        cobertura = 100.0 * float(dentro.sum()) / float(w * h)
        if mascara is not None:
            texto = ("Se aisló la marca: solo se retoca el %.0f%% de la zona que marcaste. "
                     "El resto del rectángulo queda intacto." % cobertura)
        elif not es_video(ruta):
            texto = ("Es una foto: no hay otros cuadros para comparar, así que se rellena "
                     "todo el rectángulo. Ajústalo lo más posible a la marca.")
        elif not aislar:
            texto = "La opción «Retocar solo la marca» está apagada: se rellena todo el rectángulo."
        else:
            texto = ("No se pudo aislar la marca (el fondo casi no cambia detrás de ella o "
                     "el video es muy corto). Se rellena todo el rectángulo.")
        self.resumen.config(text=texto)
        self.pestanas.select(self.tab_detecto)
        self.estado.set("Detección lista. Si se ve bien, presiona «Quitar marca».")

    # --------------------------------------------------------- quitar

    def quitar(self):
        if not self.listo_para_trabajar():
            return
        self.bloquear(True, "Procesando...")
        self.barra.config(mode="determinate")
        self.barra["value"] = 0
        x, y, w, h = self.zona
        args = argparse.Namespace(
            seleccionar=False, zona="%d,%d,%d,%d" % (x, y, w, h),
            esquina="inf-der", ancho=20.0, alto=10.0, margen=0,
            radio=5, metodo="telea", crf=18, vista_previa=False,
            rectangulo=not bool(self.aislar.get()),
            expandir=int(self.expandir.get()),
            guardar_mascara=False, sin_grano=not bool(self.grano.get()),
        )
        ruta = self.ruta
        salida = qm.ruta_salida(ruta, None, es_video(ruta))
        hilo = threading.Thread(target=self.trabajo_quitar, args=(ruta, salida, args), daemon=True)
        hilo.start()

    def trabajo_quitar(self, ruta, salida, args):
        try:
            if es_video(ruta):
                ok = qm.procesar_video(ruta, salida, args, self.avisar_progreso)
            else:
                ok = qm.procesar_imagen(ruta, salida, args)
            if ok:
                self.cola.put(("terminado", salida))
            else:
                self.cola.put(("error", "No se pudo procesar el archivo."))
        except BaseException as error:
            self.cola.put(("error", str(error) or repr(error)))

    def avisar_progreso(self, n, total):
        if n % 10 == 0 or n == total:
            self.cola.put(("progreso", n, total))

    # ---------------------------------------------------------- comun

    def bloquear(self, ocupado, mensaje=None):
        self.ocupado = ocupado
        estado = "disabled" if ocupado else "normal"
        self.boton_ver.config(state=estado)
        self.boton_quitar.config(state=estado)
        if mensaje:
            self.estado.set(mensaje)

    def detener_barra(self):
        self.barra.stop()
        self.barra.config(mode="determinate")

    def revisar_cola(self):
        try:
            while True:
                mensaje = self.cola.get_nowait()
                tipo = mensaje[0]
                if tipo == "progreso":
                    n, total = mensaje[1], mensaje[2]
                    if total > 0:
                        self.barra["value"] = 100.0 * n / total
                        self.estado.set("Procesando... %d de %d cuadros" % (n, total))
                elif tipo == "detectado":
                    self.detener_barra()
                    self.bloquear(False)
                    if mensaje[1] == self.clave_actual():
                        self.mostrar_deteccion(mensaje[1], mensaje[2])
                elif tipo == "terminado":
                    self.detener_barra()
                    self.barra["value"] = 100
                    self.bloquear(False, "Listo: " + mensaje[1])
                    messagebox.showinfo("Listo", "Se guardó el archivo limpio en:\n" + mensaje[1])
                elif tipo == "error":
                    self.detener_barra()
                    self.barra["value"] = 0
                    self.bloquear(False, "Ocurrió un error.")
                    messagebox.showerror("Error", mensaje[1])
        except queue.Empty:
            pass
        self.raiz.after(100, self.revisar_cola)


def main():
    if HAY_ARRASTRE:
        raiz = TkinterDnD.Tk()
    else:
        raiz = tk.Tk()
    Aplicacion(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
