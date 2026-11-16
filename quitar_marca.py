#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quitar_marca.py - Quita una marca de agua fija (logo, texto, sello de fecha)
de fotos y videos rellenando la zona con inpainting de OpenCV.

Requisitos:
    pip install opencv-python numpy
    FFmpeg instalado y en el PATH (solo para video; conserva el audio)

Ejemplos:
    # Elegir la zona con el mouse (abre una ventana)
    python quitar_marca.py foto.jpg --seleccionar

    # Esquina inferior derecha, 20% del ancho y 10% del alto
    python quitar_marca.py video.mp4 --esquina inf-der --ancho 20 --alto 10

    # Zona exacta en pixeles: x,y,ancho,alto
    python quitar_marca.py video.mp4 --zona 1600,980,300,80

    # Procesar una carpeta completa con la misma zona
    python quitar_marca.py ./mis_videos --esquina sup-izq

    # Ver la zona marcada en rojo antes de procesar
    python quitar_marca.py foto.jpg --esquina inf-der --vista-previa
"""

import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXT_VIDEO = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
ESQUINAS = ["sup-izq", "sup-der", "inf-izq", "inf-der"]


# ---------------------------------------------------------------- zona

def zona_desde_esquina(esquina, ancho_img, alto_img, pct_ancho, pct_alto, margen):
    """Devuelve (x, y, w, h) para una esquina dada en porcentajes."""
    w = int(ancho_img * pct_ancho / 100.0)
    h = int(alto_img * pct_alto / 100.0)
    x = margen if esquina.endswith("izq") else ancho_img - w - margen
    y = margen if esquina.startswith("sup") else alto_img - h - margen
    return (x, y, w, h)


def zona_interactiva(imagen):
    """Abre una ventana para dibujar el rectangulo con el mouse."""
    alto, ancho = imagen.shape[:2]
    escala = min(1.0, 1280.0 / ancho, 720.0 / alto)
    vista = imagen
    if escala < 1.0:
        vista = cv2.resize(imagen, (int(ancho * escala), int(alto * escala)))
    titulo = "Dibuja la zona y presiona ENTER (C para cancelar)"
    x, y, w, h = cv2.selectROI(titulo, vista, showCrosshair=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        raise ValueError("No se selecciono ninguna zona.")
    return (int(x / escala), int(y / escala), int(w / escala), int(h / escala))


def ajustar_zona(zona, ancho_img, alto_img):
    """Recorta la zona para que quede dentro de la imagen."""
    x, y, w, h = zona
    x = max(0, min(x, ancho_img - 1))
    y = max(0, min(y, alto_img - 1))
    w = max(1, min(w, ancho_img - x))
    h = max(1, min(h, alto_img - y))
    return (x, y, w, h)


def resolver_zona(args, primer_cuadro):
    alto, ancho = primer_cuadro.shape[:2]
    if args.seleccionar:
        zona = zona_interactiva(primer_cuadro)
    elif args.zona:
        try:
            partes = [int(v) for v in args.zona.split(",")]
        except ValueError:
            sys.exit("--zona debe tener el formato x,y,ancho,alto (enteros).")
        if len(partes) != 4:
            sys.exit("--zona debe tener el formato x,y,ancho,alto.")
        zona = tuple(partes)
    else:
        zona = zona_desde_esquina(
            args.esquina, ancho, alto, args.ancho, args.alto, args.margen
        )
    return ajustar_zona(zona, ancho, alto)


def crear_mascara(ancho_img, alto_img, zona):
    x, y, w, h = zona
    mascara = np.zeros((alto_img, ancho_img), dtype=np.uint8)
    mascara[y:y + h, x:x + w] = 255
    return mascara


# ---------------------------------------------------------- inpainting

def limpiar_cuadro(cuadro, zona, radio, metodo, mascara_zona=None, grano=True):
    """
    Aplica inpainting solo sobre un recorte alrededor de la zona.
    - mascara_zona: mascara del tamano de la zona con SOLO los pixeles de la
      marca (255). Si es None se rellena el rectangulo completo.
    - grano: agrega al relleno el mismo ruido/grano que tiene el entorno,
      para que no se vea como un parche liso y borroso.
    """
    alto, ancho = cuadro.shape[:2]
    x, y, w, h = zona
    borde = max(16, radio * 4)
    x0, y0 = max(0, x - borde), max(0, y - borde)
    x1, y1 = min(ancho, x + w + borde), min(alto, y + h + borde)

    recorte = cuadro[y0:y1, x0:x1]
    mascara = np.zeros(recorte.shape[:2], dtype=np.uint8)
    if mascara_zona is None:
        mascara[y - y0:y - y0 + h, x - x0:x - x0 + w] = 255
    else:
        mascara[y - y0:y - y0 + h, x - x0:x - x0 + w] = mascara_zona

    bandera = cv2.INPAINT_NS if metodo == "ns" else cv2.INPAINT_TELEA
    relleno = cv2.inpaint(recorte, mascara, radio, bandera)

    if grano:
        relleno = agregar_grano(recorte, relleno, mascara)

    cuadro[y0:y1, x0:x1] = relleno
    return cuadro


def agregar_grano(original, relleno, mascara):
    """Mide el ruido fino alrededor de la marca y lo replica en el relleno."""
    dentro = mascara > 0
    fuera = ~dentro
    if dentro.sum() == 0 or fuera.sum() < 50:
        return relleno
    orig_f = original.astype(np.float32)
    residuo = orig_f - cv2.GaussianBlur(orig_f, (0, 0), 1.5)
    # Desviacion robusta (MAD) para no contar bordes reales como ruido
    valores = residuo[fuera]
    sigma = 1.4826 * np.median(np.abs(valores - np.median(valores)))
    if sigma < 0.3:
        return relleno
    sigma = min(float(sigma), 12.0)
    ruido = np.random.normal(0.0, sigma, relleno.shape).astype(np.float32)
    salida = relleno.astype(np.float32)
    salida[dentro] += ruido[dentro]
    return np.clip(salida, 0, 255).astype(np.uint8)


def detectar_mascara_video(ruta, zona, muestras=40, expandir=3):
    """
    Encuentra los pixeles exactos de una marca FIJA dentro de la zona.
    El fondo cambia de un cuadro a otro pero la marca no: sus bordes aparecen
    en el mismo lugar en todos los cuadros. La mediana de los bordes a lo
    largo del video deja solo la marca.
    Devuelve una mascara (h, w) o None si no se pudo aislar con confianza.
    """
    x, y, w, h = zona
    captura = cv2.VideoCapture(ruta)
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 8:
        captura.release()
        return None
    indices = np.linspace(0, total - 1, min(muestras, total)).astype(int)
    bordes = []
    for i in indices:
        captura.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, cuadro = captura.read()
        if not ok:
            continue
        gris = cv2.cvtColor(cuadro[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
        bordes.append(cv2.magnitude(gx, gy))
    captura.release()
    if len(bordes) < 8:
        return None

    mediana = np.median(np.stack(bordes), axis=0)
    norma = cv2.normalize(mediana, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binaria = cv2.threshold(norma, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Unir trazos y rellenar el interior de letras/logos
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cerrada = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, nucleo, iterations=2)
    contornos, _ = cv2.findContours(cerrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mascara, contornos, -1, 255, thickness=cv2.FILLED)
    if expandir > 0:
        k = 2 * expandir + 1
        mascara = cv2.dilate(mascara, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    cobertura = float((mascara > 0).mean())
    if cobertura < 0.005 or cobertura > 0.75:
        return None
    return mascara


# -------------------------------------------------------------- imagen

def leer_imagen(ruta):
    # imdecode + fromfile soporta rutas con tildes/enie en Windows
    datos = np.fromfile(ruta, dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def guardar_imagen(ruta, imagen):
    ext = os.path.splitext(ruta)[1]
    parametros = []
    if ext.lower() in (".jpg", ".jpeg"):
        parametros = [cv2.IMWRITE_JPEG_QUALITY, 95]
    ok, buffer = cv2.imencode(ext, imagen, parametros)
    if not ok:
        raise IOError("No se pudo codificar " + ruta)
    buffer.tofile(ruta)


def procesar_imagen(entrada, salida, args):
    imagen = leer_imagen(entrada)
    if imagen is None:
        print("  [x] No se pudo leer la imagen: " + entrada)
        return False
    zona = resolver_zona(args, imagen)

    if args.vista_previa:
        guardar_vista_previa(imagen, zona, salida)
        return True

    resultado = limpiar_cuadro(imagen, zona, args.radio, args.metodo,
                               None, not args.sin_grano)
    guardar_imagen(salida, resultado)
    print("  [ok] " + salida + "  (zona x,y,w,h = " + str(zona) + ")")
    return True


def guardar_vista_previa(imagen, zona, salida):
    x, y, w, h = zona
    vista = imagen.copy()
    cv2.rectangle(vista, (x, y), (x + w, y + h), (0, 0, 255), 3)
    ruta = os.path.splitext(salida)[0] + "_vista_previa.jpg"
    guardar_imagen(ruta, vista)
    print("  [vista previa] " + ruta + "  (zona x,y,w,h = " + str(zona) + ")")


# --------------------------------------------------------------- video

def buscar_ffmpeg():
    """
    Busca FFmpeg sin depender del PATH, en este orden:
    1) junto al programa (ffmpeg.exe en la misma carpeta)
    2) el binario incluido en el paquete imageio-ffmpeg (va dentro del .exe)
    3) el PATH del sistema
    """
    if getattr(sys, "frozen", False):
        carpeta = os.path.dirname(sys.executable)
    else:
        carpeta = os.path.dirname(os.path.abspath(__file__))
    for nombre in ("ffmpeg.exe", "ffmpeg"):
        candidato = os.path.join(carpeta, nombre)
        if os.path.isfile(candidato):
            return candidato
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def procesar_video(entrada, salida, args, progreso=None):
    captura = cv2.VideoCapture(entrada)
    if not captura.isOpened():
        print("  [x] No se pudo abrir el video: " + entrada)
        return False

    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, primer = captura.read()
    if not ok:
        print("  [x] El video no tiene cuadros legibles: " + entrada)
        captura.release()
        return False
    zona = resolver_zona(args, primer)

    if args.vista_previa:
        guardar_vista_previa(primer, zona, salida)
        captura.release()
        return True

    mascara_zona = None
    if not args.rectangulo:
        mascara_zona = detectar_mascara_video(entrada, zona, expandir=args.expandir)
        if mascara_zona is None:
            print("  (no se pudo aislar la marca; se rellena el rectangulo completo)")
        else:
            pct = 100.0 * float((mascara_zona > 0).mean())
            print("  marca aislada: se retoca solo el %.0f%% de la zona" % pct)
            if args.guardar_mascara:
                guardar_imagen(os.path.splitext(salida)[0] + "_mascara.png", mascara_zona)

    ffmpeg = buscar_ffmpeg()
    if ffmpeg is None:
        captura.release()
        sys.exit(
            "No se encontro FFmpeg. Coloca ffmpeg.exe en la misma carpeta "
            "que este programa o instala el paquete imageio-ffmpeg."
        )

    # FFmpeg recibe los cuadros limpios por stdin y copia el audio del original
    comando = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", str(ancho) + "x" + str(alto), "-r", str(fps),
        "-i", "pipe:0",
        "-i", entrada,
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-shortest",
        salida,
    ]
    # CREATE_NO_WINDOW evita que parpadee una consola negra en Windows
    proceso = subprocess.Popen(
        comando, stdin=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    cuadro = primer
    n = 0
    try:
        while ok:
            limpio = limpiar_cuadro(cuadro, zona, args.radio, args.metodo,
                                    mascara_zona, not args.sin_grano)
            proceso.stdin.write(limpio.tobytes())
            n += 1
            if progreso is not None:
                progreso(n, total)
            elif n % 30 == 0 or n == total:
                pct = (100.0 * n / total) if total > 0 else 0.0
                sys.stdout.write("\r  procesando... %d/%d cuadros (%.0f%%)" % (n, total, pct))
                sys.stdout.flush()
            ok, cuadro = captura.read()
    except BrokenPipeError:
        print("\n  [x] FFmpeg se cerro antes de tiempo.")
    finally:
        captura.release()
        if proceso.stdin:
            proceso.stdin.close()
        codigo = proceso.wait()

    print("")
    if codigo != 0:
        print("  [x] FFmpeg termino con error (codigo " + str(codigo) + ")")
        return False
    print("  [ok] " + salida + "  (zona x,y,w,h = " + str(zona) + ")")
    return True


# ---------------------------------------------------------------- main

def ruta_salida(entrada, carpeta_salida, es_video):
    base, ext = os.path.splitext(os.path.basename(entrada))
    if es_video:
        ext = ".mp4"
    carpeta = carpeta_salida or os.path.dirname(os.path.abspath(entrada))
    os.makedirs(carpeta, exist_ok=True)
    return os.path.join(carpeta, base + "_limpio" + ext)


def listar_archivos(ruta):
    if os.path.isdir(ruta):
        archivos = []
        for nombre in sorted(os.listdir(ruta)):
            ext = os.path.splitext(nombre)[1].lower()
            if ext in EXT_IMAGEN or ext in EXT_VIDEO:
                if "_limpio" in nombre or "_vista_previa" in nombre or "_mascara" in nombre:
                    continue
                archivos.append(os.path.join(ruta, nombre))
        return archivos
    return [ruta]


def elegir_archivos():
    """Abre el cuadro de dialogo de Windows para elegir fotos/videos."""
    import tkinter
    from tkinter import filedialog
    raiz = tkinter.Tk()
    raiz.withdraw()
    raiz.attributes("-topmost", True)
    patrones = " ".join("*" + e for e in sorted(EXT_IMAGEN | EXT_VIDEO))
    rutas = filedialog.askopenfilenames(
        title="Elige las fotos o videos a limpiar",
        filetypes=[("Fotos y videos", patrones), ("Todos los archivos", "*.*")],
    )
    raiz.destroy()
    return list(rutas)


PAUSAR_AL_SALIR = False


def main():
    global PAUSAR_AL_SALIR
    parser = argparse.ArgumentParser(
        description="Quita una marca de agua fija de fotos y videos. "
                    "Sin argumentos (doble clic) abre un selector de archivos "
                    "y deja marcar la zona con el mouse."
    )
    parser.add_argument("entrada", nargs="*", help="Archivos o carpetas con fotos/videos")
    parser.add_argument("-o", "--salida", help="Carpeta de salida (por defecto, la misma)")

    grupo = parser.add_argument_group("zona de la marca (elige una forma)")
    grupo.add_argument("--seleccionar", action="store_true",
                       help="Dibujar la zona con el mouse")
    grupo.add_argument("--zona", help="Zona exacta en pixeles: x,y,ancho,alto")
    grupo.add_argument("--esquina", choices=ESQUINAS, default="inf-der",
                       help="Esquina donde esta la marca (por defecto inf-der)")
    grupo.add_argument("--ancho", type=float, default=20.0,
                       help="Ancho de la zona en %% del ancho total (por defecto 20)")
    grupo.add_argument("--alto", type=float, default=10.0,
                       help="Alto de la zona en %% del alto total (por defecto 10)")
    grupo.add_argument("--margen", type=int, default=0,
                       help="Separacion en pixeles respecto al borde (por defecto 0)")

    parser.add_argument("--radio", type=int, default=5,
                        help="Radio de inpainting; mas alto = relleno mas suave (por defecto 5)")
    parser.add_argument("--metodo", choices=["telea", "ns"], default="telea",
                        help="Algoritmo de inpainting (por defecto telea)")
    parser.add_argument("--rectangulo", action="store_true",
                        help="Video: rellenar todo el rectangulo en vez de aislar la marca")
    parser.add_argument("--expandir", type=int, default=3,
                        help="Video: pixeles extra alrededor de la marca aislada (por defecto 3)")
    parser.add_argument("--guardar-mascara", action="store_true",
                        help="Video: guarda la mascara detectada como PNG para revisarla")
    parser.add_argument("--sin-grano", action="store_true",
                        help="No agregar grano al relleno")
    parser.add_argument("--crf", type=int, default=18,
                        help="Calidad del video: 0-51, menor = mejor (por defecto 18)")
    parser.add_argument("--vista-previa", action="store_true",
                        help="Solo guarda una imagen con la zona marcada en rojo")

    # Doble clic (sin argumentos) o archivos arrastrados sobre el .exe:
    # modo guiado, con seleccion de zona por mouse y pausa al terminar.
    sin_opciones = all(not a.startswith("-") for a in sys.argv[1:])
    args = parser.parse_args()
    if sin_opciones:
        PAUSAR_AL_SALIR = True
        args.seleccionar = True
        if not args.entrada:
            print("Elige las fotos o videos en la ventana que se abrio...")
            args.entrada = elegir_archivos()
            if not args.entrada:
                sys.exit("No elegiste ningun archivo.")
        print("En cada archivo: dibuja un rectangulo sobre la marca y presiona ENTER.\n")
    elif not args.entrada:
        parser.error("falta el archivo o carpeta de entrada")

    archivos = []
    for ruta in args.entrada:
        if not os.path.exists(ruta):
            sys.exit("No existe: " + ruta)
        archivos.extend(listar_archivos(ruta))
    if not archivos:
        sys.exit("No se encontraron fotos ni videos.")

    exitos = 0
    for archivo in archivos:
        ext = os.path.splitext(archivo)[1].lower()
        print(os.path.basename(archivo))
        if ext in EXT_IMAGEN:
            if procesar_imagen(archivo, ruta_salida(archivo, args.salida, False), args):
                exitos += 1
        elif ext in EXT_VIDEO:
            if procesar_video(archivo, ruta_salida(archivo, args.salida, True), args):
                exitos += 1
        else:
            print("  [x] Formato no soportado: " + ext)

    print("\nListo: " + str(exitos) + " de " + str(len(archivos)) + " archivo(s) procesados.")


if __name__ == "__main__":
    CODIGO_SALIDA = 0
    try:
        main()
    except SystemExit as salida:
        if isinstance(salida.code, str):
            print(salida.code)
            CODIGO_SALIDA = 1
        else:
            CODIGO_SALIDA = salida.code or 0
    except Exception as error:
        PAUSAR_AL_SALIR = True
        CODIGO_SALIDA = 1
        print("\nError inesperado: " + repr(error))
    finally:
        if PAUSAR_AL_SALIR:
            try:
                input("\nPresiona ENTER para cerrar esta ventana...")
            except EOFError:
                pass
    sys.exit(CODIGO_SALIDA)
