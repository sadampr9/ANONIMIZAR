#!/usr/bin/env python
"""Anonimiza PDFs tapando con recuadros negros datos sensibles.

Oculta por defecto: DNI, telefonos, la parte de los correos antes del "@" y
nombres/apellidos de personas.
Detecta en dos fuentes y une los resultados:
  1. Capa de texto del PDF (exacta).
  2. OCR (RapidOCR) sobre la pagina renderizada: cubre sellos, firmas y escaneos.

Las paginas con coincidencias se reemplazan por una imagen con los recuadros
(el texto de abajo desaparece de verdad). Las paginas sin coincidencias se copian
sin tocar, salvo con --rasterizar-todo.

Nombres de personas (no hay una regla exacta, se combinan tres senales):
  - texto tras un titulo: "senor", "Sra.", "Ing.", "Dr.", "Firmado digitalmente por"...
  - secuencias de 2+ palabras con algun nombre/apellido de diccionario_nombres.txt
  - propagacion: un nombre hallado en una pagina se oculta en todo el documento
  Para nombres que el programa no conozca: --texto "NOMBRE" o --nombres lista.txt

Uso:
  python anonimizar.py contrato.pdf
  python anonimizar.py carpeta_con_pdfs -o salida
  python anonimizar.py contrato.pdf --nombres personas.txt --regex "\\bEXP-\\d+"
  python anonimizar.py contrato.pdf --reporte     # lista lo detectado, sin escribir PDF
"""
import argparse
import io
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pypdf
import pypdfium2 as pdfium

# --- Patrones por defecto -------------------------------------------------
# Cada patron puede tener un grupo "oculta": solo ese grupo se tapa.
PATRONES = {
    "dni": [
        r"(?<!\d)(?P<oculta>\d{8})(?!\d)",  # 8 digitos aislados
        r"(?i)\b(?:DNI|C\.?E\.?|CARN[EÉ]\s+DE\s+EXTRANJER[IÍ]A)\b[^\d\n]{0,8}(?P<oculta>\d(?:[ \d]{5,9})\d)",
    ],
    "telefono": [
        r"(?<!\d)(?P<oculta>(?:\+?51[ -]?)?9\d{2}[ -]?\d{3}[ -]?\d{3})(?!\d)",  # movil Peru
        r"(?<![\w-])(?P<oculta>(?:\(0\d{1,2}\)[ -]?|0\d{1,2}[ ]-?[ ]?)\d{3}[ -]?\d{4})(?![\w-])",  # fijo (01) 123 4567
        r"(?i)\b(?:tel[eé]f(?:ono)?|telf|cel(?:ular)?|anexo|fono)\.?\s*(?:N[°ºo.]*\s*)?[:\-]?\s*(?P<oculta>\+?[\d][\d \-]{5,12}\d)",
    ],
    "correo": [
        r"(?P<oculta>[\w.+-]+)(?=\s?@\s?[\w-]+\.[\w.-]+)",  # solo antes del @
    ],
}
CATEGORIAS = list(PATRONES) + ["nombre"]

# --- Deteccion de nombres de personas --------------------------------------
LETRAS = "A-Za-zÁÉÍÓÚÑÜáéíóúñü�"
CONECTORES = {"de", "del", "la", "las", "los", "van", "von", "da", "di"}
TITULOS = {"sr", "sra", "srta", "senor", "senora", "senores", "senorita", "don", "dona", "dr", "dra",
           "ing", "abog", "lic", "mg", "econ", "arq", "prof", "cpc", "cpcc", "tte", "cnel", "gral"}
CALLES = {"av", "avenida", "jr", "jiron", "psj", "pasaje", "urb", "plaza", "parque", "ca", "calle", "mz", "malecon", "alameda"}
# Palabras que cortan una secuencia (cargos e instituciones) y nunca se aprenden como nombre.
NO_NOMBRE = set("""gerente general director directora jefe jefa oficina registro nacional identificacion estado civil
superintendencia migraciones reniec republica peru lima ministerio direccion coordinador coordinadora institucional
interinstitucional convenio clausula primera segunda tercera cuarta quinta anexo adenda resolucion jefatural
presidente presidenta secretario secretaria subgerente gerencia sub administrador administradora representante
apoderado apoderada notario notaria registrador registradora juez jueza fiscal senor senora calle avenida
distrito provincia departamento ciudad servicio servicios sistema sistemas gestion unidad area oficina
tecnologias informacion comunicaciones operaciones asesoria juridica web service gobierno
tecnico tecnica peruana peruano peruanos norma ley decreto supremo articulo numeral literal tabla figura seccion capitulo
titulo nombre nombres apellido apellidos dni ruc""".split())


def norm(s):
    """Minusculas sin tildes; la n con tilde pasa a n; el caracter roto de ext. queda como '?'."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def cargar_diccionario():
    ruta = Path(__file__).with_name("diccionario_nombres.txt")
    nombres = set()
    if ruta.exists():
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            if linea and not linea.startswith("#"):
                nombres.add(norm(linea))
    return nombres


class Nombres:
    """Detector de nombres. `aprender=True` solo recolecta; luego `buscar` usa lo aprendido."""

    def __init__(self, extra=()):
        self.dic = cargar_diccionario()
        self.conocidos = set()  # palabras de nombres hallados (>=4 letras)
        self.blobs = set()  # nombres escritos con letras espaciadas, juntados
        for n in extra:
            self.agregar_extra(n)

    def agregar_extra(self, nombre):
        for w in re.findall(f"[{LETRAS}]+", nombre):
            n = norm(w)
            if len(n) >= 4 and n not in NO_NOMBRE and "?" not in n:
                self.conocidos.add(n)

    # -- tokenizacion
    @staticmethod
    def tokens(texto):
        """Lista de dicts {w, lo, hi, fin, esp}. Junta runs de letras sueltas ("H I A L M E R")."""
        crudos = [(m.start(), m.group()) for m in re.finditer(r"\S+", texto)]
        out = []
        i = 0
        letra_sola = re.compile(f"^[{LETRAS}][,.;:]?$")
        while i < len(crudos):
            j = i
            while j < len(crudos) and letra_sola.match(crudos[j][1]):
                j += 1
            if j - i >= 4:  # run de letras sueltas: separar por mayuscula/minuscula
                grupo = []
                for k in range(i, j):
                    c = crudos[k][1][0]
                    clase = None if c == "�" else c.isupper()
                    if grupo and clase is not None and grupo[-1][0] is not None and clase != grupo[-1][0]:
                        out.append(Nombres._unir(grupo, crudos))
                        grupo = []
                    grupo.append((clase if clase is not None else (grupo[-1][0] if grupo else None), k))
                if grupo:
                    out.append(Nombres._unir(grupo, crudos))
                i = j
                continue
            pos, tok = crudos[i]
            m = re.search(f"[{LETRAS}]+", tok)
            if m and not re.search(r"\d", tok):
                out.append({"w": m.group(), "lo": pos + m.start(), "hi": pos + m.end(),
                            "fin": bool(re.search(r"[,.;:)(]$", tok)) or tok != m.group() and m.end() < len(tok),
                            "esp": False})
            elif re.fullmatch(r"[,.;:)(]+", tok) and out:  # signo suelto: cierra la secuencia
                out[-1]["fin"] = True
            else:
                out.append({"w": "", "lo": pos, "hi": pos + len(tok), "fin": True, "esp": False})
            i += 1
        return out

    @staticmethod
    def _unir(grupo, crudos):
        ks = [k for _, k in grupo]
        w = "".join(crudos[k][1][0] for k in ks)
        ult = crudos[ks[-1]][1]
        return {"w": w, "lo": crudos[ks[0]][0], "hi": crudos[ks[-1]][0] + 1,
                "fin": len(ult) > 1 or False, "esp": True}

    @staticmethod
    def _estilo(w):
        if len(w) < 2:
            return None
        if w.isupper():
            return "U"
        if w[0].isupper() and w[1:].islower():
            return "T"
        return None

    def _es_titulo(self, t):
        n = norm(t["w"]) if t else ""
        return n in TITULOS or bool(re.fullmatch(r"se.or(a|es|ita)?", n))

    # -- busqueda
    def buscar(self, texto, aprender=False):
        """Devuelve [(a, b, valor)] con los rangos de texto que son nombres."""
        toks = self.tokens(texto)
        res = []
        n = len(toks)
        i = 0
        while i < n:
            t = toks[i]
            est = self._estilo(t["w"]) if not t["esp"] else ("U" if t["w"].isupper() else None)
            conector = norm(t["w"]) in CONECTORES
            if t["esp"] and est == "U" or (est and not conector):
                run, j = [], i
                while j < n:
                    tj = toks[j]
                    ej = ("U" if tj["w"].isupper() and len(tj["w"]) > 1 else None) if tj["esp"] else self._estilo(tj["w"])
                    cj = norm(tj["w"]) in CONECTORES
                    if not (ej or cj):
                        break
                    if ej and run and not cj and ej != run[0][1] and run[0][1]:
                        break
                    run.append((tj, ej if ej else None))
                    j += 1
                    if tj["fin"]:
                        break
                # recortar conectores en extremos y partir por palabras de cargo/institucion
                segs, cur = [], []
                for tj, ej in run:
                    if not tj["esp"] and norm(tj["w"]) in NO_NOMBRE:
                        segs.append(cur)
                        cur = []
                    else:
                        cur.append(tj)
                segs.append(cur)
                pos = i
                for seg in segs:
                    # pos apunta al primer token de seg dentro de toks
                    while seg and norm(seg[0]["w"]) in CONECTORES and not seg[0]["esp"]:
                        seg = seg[1:]
                        pos += 1
                    while seg and norm(seg[-1]["w"]) in CONECTORES and not seg[-1]["esp"]:
                        seg = seg[:-1]
                    if seg:
                        previo = toks[toks.index(seg[0]) - 1] if toks.index(seg[0]) > 0 else None
                        if self._aceptar(seg, previo, toks):
                            a, b = seg[0]["lo"], seg[-1]["hi"]
                            res.append((a, b, texto[a:b]))
                            if aprender:
                                self._aprender(seg)
                    pos += len(seg) + 1
                i = max(j, i + 1)
                continue
            # palabra suelta ya conocida (propagacion)
            if not aprender and not t["esp"]:
                nw = norm(t["w"])
                if nw in self.conocidos and nw not in NO_NOMBRE:
                    res.append((t["lo"], t["hi"], t["w"]))
            i += 1
        return self._sin_duplicados(res)

    @staticmethod
    def _sin_duplicados(res):
        res = sorted(set(res))
        out = []
        for a, b, v in res:
            if out and a < out[-1][1]:  # solapado con el anterior: ampliar
                pa, pb, pv = out[-1]
                out[-1] = (pa, max(pb, b), pv)
            else:
                out.append((a, b, v))
        return out

    def _aceptar(self, seg, previo, toks):
        espaciados = [t for t in seg if t["esp"]]
        if previo and norm(previo["w"]) in CALLES:  # nombres de calles/avenidas no son personas
            return False
        if previo and self._es_titulo(previo):
            return True
        k = toks.index(seg[0])
        if k >= 2 and norm(toks[k - 1]["w"]) == "por" and norm(toks[k - 2]["w"]).startswith("digitalmente"):
            return True
        for t in espaciados:  # nombre en letras separadas
            blob = norm(t["w"])
            if len(blob) >= 8 and (blob in self.blobs or any(c in blob for c in self.conocidos)
                                   or any(len(d) >= 5 and d in blob for d in self.dic)):
                return True
        normales = [t for t in seg if not t["esp"] and norm(t["w"]) not in CONECTORES]
        hits = [t for t in normales if norm(t["w"]) in self.dic or norm(t["w"]) in self.conocidos]
        return len(normales) >= 2 and len(hits) >= 1

    def _aprender(self, seg):
        for t in seg:
            n = norm(t["w"])
            if t["esp"]:
                if len(n) >= 8:
                    self.blobs.add(n)
            elif len(n) >= 4 and n not in NO_NOMBRE and n not in CONECTORES and "?" not in n:
                self.conocidos.add(n)


# --- Coincidencias ----------------------------------------------------------
def compilar(categorias, regex_extra, textos):
    pats = []
    for cat in categorias:
        if cat in PATRONES:
            pats += [(cat, re.compile(p)) for p in PATRONES[cat]]
    for r in regex_extra:
        pats.append(("extra", re.compile(r)))
    for t in textos:
        pats.append(("texto", re.compile(re.escape(t), re.I)))
    return pats


def coincidencias(texto, pats, nombres=None):
    """Devuelve [(categoria, inicio, fin, valor)] de lo que hay que ocultar."""
    out = []
    for cat, rx in pats:
        for m in rx.finditer(texto):
            g = "oculta" if "oculta" in rx.groupindex else 0
            if m.group(g):
                out.append((cat, m.start(g), m.end(g), m.group(g)))
    if nombres is not None:
        out += [("nombre", a, b, v) for a, b, v in nombres.buscar(texto)]
    return out


# --- Geometria ------------------------------------------------------------
def fusionar_por_linea(rects):
    """Une rectangulos consecutivos de la misma linea; una linea nueva abre otro recuadro."""
    out = []
    for r in rects:
        if out:
            c = out[-1]
            solape = min(c[3], r[3]) - max(c[1], r[1])
            if solape > 0.5 * min(c[3] - c[1], r[3] - r[1]):
                out[-1] = (min(c[0], r[0]), min(c[1], r[1]), max(c[2], r[2]), max(c[3], r[3]))
                continue
        out.append(r)
    return out


def cajas_capa_texto(page, texto, pats, nombres, escala):
    """Cajas en pixeles segun la capa de texto. Devuelve [(cat, valor, (x0,y0,x1,y1))]."""
    if not texto or page.get_rotation():
        return []  # con rotacion se confia en el OCR
    tp = page.get_textpage()
    if tp.count_chars() != len(texto):  # indices desalineados: mejor no arriesgar
        return []
    izq, _, _, arr = page.get_cropbox() if page.get_cropbox() else page.get_mediabox()
    res = []
    for cat, a, b, val in coincidencias(texto, pats, nombres):
        rects = []
        for i in range(a, b):
            if texto[i].isspace():
                continue
            l, bo, r, t = tp.get_charbox(i)
            rects.append(((l - izq) * escala, (arr - t) * escala, (r - izq) * escala, (arr - bo) * escala))
        for rect in fusionar_por_linea(rects):
            res.append((cat, val, rect))
    return res


_ocr = None


def ocr_pagina(img):
    """[(caja(x0,y0,x1,y1), texto)] de las lineas que lee el OCR."""
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    resultado, _ = _ocr(np.array(img))
    lineas = []
    for box, texto, _score in resultado or []:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        lineas.append(((min(xs), min(ys), max(xs), max(ys)), texto))
    return lineas


def cajas_ocr(lineas, pats, nombres):
    res = []
    for (x0, y0, x1, y1), texto in lineas:
        ancho = (x1 - x0) / max(len(texto), 1)  # ancho medio por caracter
        for cat, a, b, val in coincidencias(texto, pats, nombres):
            res.append((cat, val, (x0 + a * ancho - ancho * 0.5, y0, x0 + b * ancho + ancho * 0.5, y1)))
    return res


# --- Proceso --------------------------------------------------------------
def procesar(ruta, salida, args, pats, extra_nombres):
    doc = pdfium.PdfDocument(str(ruta))
    escala = args.dpi / 72
    nombres = Nombres(extra_nombres) if "nombre" in args.solo else None

    # Fase 1: leer texto (capa de texto + OCR) de todas las paginas y aprender nombres
    textos, ocrs = [], []
    for i in range(len(doc)):
        page = doc[i]
        tp = page.get_textpage()
        textos.append(tp.get_text_range(0, tp.count_chars()))
        ocrs.append([] if args.sin_ocr else ocr_pagina(page.render(scale=escala).to_pil().convert("RGB")))
        print(f"  leida pagina {i + 1}/{len(doc)}", file=sys.stderr)
        if nombres:
            nombres.buscar(textos[-1], aprender=True)
            for _, t in ocrs[-1]:
                nombres.buscar(t, aprender=True)

    # Fase 2: localizar y tapar
    paginas_img, reporte = {}, []
    for i in range(len(doc)):
        page = doc[i]
        cajas = cajas_capa_texto(page, textos[i], pats, nombres, escala) + cajas_ocr(ocrs[i], pats, nombres)
        for cat, val, _ in cajas:
            reporte.append((i + 1, cat, val))
        if (cajas or args.rasterizar_todo) and not args.reporte:
            from PIL import ImageDraw
            img = page.render(scale=escala).to_pil().convert("RGB")
            d = ImageDraw.Draw(img)
            m = args.margen
            for _, _, (x0, y0, x1, y1) in cajas:
                d.rectangle((x0 - m, y0 - m, x1 + m, y1 + m), fill=(0, 0, 0))
            paginas_img[i] = img

    if args.reporte:
        for pag, cat, val in sorted(set(reporte)):
            print(f"{pag}\t{cat}\t{val}")
        return len(reporte)

    lector = pypdf.PdfReader(str(ruta))
    esc = pypdf.PdfWriter()
    for i, p in enumerate(lector.pages):
        if i in paginas_img:
            buf = io.BytesIO()
            paginas_img[i].save(buf, "PDF", resolution=args.dpi, quality=92)
            esc.add_page(pypdf.PdfReader(buf).pages[0])
        else:
            if "/Annots" in p:  # campos de formulario/anotaciones pueden guardar datos
                del p["/Annots"]
            esc.add_page(p)
    esc.add_metadata({"/Producer": "anonimizar.py"})
    with open(salida, "wb") as f:
        esc.write(f)
    print(f"-> {salida}  ({len(reporte)} datos ocultos en {len(paginas_img)} paginas)")
    return len(reporte)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", help="PDF o carpeta con PDFs")
    ap.add_argument("-o", "--salida", help="carpeta de salida (por defecto: <entrada>/anonimizados)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--margen", type=int, default=3, help="margen extra del recuadro en pixeles")
    ap.add_argument("--solo", nargs="+", choices=CATEGORIAS, default=CATEGORIAS,
                    help="categorias a ocultar (dni telefono correo nombre)")
    ap.add_argument("--regex", action="append", default=[], help="regex adicional (repetible)")
    ap.add_argument("--texto", action="append", default=[], help="texto literal adicional, ej. un nombre (repetible)")
    ap.add_argument("--nombres", help="archivo .txt con nombres completos a ocultar, uno por linea")
    ap.add_argument("--sin-ocr", action="store_true", help="solo capa de texto (mas rapido)")
    ap.add_argument("--rasterizar-todo", action="store_true", help="convertir tambien las paginas sin coincidencias")
    ap.add_argument("--reporte", action="store_true", help="solo listar lo detectado, no escribir PDF")
    args = ap.parse_args()

    extra = []
    if args.nombres:
        extra = [l.strip() for l in Path(args.nombres).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
    textos = args.texto + extra  # los nombres del archivo tambien se buscan tal cual

    entrada = Path(args.entrada)
    pdfs = sorted(entrada.glob("*.pdf")) if entrada.is_dir() else [entrada]
    destino = Path(args.salida) if args.salida else (entrada if entrada.is_dir() else entrada.parent) / "anonimizados"
    if not args.reporte:
        destino.mkdir(parents=True, exist_ok=True)
    pats = compilar(args.solo, args.regex, textos)
    for pdf in pdfs:
        if destino in pdf.parents:
            continue
        print(pdf.name, file=sys.stderr)
        procesar(pdf, destino / f"{pdf.stem}_ANONIMIZADO.pdf", args, pats, extra)


if __name__ == "__main__":
    main()
