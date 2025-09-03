# SheShe
<div align="right"><a href="README.md">EN</a></div>
**Smart High-dimensional Edge Segmentation & Hyperboundary Explorer**

SheShe convierte cualquier modelo probabilístico en un explorador guiado de su superficie de decisión, descubriendo regiones interpretables a partir de los máximos locales de probabilidad por clase o valor predicho.

## Características
- Clustering supervisado para clasificación y regresión
- Extracción de reglas y exploración de subespacios
- Herramientas de graficado 2D/3D

## Resumen matemático
SheShe aproxima el problema <code>max_x f(x)</code> siguiendo trayectorias de ascenso por gradiente hacia máximos locales y delimitando las regiones vecinas. Las ecuaciones detalladas se encuentran en la documentación.

## Instalación
Requiere Python ≥3.9. Instala desde [PyPI](https://pypi.org/project/sheshe/):

```bash
pip install sheshe
```

## Documentación
Consulta la [documentación](https://jcval94.github.io/SheShe/) para guías e información detallada.

## Autor
SheShe es desarrollado por José Carlos Del Valle – [LinkedIn](https://www.linkedin.com/in/jose-carlos-del-valle/) | [Portafolio](https://jcval94.github.io/Portfolio/)

## Licencia
MIT
