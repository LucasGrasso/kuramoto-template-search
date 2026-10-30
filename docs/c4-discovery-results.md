# Redescubrimiento desde C4 y búsqueda con margen cero

Experimento del 29 de octubre de 2026, rama `codex/c4-unseeded-search`.

Partiendo de divisiones de cuatro grupos de masa 1/4, sin llamar al constructor
S7, leer sus testigos ni usar sus constantes, obtuvimos una variante racional
certificada con

\[
\lim_{h\to0^+}\mu(T_h)=\mu_0,\qquad
\mu_0\ge 0.6929076590503417.
\]

Mejora nuestro límite certificado anterior en aproximadamente 3.9530e-7 y
queda 4.7319e-8 por debajo del límite publicado de S7. Es una familia estricta
para h positivo suficientemente pequeño; no proporciona un h explícito ni un
grafo finito certificado. El mejor testigo finito histórico es otro resultado.

## Qué se exploró

Se probaron los 15 subconjuntos no vacíos de padres que pueden dividirse en
pares de masas iguales y offsets opuestos, con anchuras nuevas. Se añadieron
40 divisiones independientes y desiguales de 6,7,8,9,10,12 clústeres. El SDP
eligió libremente todas las densidades internas y vecinas permitidas.

La mejor cadena utilizó además las simetrías de C4 que preservan cada
subconjunto de padres divididos. Estas simetrías se calcularon para todos los
subconjuntos. El ganador tiene partición (1,2,2,2), masas de los pares casi
iguales y cociente de anchuras aproximadamente 1.39827848: redescubre una
geometría próxima a S7. Esta reducción es una restricción explícita de la
familia; el experimento no demuestra optimalidad global ni independencia de
toda elección de parametrización.

## Qué ocurrió con gamma=0

En la misma geometría descubierta por Nelder-Mead, el SDP dio:

| Margen exigido | Densidad numérica | Estado numérico | Variante racional |
| --- | ---: | --- | --- |
| 1e-7 | 0.6929072398978308 | estricta | no se intentó certificar este punto |
| 1e-9 | 0.6929076554552697 | neutral por el umbral 1e-8 | certificada |
| 1e-10 | 0.6929076590503338 | neutral por el umbral 1e-8 | certificada |
| 0 | 0.6929076602026789 | neutral | Cholesky intervalar falla |

La variante racional ensayada del punto con gamma=0 presenta un pivote negativo.
Esto impide usar esa variante como certificado mediante el criterio de segundo orden; no demuestra
que todos los extremos semidefinidos sean inestables. Otros puntos encontrados
al exigir gamma=0 sí tuvieron margen numérico positivo.

Una segunda continuación con margen 1e-10 alcanzó numéricamente
0.6929076604570537, pero su variante racional también falló Cholesky intervalar.
Se conserva como intento rechazado y no mejora el resultado certificado.

La certificación del ganador con gamma=1e-10 repara exactamente F0, F1 y K,
y verifica las condiciones del lema de levantamiento. La cota inferior exacta
redondeada hacia abajo a 20 decimales es 0.69290765905034172312. Las variaciones
máximas son menores que 1.198e-17 en masas, 2.012e-14 en P0 y 4.627e-15 en R.
El último pivote cuadrado de Cholesky es aproximadamente 5.05758e-10 y tiene
extremo inferior estrictamente positivo. La verificación pasó a 192,256,320
bits y mediante el comando independiente de verificación. El certificado
corresponde a esos datos racionales, no a los arrays flotantes originales.

## Presupuesto y controles

Las seis corridas principales completas conservaron todos los fallos e
incertidumbres, además de los ganadores:

| Corrida local bajo runs/ | Llamadas SDP | Mejor densidad aceptada |
| --- | ---: | ---: |
| c4-discovery-strict-v2-20261029 | 3569 | 0.6928491396177467 |
| c4-discovery-zero-v2-20261029 | 3864 | 0.6928493531763380 |
| c4-dihedral-strict-20261029 | 1309 | 0.6929071841189222 |
| c4-dihedral-zero-20261029 | 1197 | 0.6929027940916148 |
| c4-nelder-zero-20261029 | 549 | 0.6929076602026789 |
| c4-nelder-positive-20261029 | 360 | 0.6929076604570537 |

Cada corrida incluyó una auditoría Decimal separada. Las comparaciones
cruzadas de márgenes, tres intentos con P0 fijo y la reproducción del último
SDP añadieron 19 llamadas: 10867 en total para estas corridas y controles.
Las diferencias entre máximos de corridas externas no prueban monotonía del
óptimo; la tabla de márgenes anterior mantiene la geometría fija.

Los pilotos previos gastaron 80,1500,1327 llamadas. Una corrida detenida por
un error de escritura de Windows gastó 213. Otra interrupción conserva 707
resultados completos y puede haber interrumpido una llamada adicional; su
recibo quedó con propuesta pendiente. Estos costes no se omiten ni se cuentan
como parte de una comparación con presupuesto igualado. Los solves de pytest
son validación del software y se contabilizan aparte de la búsqueda.

## Reproducción y evidencia

Los [comandos completos](c4-discovery.md) reproducen la cadena C4, Nelder-Mead
con gamma=0, nuevo SDP con gamma=1e-10 y certificación racional. La cadena
original empleó 1309+549+1=1859 llamadas SDP y dos auditorías Decimal; la
certificación no usa solves cónicos. Se conservan los hashes históricos de
código, las versiones, los parámetros, los snapshots y los recibos de cada
ejecución. Se añadió posteriormente almacenamiento explícito de cada input
rechazado, sin cambiar las ecuaciones o la selección de propuestas.

Después de ese cambio se repitió la cadena completa desde cero. Añadió otras
1859 llamadas SDP y dos auditorías Decimal, reprodujo exactamente los mismos
datos racionales y la misma cota, y pasó nuevamente la verificación independiente.
Las tres fases del replay registran el hash de código
`f6679b82873702554b85d3329e057a4180702bf8996bd0191c261638adab9b12`.

El testigo numérico reproducido está en `runs/c4-final-replay-positive-20261029.json`; el
certificado y sus datos exactos, en
`runs/c4-final-replay-certificate-20261029/certificate.json`. Los outputs completos
permanecen en runs/, conforme a la política de esta rama; los tres testigos
históricos versionados y su manifiesto permanecen como evidencia histórica.

Validación final: 93 tests pasan, incluidas las pruebas de los solvers y de
certificación. Los testigos históricos pasaron la reproducción de verificación.
