# Búsqueda desde C4: presupuesto de 90 minutos

Presupuesto solicitado: 90 minutos, en la ventana del 29 de octubre de 2026,
21:27:37--22:57:37 (Argentina).
Rama: `codex/c4-bound-push-90m`.

## Resultado

La búsqueda y el refinamiento exacto, utilizando exclusivamente candidatos
derivados de C4, certifican

\[
  \mu_0\ge 0.69290770636958636365.
\]

También se construyó un **template finito explícito**, con masas dependientes
de h y h=2^-139, que satisface la misma cota decimal. Su equilibrio racional,
inclusión estricta de Krawczyk y estabilidad por Cholesky intervalar se
verificaron a 4096,4352,4608 bits. Existe además un levantamiento con masas
fijas y h=2^-300, verificado a 8192,8448,8704 bits.

La cota anterior de nuestra búsqueda desde C4 era 0.6929076590503417. El nuevo
resultado recupera la cota publicada de S7 a los dígitos de comparación
disponibles. **No se encontró una mejora rigurosa sobre la cota publicada.**
Los objetos certificados son templates ponderados; no se construyó ni
certificó un grafo explícito. El resultado finito no depende solamente del
lema de existencia para h suficientemente pequeño: tiene datos racionales,
un h especificado y una prueba independiente de estabilidad.

## Procedencia y método

No se usaron el constructor S7, sus constantes, sus testigos ni su fórmula
de estabilidad para proponer o refinar candidatos. Se partió del ganador de
la búsqueda anterior desde C4, cuya cadena se reprodujo nuevamente desde
cuatro masas iguales. El refinamiento exacto produjo datos idénticos usando
dos entradas diferentes de esa misma procedencia C4.

Se exploraron divisiones múltiples de hasta 16 clústeres, liberación de
simetrías, nuevas divisiones sucesivas, masas muy pequeñas y fases alejadas
del centro de su grupo. También se probaron objetivos de margen con grado
mínimo fijado, penalizaciones explícitas por no alcanzar ese objetivo,
relajaciones de las restricciones conservadoras sobre R y probes finitos,
incluyendo conexiones entre padres opuestos en estos últimos.

El motor reutilizable conserva las restricciones del SDP y su validador
independiente. Las derivadas de la función lagrangiana se contrastaron con
solves independientes. Para masas pequeñas se utilizó una congruencia con
la métrica de masas; esto evita esconder modos negativos pequeños bajo la
tolerancia absoluta del solver. Los valores numéricos que parecían superiores
se conservaron como intentos inciertos o rechazados, sin elevarlos a cotas.

La mejoría certificada decisiva vino del patrón de densidades observado en
el candidato de C4: dos niveles en las conexiones vecinas, tres pares y un
grupo sin dividir. A partir de ese patrón se derivaron por sumas de aristas
las condiciones de equilibrio y las tres energías colectivas. Se ajustaron
exactamente sus polinomios y se verificó la interpolación en puntos racionales
independientes. Newton de alta precisión sólo propuso parámetros dyádicos;
la prueba final no depende de que Newton haya encontrado un óptimo.

Un desplazamiento racional estrictamente hacia el interior de las densidades
produce una forma límite positiva. La corrección R se recupera exactamente.
El verificador independiente comprueba F0,F1,K, la identidad presupuestaria,
las cajas, la rigidez interna y Cholesky intervalar a 192,256,384 bits.

Para el punto finito con masas variables, un programa lineal propuso una
dirección de primer orden. Se proyectó exactamente la corrección a las
identidades requeridas, se normalizó exactamente la variación de masas y
se reparó el equilibrio finito en un árbol. Los pasos mayores fallaron
Cholesky; h=2^-139 sí pasó todos los chequeos. El programa lineal por sí
solo no certifica ninguna mejora.

## Registro de recursos

El [recibo agregado](c4-bound-push-90m-receipt.json) enumera 23 corridas
principales, incluidos fallos, incertidumbres y dos controles de implementación:

| Tipo de invocación | Total |
| --- | ---: |
| SDP límite | 278883 |
| SDP finito | 14573 |
| Recuperación lineal de R | 193536 |

Son **293456 llamadas cónicas** en estas corridas. El replay final agrega
1859 llamadas SDP y dos auditorías Decimal. Los solves de regresión y las
pequeñas comprobaciones de desarrollo se registran aparte de este presupuesto
de búsqueda; no se presenta un total que los confunda con propuestas nuevas.
Los programas lineales y los solves racionales de los certificados también
tienen contadores separados. Se retuvieron los intentos que fallaron por
precisión insuficiente o por la elección inicial de la referencia angular.

Los journals completos están comprimidos en `events.jsonl.gz`, bajo cada
directorio `runs/c4-push90-*`. Incluyen cada propuesta, sus padres o fuente,
parámetros, configuración, comienzo y final de cada invocación, resultados
rechazados y hashes de código. Las corridas largas terminan por presupuesto
o reloj, sin omitir llamadas fallidas. Las copias de fuente contienen sólo
código del proyecto y del driver, sin entornos ni material privado.

## Reproducción del resultado ganador

Primero se genera un testigo exclusivamente desde C4 mediante los comandos
de [descubrimiento](c4-discovery.md). El replay final de esta sesión usa:

```powershell
uv run --locked --extra certification python -m kuramoto_template.asymptotic_search.c4_discovery --config examples/c4-dihedral.json --output runs/new-c4/discovery
uv run --locked --extra certification python scripts/refine_c4_result.py --run runs/new-c4/discovery --output runs/new-c4/nelder --gamma 0 --max-calls 1500
uv run --locked --extra certification python -m kuramoto_template.asymptotic_search.joint_limit --input runs/new-c4/nelder/top-search.json --config examples/c4-small-margin.json --output runs/new-c4/positive.json --max-joint-calls 1
uv run --locked --extra certification python scripts/exact_c4_active_family.py --input runs/new-c4/positive.json --output runs/new-c4/exact-limit
uv run --locked --extra certification python scripts/lift_exact_c4.py --certificate runs/new-c4/exact-limit/certificate.json --output runs/new-c4/finite-fixed-mass
uv run --locked --extra certification python scripts/probe_c4_mass_tangent.py --certificate runs/new-c4/exact-limit/certificate.json --output runs/new-c4/finite-variable-mass --h-bits 139
uv run --locked --extra certification python scripts/certify_witnesses.py --verify runs/new-c4/finite-variable-mass/certificate-139.json
```

Las familias simétricas son restricciones explícitas, no reducciones sin
pérdida de generalidad. La búsqueda amplia no constituye una demostración
de optimalidad de C4 ni de la configuración encontrada.

Validación del código final: 108 tests pasan. Se verificaron nuevamente los
certificados históricos. Se corrigió además una limitación de serialización
de fracciones de más de 4300 cifras, conservando el redondeo decimal exacto
y sin cambiar las decisiones de certificación.
