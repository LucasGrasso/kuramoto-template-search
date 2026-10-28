import sys,pathlib,os,json,base64,html,math,statistics,csv,shutil,zipfile
HERE=pathlib.Path.cwd()
os.environ['MPLCONFIGDIR']=str(HERE/'work/mpl-cache')
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=HERE/'outputs';FIG=OUT/'figuras';FIG.mkdir(exist_ok=True)
d=json.loads((OUT/'analisis-datos.json').read_text(encoding='utf-8'));exp=json.loads((OUT/'experimentos.json').read_text(encoding='utf-8'))
lin=json.loads((OUT/'invariantes-y-linaje.json').read_text(encoding='utf-8'))
ROOT=pathlib.Path(os.environ['KURAMOTO_REPO']);RUN=pathlib.Path(os.environ['KURAMOTO_SOURCE_RUN'])
state=json.loads((RUN/'sweep/checkpoint.json').read_text())
rows={r['id']:r for r in d['all_rows']};strict=[r for r in rows.values() if r['status']=='joint_asymptotic_strict'];mu_ref=d['reference']['mu']
palette={'complete':'#197c80','anchor_zero':'#a86724','cut_zero':'#b04f5e','mixed_zero':'#7160a3','internal_sparse':'#73812f'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
def save(fig,name):
 fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12.4,4.8),gridspec_kw={'width_ratios':[1.15,1]})
for f,col in palette.items():
 rr=[r for r in strict if r['family']==f]
 axs[0].scatter([max(r['score'],1e-7) for r in rr],[r['mu0'] for r in rr],s=25,alpha=.65,c=col,label=f,edgecolors='white',linewidths=.25)
axs[0].axhline(mu_ref,c='#232e39',ls='--',lw=1,label='Límite S7: Teorema 1')
axs[0].set(xscale='log',xlabel='Disimilitud a S7 (D, escala logarítmica)',ylabel='μ límite',title='416 límites numéricamente estrictos',ylim=(.12,.72))
axs[0].legend(fontsize=8,loc='lower left',ncol=2)
for m,marker in zip([7,8,9,10,12],['o','^','s','D','P']):
 rr=[r for r in strict if r['size']==m and r['mu0']>.69]
 axs[1].scatter([r['score'] for r in rr],[(mu_ref-r['mu0'])*1e6 for r in rr],label=f'{m} clusters',s=40,marker=marker,alpha=.8)
for i in (600,1,782,991):
 r=rows[i];axs[1].annotate(f'#{i}',(r['score'],(mu_ref-r['mu0'])*1e6),xytext=(5,5),textcoords='offset points',fontsize=8)
axs[1].set(xscale='log',yscale='log',xlabel='D a S7',ylabel='Déficit frente a S7 (10⁻⁶, log)',title='Los puestos altos conservan la geometría S7')
axs[1].legend(fontsize=8,loc='lower right');fig.tight_layout();save(fig,'01-densidad-estructura')

ids=[600,1,991,912,1012]
fig,axs=plt.subplots(len(ids),2,figsize=(11.7,12),gridspec_kw={'width_ratios':[1.2,1]})
rg=np.array([0,0,1,1,2,2,3]);rs=np.array(d['reference']['s'])
for k,i in enumerate(ids):
 t=d['details'][str(i)];r=rows[i];g=np.repeat(np.arange(4),t['parent_counts']);x=np.array(t['masses']);s=np.array(t['offsets']);P=np.array(t['p0'])
 ax=axs[k,0]
 ax.scatter(rs,rg,s=900*np.array(d['reference']['x']),facecolors='none',edgecolors='#bbb',linewidths=1.5,label='S7 referencia')
 ax.scatter(s,g,s=900*x,color=palette.get(r['family'],'#197c80'),alpha=.85,edgecolors='white')
 ax.set(xlim=(-1.65,1.65),ylim=(-.4,3.4),yticks=range(4),ylabel='Padre',xlabel='Offset normalizado s',title=f'#{i} · m={r["size"]} · μ={r["mu0"]:.9f}')
 ax.grid(axis='x',color='#eee');ax.set_axisbelow(True)
 im=axs[k,1].imshow(P,vmin=0,vmax=1,cmap='viridis',interpolation='nearest',aspect='equal')
 for b in np.cumsum(t['parent_counts'])[:-1]:axs[k,1].axhline(b-.5,color='white',lw=.7);axs[k,1].axvline(b-.5,color='white',lw=.7)
 axs[k,1].set(xticks=[],yticks=[],title=f'P0 · ceros adicionales: {r["extra_forbidden"]}')
fig.subplots_adjust(hspace=.55,wspace=.12,right=.86)
cax=fig.add_axes([.91,.14,.015,.71]);fig.colorbar(im,cax=cax,label='Densidad de conexión P0')
save(fig,'02-huellas')

fig,axs=plt.subplots(1,2,figsize=(12.4,4.8))
for ident in (1,62,782,304,991):
 cc=sorted([c for c in exp['calls'] if c['meta']['kind']=='collapse' and c['meta']['original_id']==ident],key=lambda c:c['meta']['alpha'])
 fused=next(c for c in exp['calls'] if c['meta']['kind']=='fused' and c['meta']['original_id']==ident)
 xx=[c['meta']['alpha'] for c in cc];yy=[(fused['mu']-c['mu'])*1e6 for c in cc]
 if ident==991:xx.append(1.);yy.append((fused['mu']-rows[ident]['mu0'])*1e6)
 axs[0].plot(xx,yy,marker='o',ls='--' if ident==991 else '-',label=f'#{ident} → 7 clusters'+(' (2 puntos)' if ident==991 else ''))
axs[0].set(xlabel='α: 0 = offsets colapsados; 1 = original',ylabel='Pérdida frente al caso fusionado (10⁻⁶)',title='Separaciones extra y fusión real')
axs[0].legend(fontsize=8)
for guard in (.0003,.003,.01):
 cc=sorted([c for c in exp['calls'] if c['meta']['kind']=='floor_guard' and c['meta']['guard']==guard],key=lambda c:c['meta']['gamma'])
 axs[1].plot([c['meta']['gamma'] for c in cc],[(mu_ref-c['mu'])*1e6 for c in cc],marker='o',label=f'guard={guard:g}')
 for c in cc:
  if c['status']!='joint_asymptotic_strict':axs[1].scatter(c['meta']['gamma'],(mu_ref-c['mu'])*1e6,s=110,facecolors='white',edgecolors='#b04f5e',lw=1.5,zorder=6)
xx=np.logspace(-7,-5,100);axs[1].plot(xx,4.381863597*xx*1e6,c='black',ls='--',label='Pendiente dual: 4,38186 × γ')
axs[1].set(xscale='log',yscale='log',xlabel='Piso γ del Hessiano completo',ylabel='Déficit frente a S7 (10⁻⁶)',title='El piso explica la pérdida del ganador')
axs[1].legend(fontsize=8);fig.tight_layout();save(fig,'03-fusion-y-piso')

fig,axs=plt.subplots(1,2,figsize=(12.4,4.8))
labels={'mass_opposite_02':'Masas: padres 0 ↔ 2','mass_opposite_13':'Masas: padres 1 ↔ 3','half_mass_02':'Masas: mitades 0 y 2','half_mass_1':'Masas: mitades 1','offset_w':'Offset w central','offset_parent_02':'Offset: medias 0 ↔ 2','offset_half_02':'Offset: anchuras 0 ↔ 2'}
mu0=rows[600]['mu0']
for name,label in labels.items():
 cc=sorted([c for c in exp['calls'] if c['meta']['kind']=='local' and c['meta']['direction']==name],key=lambda c:c['meta']['epsilon'])
 pairs=sorted([(c['meta']['epsilon'],(mu0-c['mu'])*1e6) for c in cc]+[(0.,0.)]);xs,ys=zip(*pairs)
 axs[0].plot(xs,ys,marker='o',label=label,ms=3)
axs[0].set(xlabel='ε, dirección normalizada',ylabel='Pérdida de μ (10⁻⁶)',title='Siete direcciones de perturbación reoptimizadas')
axs[0].legend(fontsize=7,loc='upper center',ncol=2)
cc=[c for c in exp['calls'] if (c['meta']['kind']=='local' and c['meta']['direction']=='offset_half_02') or (c['meta']['kind']=='fine_offset' and c['meta']['gamma']==1e-6)]
cc=sorted(cc,key=lambda c:c['meta']['epsilon']);pairs=sorted([(c['meta']['epsilon'],(c['mu']-mu0)*1e6) for c in cc]+[(0.,0.)]);xx,yy=zip(*pairs)
axs[1].plot(xx,yy,marker='o',color='#197c80')
for c in cc:
 if c['status']!='joint_asymptotic_strict':axs[1].scatter(c['meta']['epsilon'],(c['mu']-mu0)*1e6,facecolors='white',edgecolors='#b04f5e',s=100,zorder=5)
axs[1].axhline(0,color='#999',lw=.7);axs[1].set(xlabel='ε, diferencia entre anchuras de padres 0 y 2',ylabel='Ganancia de μ (10⁻⁶)',title='Mejora local pequeña a γ=10⁻⁶')
fig.tight_layout();save(fig,'04-sensibilidad-local')

fig,axs=plt.subplots(1,2,figsize=(12.4,5.7),gridspec_kw={'width_ratios':[1.3,1]})
order=d['cluster_leaf_order'];idx=[d['selected'].index(i) for i in order];dist=np.array(d['distance_matrix'])[np.ix_(idx,idx)]
im=axs[0].imshow(dist,vmin=0,vmax=.5,cmap='magma_r',interpolation='nearest')
axs[0].set(xticks=range(len(order)),yticks=range(len(order)),xticklabels=order,yticklabels=order,title='42 candidatos: disimilitud alineada')
axs[0].tick_params(axis='x',rotation=90,labelsize=5.3);axs[0].tick_params(axis='y',labelsize=5.3);fig.colorbar(im,ax=axs[0],fraction=.04,pad=.025)
for i in (600,1,62,782,991,912):
 vals=np.array(d['details'][str(i)]['full_spectrum']);axs[1].plot(range(1,len(vals)+1),vals,marker='o',lw=1,ms=3,label=f'#{i}')
axs[1].axhline(1e-6,color='#222',ls='--',lw=.8,label='Piso γ=10⁻⁶');axs[1].set(yscale='log',xlabel='Índice del autovalor ordenado',ylabel='Autovalor del Hessiano reescalado',title='Modos blandos e internos, en el gauge guardado')
axs[1].legend(fontsize=8,ncol=2);fig.tight_layout();save(fig,'05-clustering-y-espectros')

def fmt(x,n=10):return f'{x:.{n}f}' if x is not None else '—'
def table(headers,data):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in data)+'\n'
top_table=table(['ID','m','μ₀','guard','Δμ ref. ×10⁶','W₂ offsets','RMS P0','Fusión ε=0,05'],[
 [i,rows[i]['size'],fmt(rows[i]['mu0'],13),f'{rows[i]["h_guard"]:g}',f'{(mu_ref-rows[i]["mu0"])*1e6:.3f}',f'{rows[i]["offset_w2"]:.5f}',f'{rows[i]["p0_rms"]:.5f}',rows[i]['merged_0.05']] for i in d['top20']])
div_ids=[i for i in d['selected'] if i not in d['top20']]
div_table=table(['ID','m','Origen','μ₀','Ceros extra','Δ masas L1','W₂ offsets','RMS P0'],[
 [i,rows[i]['size'],rows[i]['family'],fmt(rows[i]['mu0'],10),rows[i]['extra_forbidden'],f'{rows[i]["mass_l1"]:.4f}',f'{rows[i]["offset_w2"]:.4f}',f'{rows[i]["p0_rms"]:.4f}'] for i in div_ids])
status_table=table(['m','Estrictos numéricos','Inciertos','Inviables condicionales','Inestables'],[
 [m]+[sum(r['size']==m and r['status']==s for r in rows.values()) for s in ('joint_asymptotic_strict','uncertain','conditional_infeasible','joint_unstable')] for m in (7,8,9,10,12)])
family_table=table(['Familia de origen','Límites estrictos','Mejor μ₀'],[[f,sum(r['family']==f for r in strict),fmt(max((r['mu0'] for r in strict if r['family']==f),default=None),12)] for f in ('complete','anchor_zero','cut_zero','mixed_zero','internal_sparse','nonuniform_anchor')])
fusion_ids=(1,62,782,153,991,304)
fusion_table=table(['ID original','m','μ₀ original','μ₀ tras fusión a 7','Ganancia ×10⁶'],[
 [i,rows[i]['size'],fmt(rows[i]['mu0'],13),fmt(next(c['mu'] for c in exp['calls'] if c['meta']['kind']=='fused' and c['meta']['original_id']==i),13),
  f'{(next(c["mu"] for c in exp["calls"] if c["meta"]["kind"]=="fused" and c["meta"]["original_id"]==i)-rows[i]["mu0"])*1e6:.3f}'] for i in fusion_ids])
floor_table=table(['γ','guard','μ₀','Estado'],[[f'{c["meta"]["gamma"]:g}',f'{c["meta"]["guard"]:g}',fmt(c['mu'],13),{'joint_asymptotic_strict':'estricto numérico','uncertain':'incierto','failed':'rechazado'}[c['status']]] for c in exp['calls'] if c['meta']['kind']=='floor_guard'])
local_table=table(['Dirección','Mejor μ₀ en ±0,001','Pérdida mínima ×10⁶'],[[label,
 fmt(max(c['mu'] for c in exp['calls'] if c['meta']['kind']=='local' and c['meta']['direction']==name and abs(c['meta']['epsilon'])==.001 and c['status']=='joint_asymptotic_strict'),13),
 f'{(mu0-max(c["mu"] for c in exp["calls"] if c["meta"]["kind"]=="local" and c["meta"]["direction"]==name and abs(c["meta"]["epsilon"])==.001 and c["status"]=="joint_asymptotic_strict"))*1e6:.3f}'] for name,label in labels.items()])
reopen_table=table(['ID','μ₀ soporte original','μ₀ soporte completo','Ganancia'],[[c['meta']['original_id'],fmt(c['meta']['original_mu'],10),fmt(c['mu'],10),f'{c["mu"]-c["meta"]["original_mu"]:.6f}'] for c in exp['calls'] if c['meta']['kind']=='reopen'])
lineage_table=table(['Movimiento','SDPs pagados','Pares estrictos comparables','Mejoran','Empeoran','Empatan'],[[op,v['paid'],v['strict_pairs'],v['improvements'],v['losses'],v['ties']] for op,v in lin['summary'].items()])
best=next(c for c in exp['calls'] if c['name']=='replay-000600');du=best['dual'];psd=du['full_psd']
original=json.loads((RUN/'sweep/top-limit.json').read_text());P=np.array(original['candidate']['p0']);P_ref=np.array(d['reference']['p0'])
maxp=float(np.max(abs(P-P_ref)));qs=[P[a,b] for a,b in ((0,3),(1,2),(2,5),(3,4))];anchors=[P[a,6] for a in (0,1,4,5)]
newbest=max((c for c in exp['calls'] if c['status']=='joint_asymptotic_strict'),key=lambda c:c['mu'])
dual_table=table(['Magnitud, ganador #600','Valor / observación'],[
 ['Grados mínimos activos','6 de 7; multiplicadores positivos y suma ≈ 1'],
 ['Grado del cluster ancla',f'{original["diagnostics"]["degrees"][6]:.12f}; multiplicador ≈ 8,5×10⁻¹⁴'],
 ['Dos autovalores completos menores','9,9999989×10⁻⁷ y 1,0970523×10⁻⁶'],
 ['Traza del dual PSD',f'{-psd["gamma_derivative"]:.9f}'],
 ['Autovalores duales significativos','4,38186353; siguiente ≈ 6,94×10⁻⁸'],
 ['Complementariedad tr(Y(H−γI))',f'{psd["complementarity_trace"]:.3e}'],
 ['Piso hard / mínimo observado','10⁻⁶ / 0,0341698; multiplicador despreciable'],
 ['Piso restoring / mínimo observado','10⁻⁴ / 0,215745; multiplicador despreciable'],
 ['Límite abs(R) / máximo observado','30 / 20,1015; no activo'],
 ['P0 cerca del techo buffered','7 aristas (3 internas y 4 vecinas no cruzadas), a 1−10⁻⁸'],
 ['Norma del dual del buffer superior',f'{du["buffer_upper"]["norm"]:.6f}']])
cost_table=table(['Trabajo nuevo','SDPs conjuntos','Auditorías Decimal','Lifts finitos / SDPs completos'],[
 ['42 reproducciones para duales',42,4,'0 / 0'],['Barrido piso × guard',9,0,'0 / 0'],['7 direcciones × 4 perturbaciones',28,0,'0 / 0'],
 ['Colapsos de offsets',21,0,'0 / 0'],['Fusiones efectivas a 7 clusters',6,0,'0 / 0'],['Reapertura de soporte',4,0,'0 / 0'],['Afinado local de offsets',10,0,'0 / 0'],['Total',120,4,'0 / 0']])
exp_path='experimentos.json'
text=f'''# ¿Los mejores candidatos son S7 o nuevas estructuras?

Informe de investigación — 26 de octubre de 2026{chr(32)*2}
Proyecto: `kuramoto-template` (proyecto registrado como `sync-template`); rama `asymptotic_search`.

## Resultado principal

El ganador original **#600 es S7 en su geometría**, con las mismas masas y offsets normalizados que la referencia. Su P0 es una perturbación de magnitud máxima {maxp:.3e}. Los 20 mejores conservan el soporte completo entre padres permitidos, provienen de reinicios y tienen masas agregadas en apenas tres patrones S7. Los casos de 8–12 clusters incluyen refinamientos próximos y algunas densidades de conexión heterogéneas; **no son todos copias exactas de S7**.

Las seis fusiones comprobadas conservan o mejoran la densidad al reducir a siete clusters. El caso #991 es la excepción visual más interesante: tiene conexiones cruzadas heterogéneas y no se identifica con S7 sólo permutando o fusionando gemelos exactos. Sin embargo, su fusión reoptimizada mejora μ de 0,692526846 a 0,692631716. No se observa una ventaja de sus grados de libertad extra.

Hay soportes realmente distintos y soluciones con masas desiguales, pero ninguno domina S7. El mejor límite estricto con una prohibición adicional de arista es #912, de 12 clusters: **μ₀ = 0,677811564386**. Es 0,015096142 inferior a la referencia. La conclusión es evidencia favorable a S7 dentro de este experimento; **no es una prueba de optimalidad, ni evidencia suficiente de una cuenca de atracción universal**.

La mejor μ₀ del barrido sigue siendo **0,6929033189148635**, inferior a **0,6929077063695864** por **4,3874547×10⁻⁶**. Al reducir el piso γ en las pruebas nuevas, se obtiene una solución estricta numérica con μ₀ = **{newbest['mu']:.16f}**, todavía inferior a la referencia por **{mu_ref-newbest['mu']:.3e}**, sin demostrar una familia superior.

**Aclaración sobre los dos valores del paper.** El número 0,6929050246493944 corresponde al ejemplo concreto certificado de la Proposición 1. El Teorema 1 demuestra la cota mayor 0,6929077063695864 mediante estados que se aproximan a ese límite. Ambos aparecen en la [sección V del paper](https://arxiv.org/html/2610.21271v1). Por tanto, nuestro número límite sí supera el valor del ejemplo concreto, pero sigue por debajo de la cota demostrada por el teorema. Además, todavía no se ha obtenido un levantamiento finito de ese candidato nuevo que supere el ejemplo de la proposición.

| Resultado | μ o μ₀ | Qué está establecido |
| --- | --- | --- |
| Paper, Proposición 1 | 0,6929050246493944 | Ejemplo concreto certificado y realización en grafos |
| Paper, Teorema 1 | 0,6929077063695864 | Cota inferior demostrada por una familia que se aproxima al límite |
| Nuevo candidato de este análisis | 0,6929072637491324 | Resultado numérico del modelo límite; sin lift finito de ese candidato ensayado |
| Mejor lift finito guardado del proyecto | 0,6926591963503084 | Estricto numérico, sin certificado riguroso |

La diferencia del candidato nuevo frente al ejemplo concreto del paper es +2,23910×10⁻⁶, y frente a su cota teórica es −4,42620×10⁻⁷. Estas comparaciones numéricas no convierten al candidato límite en una construcción finita certificada. Conseguir un ejemplo finito certificado por encima de la Proposición 1 sería una mejora del ejemplo explícito, aunque aún podría quedar por debajo del Teorema 1.

![Densidad frente a disimilitud estructural](figuras/01-densidad-estructura.png)

## Datos localizados, alcance y trazabilidad

La carpeta accesible es `{RUN}`. Existen **1.048 snapshots inmutables**, `sweep/checkpoint.json`, `sweep/result.json`, `sweep/trace.jsonl`, `sweep/top-limit.json`, `sweep/top-finite.json`. El testigo finito se lee de `{os.environ['KURAMOTO_FINITE_WITNESS']}`. No hay que suponer un `top-candidate.json` en la raíz. Los archivos están ignorados por `/runs/` en `.gitignore`, pero están presentes localmente. No fue necesario recuperar propuestas ni repetir el barrido.

El checkpoint termina en `phase=done`, cursor 1.050 y dos aciertos de caché exacta. Sus estados coinciden con el reporte:

{status_table}

Los 473 casos inviables son **inviabilidad numérica condicional** de un problema interno restringido a una geometría, soporte, guard, pisos y límites de R. No excluyen otras geometrías, correcciones, soportes o clases de modelos. Los 158 inciertos conservan su etiqueta; no se incorporan al ranking estricto.

Se analizan invariantes de todos los resultados con candidato disponible. Se seleccionan **42 testigos estrictos**: los 20 mejores, 17 mejores adicionales por estrato (m, familia) de 8–12 clusters, cuatro mejores descendientes de masas/offsets de esos tamaños y el mejor descendiente topológico. Se reensamblan y validan los 42; todos reproducen su estado y μ. La máxima violación primal en esa muestra es {max(v['max_primal'] for v in d['validation']):.3e}, dentro de la tolerancia configurada 10⁻⁹. Las 42 reproducciones con solver reproducen μ exactamente en este entorno.

Se recuperaron duales repitiendo sólo esos problemas internos, sin modificar la implementación: los snapshots guardaban residuos primal/dual del solver, pero no sus multiplicadores por restricción. Los datos, matrices, duales y nuevos resultados están en los archivos JSON de evidencia y en [experimentos.csv](experimentos.csv).

El hash de fuente actual coincide con el guardado: `{d['source_hash_saved']}`. HEAD: `4ba59ec75ca8759511978dace38ec968455fdbeb`. Se verificó el contenido de **103 archivos** de fuente/documentación/ejemplos: ninguno cambió. El checkout ya tenía cambios y archivos sin seguimiento de la implementación conjunta; se conservaron. No se cambió de rama ni se modificó `repair_sdp`. Los **445 tests** son el resultado previo informado; no se volvió a ejecutar esa suite porque este trabajo no cambia la implementación.

## Qué significa aquí «parecido a S7»

La comparación usa masas x, offsets s normalizados con ∑x s = 0 y ∑x s² = 1, etiquetas de los cuatro padres y P0. Se minimiza sobre las cuatro rotaciones cíclicas de padres; también se permite reflexión global de fase (g,s) ↦ (−g,−s), que es una simetría adicional indicada explícitamente. Las permutaciones de clusters se eliminan mediante transporte monótono de las medidas de offsets dentro de cada padre. No se comparan matrices por su orden de filas.

Se reportan por separado:

- **Δ masas L1**: diferencia de masas totales de los cuatro padres después del alineamiento.
- **W₂ offsets**: distancia cuadrática de transporte entre distribuciones condicionales de offsets, ponderada por la masa del padre candidato.
- **RMS P0**: error cuadrático medio de densidades bajo ese mismo acoplamiento de masas. Incluye las diagonales: un refinamiento exacto debe conectar a sus clones con densidad uno.
- **Desacuerdo de máscara**: masa de pares cuyo permiso de conexión cambia; además se cuenta el número de ceros adicionales respecto de los padres opuestos prohibidos.

La disimilitud descriptiva es D² = Δ masas L2² + W₂² + RMS P0². Para clustering entre dos candidatos se promedian las dos direcciones del transporte, pues la ponderación condicional es asimétrica. D **no se presenta como una distancia métrica certificada ni como criterio de equivalencia matemática**. En los candidatos originales seleccionados no hay empates ambiguos de offsets dentro de un padre. Se comprobaron 126 transformaciones aleatorias de permutación, rotación/reflexión y traslación de offsets: diferencia máxima de los invariantes {lin['max_invariance_error']:.2e}.

La fusión exploratoria agrupa sólo offsets del mismo padre que caben en un intervalo de diámetro ε; no usa encadenamiento simple. Se repite con ε = 0,02 / 0,05 / 0,1 / 0,2 y mide la dispersión de filas de P0 dentro de las celdas. Tener offsets próximos no basta para ser gemelos: se necesita también conexión interna completa y filas compatibles. La posterior fusión con reoptimización es una prueba numérica de redundancia para la densidad, no una identidad exacta entre testigos.

## Los 20 mejores

Los IDs son índices de snapshot, no índices de propuesta: por ejemplo #600 corresponde al archivo `sweep/joint-results/000600.json`. Todos los siguientes tienen familia de origen `complete`, máscara completa permitida y operación `restart`.

{top_table}

El caso #600 tiene x = (a,a,a,a,a,a,b), a = 0,13070922936304136 y b = 0,21574462382175186; s = (−u,u,−v,v,−u,u,0), u = 0,9834417763505504 y v = 1,3751251319532154. Por tanto v/u = w* = 1,3982781340204612. La discrepancia de x y s con S7 normalizado es de redondeo, no una deformación estructural.

Sus cuatro densidades cruzadas son q ≈ {sum(qs)/4:.13f}, con dispersión {max(qs)-min(qs):.2e}; las cuatro conexiones al ancla son ≈ {sum(anchors)/4:.13f}. Las otras siete aristas permitidas están en el techo buffered 1−10⁻⁸. Los seis clusters divididos tienen grado mínimo activo; el ancla tiene grado 0,738576226565.

Los casos #600 y #180 repiten la misma geometría con guard distinto. #1 y #391 se reducen a siete celdas a ε=0,05, con errores RMS P0 de 1,47×10⁻⁵ y 1,71×10⁻⁵. #782 necesita una fusión menos estricta (ε=0,2) para reducir a siete celdas; sus offsets extra tienen un efecto pequeño pero real. No debe confundirse estar cerca de S7 con ser idéntico a él.

**#991 merece conservarse como representante de una deformación heterogénea**: RMS P0 ≈ 0,07119 y dispersión de filas en la proyección a siete celdas ≈ {d['details']['991']['projection7']['row_scatter']:.5f}. Sus conexiones cruzadas incluyen valores ≈ 0,2486, 0,2857, 0,4564, 0,8519 y 0,8725. Estas diferencias no desaparecen por relabeling ni por fusionar gemelos exactos. Aun así, el resultado de fusión más abajo muestra que no mejora el objetivo frente a un S7 reajustado.

## Muestra diversa de 8–12 clusters y descendientes

{div_table}

«Origen» es una etiqueta de generación y no un invariante de soporte. Por ejemplo #616 procede de `mixed_zero` pero sus mutaciones reabrieron la máscara completa. Por eso la columna «ceros extra» es más informativa para juzgar novedades topológicas.

{family_table}

Los **41 límites estrictos con μ₀ > 0,69** y los **64 con μ₀ > 0,68** provienen de `complete`; el mejor soporte con ceros adicionales es #912. La familia con anclas fijadas no uniformes no tiene límites estrictos en esta muestra; esto no demuestra que esa familia sea inviable en general.

![Masas, offsets y densidades de representantes](figuras/02-huellas.png)

El clustering jerárquico de enlace promedio produce 24 / 20 / 14 / 12 / 4 grupos al cortar D en 0,05 / 0,1 / 0,15 / 0,2 / 0,3. A 0,15 hay un grupo grande con 29 candidatos y 13 grupos unitarios. Esta sensibilidad al umbral impide llamar a esos grupos «14 familias matemáticas». Las máscaras distintas y las dispersiones de filas son evidencia estructural más directa.

![Clustering y espectros](figuras/05-clustering-y-espectros.png)

## Fusión, capacidad de masas y separación de clones

Se asignó cada cluster al átomo S7 más próximo del mismo padre. Para cada celda se sumaron masas y se promediaron offsets con esas masas; después se normalizó el gauge y se resolvió de nuevo el problema de siete clusters con el mismo guard y pisos, soporte completo y árbol de reparación recalculado.

{fusion_table}

En cuatro representantes (#1, #62, #782, #304) también se recorrió s(α) = normalize((1−α)s_S7 + αs_original) para α = 0, 1/4, 1/2, 3/4, 1, manteniendo las masas y el número original de clusters. Al colapsar offsets sin fusionar filas todavía cambia la normalización del piso euclídeo del bloque hard: por eso μ en α=0 puede diferir en ≈10⁻⁶ del problema realmente fusionado. Las comparaciones de márgenes entre distintos tamaños deben interpretarse con esa diferencia de gauge, no como mejoras físicas de estabilidad.

Una cota elemental explica las mesetas del ranking. Sea M_g la masa total del padre g. De F0=0, cada cluster recibe la misma contribución de densidad desde los dos padres vecinos. Como cada contribución está acotada por la masa del vecino, y la contribución interna por M_g,

    μ₀ ≤ min_g [ M_g + 2 min(M_(g−1), M_(g+1)) ].

Es una desigualdad para este modelo con enlaces opuestos nulos y equilibrio de orden cero; no utiliza estabilidad. Para las masas del ganador, da exactamente la referencia 0,69290770637. Para el patrón S7 con t=0,003 da 0,692394952612. Los ocho últimos puestos de la tabla top20 casi saturan esta última cota: esa meseta refleja sobre todo sus masas iniciales. En #304, fusionar cambia μ en menos de 10⁻⁹; ese no es un indicio de una nueva densidad óptima de 12 clusters.

![Fusiones y dependencia del piso](figuras/03-fusion-y-piso.png)

## Restricciones activas, duales y modos débiles

{dual_table}

El mínimo completo está aproximadamente 10⁷ veces por encima de la estimación de redondeo matricial (9,33×10⁻¹⁴), aunque está en el piso impuesto de 10⁻⁶. La menor dirección, proyectada a los cuatro padres y renormalizada, es aproximadamente (0,707107, 0, −0,707107, 0). El subespacio de los dos modos efectivos menores coincide casi con el plano de dipolos (1,0,−1,0) y (0,1,0,−1): los valores singulares del solapamiento para #600 son 1 y 0,9999999982. Se usa el subespacio porque dos modos próximos pueden rotar y cambiar de signo entre soluciones.

Las normas de duales de las cotas guard contienen contribuciones artificiales de constantes exactas (diagonales y enlaces prohibidos); no se interpretan como aristas físicamente limitantes. Las ecuaciones también tienen dependencias, por lo que sus multiplicadores individuales no son únicos. Son útiles los pares primal/dual de grados, la matriz dual PSD y su traza, junto con la prueba perturbativa independiente.

Para un problema regular con piso H ⪰ γI, la sensibilidad de valor predicha por el dual es ∂μ*/∂γ = −tr(Y) ≈ −4,38186. El barrido de pisos conserva x y s del ganador:

{floor_table}

Entre γ=10⁻⁷ y 10⁻⁶, usando guard=0,003 y los resultados estrictos, la pendiente medida es ≈ −4,38222. La concordancia con el dual indica que **el piso de estabilidad, más que el guard en este rango, limita la aproximación de S7**. No todos los solves con γ=10⁻⁷ son aceptables: un resultado excede la tolerancia primal y otros son `optimal_inaccurate`; se mantienen rechazados o inciertos.

Los slacks dipolares, independientes de R, son particularmente útiles para orientar mutaciones. Con I la energía interna y G y L_in las matrices de la expansión,

    σ_j = I − d_jᵀ Gᵀ L_in† G d_j,   d₁=(1,0,−1,0), d₂=(0,1,0,−1).

La identidad ∑k_g=I y el hecho de que cada dipolo cambia ±1 a través de las cuatro aristas del ciclo dan d_jᵀ L_cut(k)d_j=I. Por tanto σ_j≥0 es una condición necesaria que ninguna elección de R puede remediar. Para #600 los slacks son 9,65051×10⁻⁶ y 1,05887×10⁻⁵. En S7 simétrico en la frontera ambos se anulan. Fuera de esa clase deben recalcularse las matrices completas; los dos polinomios de S7 simétrico no se pueden reutilizar como exclusión global.

La derivada parcial interna guardada es ∂λ_eff/∂w_ab = (z_a−z_b)², z=L_in†Gη, y respecto de densidad lleva el factor x_a x_b. Es no negativa **manteniendo G y los cortes fijos**. Cambiar masas, offsets o soporte altera también esos objetos y el conjunto factible: no es una predicción del efecto total reoptimizado.

## Sensibilidad local y calidad de la guía

Las direcciones de masas actúan como x(ε) ∝ x exp(εv), con ∑xv=0 y ∑xv²=1. Las de offsets se proyectan fuera de la traslación y la escala, luego se normalizan de nuevo. Cada punto se reoptimiza con el mismo soporte, árbol, guard=0,001 y γ=10⁻⁶. No se reutilizan matrices P0 como soluciones.

{local_table}

Se conservó el resultado incierto de ε=+0,001 en la dirección de anchuras, sin usarlo como testigo. Su pareja ε=−0,001 sí es estricta numérica y da **0,6929034376079906**, una mejora de **1,18693×10⁻⁷** frente a #600 al mismo piso. Reduce la anchura de los offsets del padre 0 y aumenta la del 2, conservando masas; rompe la simetría estrecha original. Al pasar a |ε|=0,005 la mejora desaparece. El afinado adicional confirma una ventana pequeña de mejora, no un nuevo récord: sigue a 4,26876×10⁻⁶ de la referencia. La evidencia **no permite afirmar que S7 simétrico maximice el SDP con piso positivo sobre todas las geometrías vecinas**.

![Sensibilidad local](figuras/04-sensibilidad-local.png)

De las 1.050 propuestas, 584 son reinicios, 249 movimientos topológicos guiados y 217 movimientos continuos. Los dos aciertos de caché dejan 583 / 248 / 217 solves pagados. Para evaluar cambios de μ sólo se comparan padres e hijos con modo densidad y el mismo guard; las diferencias de objetivos o restricciones entre rondas se excluyen.

{lineage_table}

De los 248 movimientos topológicos pagados, 246 tenían predicción fija positiva; sólo 3 de los 38 pares estrictos comparables mejoran μ. La predicción intenta reforzar una dirección cuadrática del Hessiano, no aumentar μ bajo todas las ecuaciones. Estos datos justifican revisar la función de propuesta, pero no prueban ineficacia general ni sustituyen una ablación con semillas y costes emparejados.

## Soportes nuevos: cuánto cuestan las prohibiciones

Se mantuvo cada geometría de cuatro representantes y se reabrieron todas sus prohibiciones adicionales, resolviendo de nuevo y recalculando el árbol:

{reopen_table}

Esta intervención relaja el soporte y también las restricciones internas del árbol, de modo que no aísla un único coeficiente causal. Muestra que esas máscaras son caras en el modelo actual. #912 y #1012 siguen por debajo de S7 incluso con soporte completo; sus masas/offsets y el resto de restricciones también importan. Antes de cerrar aristas para estabilizar modos, conviene probar redistribuciones de densidad que conserven F0 y K, o abrir primero un ciclo compensador.

## Lo finito y lo riguroso

El mejor límite original y el mejor caso finito son objetos diferentes. El refinamiento guardado conserva exactamente P0 y cambia R, por lo que no aporta una familia límite nueva.

- Mejor límite del barrido: **0,6929033189148635**.
- Mejor lift finito del barrido principal: **0,6922885138802887**, h=10⁻⁴.
- Mejor lift finito postprocesado: **0,6926591963503084**, h=3×10⁻⁵.
- En el postprocesado, ||R||_F **aumenta** de 67,0473 a 78,8367. El solve secundario L1 es `optimal_inaccurate` y no fue seleccionado. No debe describirse como una reducción exitosa de norma.

El lift postprocesado tiene fuerza residual 5,55×10⁻¹⁷ y gap restringido 7,18×10⁻¹⁴ frente a un piso de redondeo de 1,36×10⁻¹⁴: apenas ≈5,28 veces de holgura. La auditoría Decimal de 60 dígitos tiene pivotes positivos (menor ≈1,47×10⁻¹³), pero evalúa entradas float guardadas. No encierra un cero exacto de equilibrio ni realiza intervalos con redondeo exterior. Sus pendientes medidas del gap en el rango finito rondan h³, reflejando que el término de orden superior domina al muy pequeño coeficiente límite en ese rango; no contradicen la expansión asintótica a h aún menor.

Las cuatro auditorías nuevas de los replays #600, #1, #782 y #912 también tienen pivotes positivos. **Ninguna de estas auditorías, ni el solver, es un certificado riguroso**. Los flags de prueba siguen falsos y el archivo riguroso sigue vacío. Para una cota demostrada se necesitan datos exactos/racionales o intervalos, encerrar el equilibrio (o construirlo exactamente), positividad intervalar del Hessiano y realización como grafo. La optimalidad en la subclase simétrica S7 no implica optimalidad de las variantes heterogéneas o con masas diferentes.

## Mutaciones y próximos experimentos concretos

1. **Propuestas factibles de densidad en el núcleo de F0/K.** Para x,s fijos, construir A apilando `f0_map` y `cut_map`, y buscar Δp en ker(A) sobre aristas vecinas. En lugar de abrir/cerrar una arista aislada, optimizar un bloque de cuatro a ocho aristas compensadas sujeto a tangentes de caja. Priorizar mejora del grado mínimo y del menor de σ₁,σ₂; aceptar sólo tras el solve conjunto. Añadir un término de coste de pérdida de grado por multiplicadores duales.

2. **Equilibrar los dos dipolos al mover masas y offsets.** Usar diferencias centrales o derivadas del lagrangiano para σ₁,σ₂ y μ, con pasos logit 10⁻⁴–10⁻³. Resolver una dirección de confianza con dσ₁=dσ₂ (o con ambos no negativos) y masa/gauge preservados. Las perturbaciones aleatorias de 0,008 en logits son grandes frente a las pérdidas de 7×10⁻⁴–1,3×10⁻³ ya visibles con ε=0,001. El vector de anchuras (−1,1,0,0,1,−1,0), proyectado y normalizado, proporciona una dirección concreta de partida; debe seguirse con q cruzados independientes, sin invocar la obstrucción simétrica.

3. **Separar refinamiento redundante de nuevos modos.** Repetir splits de masa ρ∈{{0,2;0,5;0,8}} y separación δ∈{{0;0,01;0,03;0,1}} sobre cada uno de los siete átomos, incluyendo el ancla. Comparar contra su fusión de siete celdas con una normalización de margen físicamente consistente. Buscar ganancia robusta respecto de la fusión, no sólo un cluster extra o un P0 heterogéneo.

4. **Cambiar la población de reinicios para poner a prueba la atracción.** Mantener controles S7, pero añadir masas Dirichlet con piso explícito, particiones de padres equilibradas y asimétricas, offsets de dos a cuatro puntos por padre no derivados de S7, y soportes con ciclos compensadores. Registrar distancia inicial/final a S7 y supervivencia por estrato. El barrido actual no permite esa comparación: `make_outer_seed` siempre empieza en S7 y luego divide/modifica.

5. **Separar el límite del coste del lift finito.** Trazar una frontera de Pareto (μ₀, γ, ||R||, h viable) y minimizar directamente la pérdida de grado de primer orden max_a(−(Rx)_a), con reparaciones y caja incluidas, en vez de asumir que minimizar ||R|| basta. Usar un cociente generalizado o una forma física homogénea al comparar tamaños; no ordenar por gaps crudos de gauges distintos.

Presupuesto propuesto para el siguiente piloto — **no ejecutado aquí**: 900 SDPs conjuntos primarios con semilla fija (300 reinicios independientes, 300 movimientos dipolares, 150 splits/fusiones y 150 propuestas compensadas de soporte), más hasta 60 solves secundarios y 90 lifts de árbol, 12 auditorías y cero SDPs completos. Detener por esos contadores o 15 minutos de pared, lo que ocurra primero; conservar fallos y caches. Calibrar después con varias semillas y presupuesto idéntico antes de comparar métodos.

Un criterio de novedad útil sería: superar la referencia en μ₀ por una holgura mayor que los errores primal/dual; persistir al reducir γ y guard; retener dispersión estructural tras transporte y fusión; mejorar respecto de la fusión bajo un margen comparable; y disponer de lift finito independiente. Eso todavía es evidencia numérica. El certificado de un teorema sigue siendo una etapa adicional.

## Recursos gastados y entregables

Se anunció un máximo de **120 SDPs nuevos** y se respetó. No se regeneraron las 1.050 propuestas ni se entrenó una política.

{cost_table}

Estados de los 120 solves nuevos: **116 estrictos numéricos, 3 inciertos y 1 rechazado**. Los replays y controles también cuentan. Las primeras 107 llamadas, la persistencia y cuatro auditorías tardaron 7,32 s; el afinado posterior tiene 0,158 s de tiempos acumulados por resultado más las tres comprobaciones finales (no se usa esta cifra parcial como un tiempo total emparejado). Una falla de serialización del resumen posterior a diez solves se recuperó de sus diez snapshots sin repetir ni omitir llamadas.

Archivos de consulta:

- [top20.csv](top20.csv), [muestra-diversa.csv](muestra-diversa.csv) y [todos-los-resultados.csv](todos-los-resultados.csv).
- [experimentos.csv](experimentos.csv) y [experimentos.json](experimentos.json): configuraciones, estados y contadores.
- [analisis-datos.json](analisis-datos.json): invariantes, fusiones, espectros y validación.
- [invariantes-y-linaje.json](invariantes-y-linaje.json): controles de simetría y padres reales.
- [manifiesto.json](manifiesto.json): rutas, hashes y confirmación de integridad de fuente.
- `evidencia/originales/`: copia de los 42 snapshots seleccionados. `evidencia/`: 120 resultados nuevos, duales disponibles y cuatro auditorías.

La referencia S7, su fórmula μ=(4+2q*)/(7+q*) y la construcción de siete clusters corresponden a [O’Keeffe, *Beyond 11/16: dense Kuramoto networks that resist synchrony*, v1, Tabla 2 y Apéndice A](https://arxiv.org/html/2610.21271v1). El análisis y las intervenciones de este informe se basan en los archivos locales indicados, no en una afirmación externa de optimalidad global.
'''
(OUT/'informe-s7.md').write_text(text,encoding='utf-8')

# Minimal Markdown renderer: portable HTML with embedded figures, no external scripts or fonts.
import re
def inline(s):
 s=html.escape(s)
 s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
 s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
 s=re.sub(r'\*([^*]+)\*',r'<em>\1</em>',s)
 s=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
 return s
lines=text.splitlines();parts=[];i=0;toc=[]
while i<len(lines):
 line=lines[i]
 if not line.strip():i+=1;continue
 if line.startswith('#'):
  m=re.match(r'(#+) (.*)',line);level=len(m[1]);slug='title' if level==1 else f'section-{len(toc)}'
  if level==2:toc.append((slug,m[2]))
  parts.append(f'<h{level} id="{slug}">{inline(m[2])}</h{level}>');i+=1
 elif line.startswith('!['):
  m=re.match(r'!\[([^]]+)\]\(([^)]+)\)',line);data=base64.b64encode((OUT/m[2]).read_bytes()).decode()
  parts.append(f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(m[1])}"><figcaption>{inline(m[1])}</figcaption></figure>');i+=1
 elif line.startswith('|'):
  rr=[]
  while i<len(lines) and lines[i].startswith('|'):rr.append([v.strip() for v in lines[i].strip('|').split('|')]);i+=1
  parts.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(v)+'</th>' for v in rr[0])+'</tr></thead><tbody>')
  for r in rr[2:]:parts.append('<tr>'+''.join('<td>'+inline(v)+'</td>' for v in r)+'</tr>')
  parts.append('</tbody></table></div>')
 elif line.startswith('    '):
  code=[]
  while i<len(lines) and lines[i].startswith('    '):code.append(lines[i][4:]);i+=1
  parts.append('<pre>'+html.escape('\n'.join(code))+'</pre>')
 elif line.startswith('- '):
  parts.append('<ul>')
  while i<len(lines) and lines[i].startswith('- '):parts.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
  parts.append('</ul>')
 else:
  para=[line];i+=1
  while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','- ','    ')):para.append(lines[i]);i+=1
  parts.append('<p>'+inline(' '.join(para))+'</p>')
css='''body{margin:0;background:#f4f5f4;color:#24313a;font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1120px;margin:30px auto;background:white;padding:48px 58px;border-top:7px solid #197c80;border-radius:5px}h1{font-size:35px;line-height:1.22;letter-spacing:-.7px;color:#153a43;margin-top:0}h2{font-size:23px;margin:46px 0 15px;color:#19535b;border-top:1px solid #dfe7e6;padding-top:25px}p{max-width:100ch}a{color:#146e80}code{font:13px Consolas,monospace;background:#eef2f2;padding:2px 4px;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#edf4f3;border-left:3px solid #197c80;padding:17px;font:15px/1.6 Consolas,monospace}figure{margin:30px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#6d7b81;text-align:center}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.45;font-variant-numeric:tabular-nums}th{background:#e9f2f1;text-align:left;color:#24515a;font-weight:650}td,th{padding:8px 9px;border-bottom:1px solid #e3e8e8}tr:nth-child(even) td{background:#fafbfb}.table-wrap{overflow-x:auto;margin:20px 0}nav{background:#f2f7f6;border:1px solid #d9e5e2;border-radius:4px;padding:18px 22px;font-size:14px}nav a{display:inline-block;margin:4px 18px 4px 0}li{padding:3px 0}strong{color:#173e48}@media(max-width:800px){main{padding:25px 18px;margin:0}h1{font-size:29px}}@media print{body{background:white}main{margin:0;padding:0;max-width:none;border:0}h2{break-after:avoid}figure,table{break-inside:avoid}nav{display:none}a{color:inherit}}'''
nav='<nav>'+''.join(f'<a href="#{slug}">{html.escape(label)}</a>' for slug,label in toc)+'</nav>'
content='\n'.join(parts);content=content.replace('</h1>','</h1>'+nav,1)
(OUT/'informe-s7.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Análisis estructural del barrido SDP: S7 y variantes</title><style>'+css+'</style><main>'+content+'</main></html>',encoding='utf-8')
print('REPORT',len(text.split()),'WORDS','FIGURES',len(list(FIG.glob('*.png'))))
