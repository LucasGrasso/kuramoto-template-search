"""Draw the certified C4-derived template as a publication SVG.

The phase separation is explicitly magnified. Node areas and edge classes
come from the finite certificate; no visual coordinates enter the certificate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from flint import fmpq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from certify_witnesses import verify_certificate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--certificate',type=Path,required=True)
    parser.add_argument('--limiting-certificate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='SVG output')
    parser.add_argument('--display-h',type=float,default=.16)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    finite=json.loads(args.certificate.read_text())
    limiting=json.loads(args.limiting_certificate.read_text())
    verify_certificate(args.certificate)
    verify_certificate(args.limiting_certificate)
    source=json.loads((root/finite['source_file']).read_text())
    order=source['input_order']
    groups=np.array([limiting['exact_data']['groups'][i] for i in order])
    offsets=np.array([float(fmpq(limiting['exact_data']['s'][i])) for i in order])
    masses=np.array([float(fmpq(v)) for v in finite['exact_data']['x']])
    P=np.array([[float(fmpq(v)) for v in row] for row in finite['exact_data']['P']])
    h=fmpq(source['h'])
    h_bits=int(h.denominator).bit_length()-1
    if h!=fmpq(1,2**h_bits):raise ValueError('Expected a dyadic h')
    # Global rotation puts the singleton below, matching the supplied style.
    angles=(groups-1)*np.pi/2+args.display_h*offsets
    position=np.column_stack((np.cos(angles),np.sin(angles)))
    neighbors=[P[a,b] for a in range(len(masses)) for b in range(a+1,len(masses))
               if (groups[a]-groups[b])%2 and P[a,b]>0]
    q=float(min(neighbors))
    threshold=(q+1)/2
    palettes={1:('blue','#3b96fb','#80bfff','#257cfc'),
              2:('red','#ff5c6d','#ff9ba5','#ff4b65'),
              3:('green','#35ce86','#8ce5b4','#19bd70'),
              0:('yellow','#ffe44f','#fff4a8','#ffe13c')}
    letters={1:'A',2:'B',3:'C',0:'W'}
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix',
                         'svg.fonttype':'path','svg.hashsalt':'c4-certified-template'})
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_axes([0,0,1,1]);ax.set_aspect('equal')
    ax.set_xlim(-1.48,1.48);ax.set_ylim(-1.48,1.48);ax.axis('off')
    ax.add_patch(Circle((0,0),1,fill=False,edgecolor='#777d82',linewidth=2.7,
                        linestyle=(0,(4.3,3.2)),zorder=0))
    edge_counts={'near_one':0,'near_q':0}
    for partial in (False,True):
        for a in range(len(masses)):
            for b in range(a+1,len(masses)):
                if P[a,b]<=0:continue
                is_partial=P[a,b]<threshold
                if is_partial!=partial:continue
                color='#ec963b' if partial else '#303741'
                ax.plot(position[[a,b],0],position[[a,b],1],color=color,
                        linewidth=9.8,solid_capstyle='round',zorder=2 if partial else 1)
                edge_counts['near_q' if partial else 'near_one']+=1
    radius=.12*np.sqrt(masses/np.median(masses))
    for i,((x,y),r,j,s) in enumerate(zip(position,radius,groups,offsets)):
        name,base,_,_=palettes[int(j)]
        node=Circle((x,y),r,facecolor=base,edgecolor='#08090b',linewidth=5.1,zorder=3)
        node.set_gid(f'node-{i}-{name}');ax.add_patch(node)
        label=letters[int(j)]
        if np.count_nonzero(groups==j)>1:label+=r'^{+}' if s>0 else r'^{-}'
        if j==1:xy=(x+r+.075,y);ha='left';va='center'
        elif j==3:xy=(x-r-.075,y);ha='right';va='center'
        elif j==2:xy=(x,y+r+.065);ha='center';va='bottom'
        else:xy=(x,y-r-.07);ha='center';va='top'
        ax.text(*xy,'$'+label+'$',fontsize=25,ha=ha,va=va,color='#08090b',zorder=5)
    ax.text(0,.055,r'$T_{\mathrm{cert}}$',fontsize=46,ha='center',va='center',color='#08090b')
    density_lower=finite['result']['density']['lower']
    ax.text(0,-.19,rf'$\mu\geq {density_lower}$',fontsize=19,
            ha='center',va='center',color='#08090b')
    ax.text(0,-.35,rf'$h=2^{{-{h_bits}}}$',fontsize=19,ha='center',va='center',color='#08090b')
    ax.text(.37,.34,r'$q$',fontsize=28,ha='center',va='center',color='#08090b')
    ax.text(0,-.50,rf'$q\simeq {q:.10f}$',fontsize=16,ha='center',va='center',color='#303741')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,transparent=True,metadata={'Date':None,'Creator':'C4 certificate diagram'})
    plt.close(fig)
    ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns)
    tree=ET.parse(args.output);svg=tree.getroot();defs=svg.find(f'{{{ns}}}defs')
    for _,(name,base,light,edge) in palettes.items():
        gradient=ET.SubElement(defs,f'{{{ns}}}radialGradient',dict(id=f'gradient-{name}',cx='30%',cy='25%',r='85%',fx='30%',fy='25%'))
        for offset,color in [('0%',light),('55%',base),('100%',edge)]:
            ET.SubElement(gradient,f'{{{ns}}}stop',{'offset':offset,'stop-color':color})
    for group in svg.iter(f'{{{ns}}}g'):
        ident=group.get('id','')
        if not ident.startswith('node-'):continue
        name=ident.split('-')[-1]
        for path in group.findall(f'{{{ns}}}path'):
            style=path.get('style','')
            entries=[item.strip() for item in style.split(';') if item.strip() and not item.strip().startswith('fill:')]
            path.set('style','; '.join([f'fill: url(#gradient-{name})']+entries))
    meta=ET.SubElement(svg,f'{{{ns}}}metadata',{'id':'certificate-provenance'})
    meta.text=json.dumps(dict(certificate_sha256=hashlib.sha256(args.certificate.read_bytes()).hexdigest(),
        source=str(args.certificate),finite_h=str(h),display_h=args.display_h,
        angular_separation_is_schematic=True,node_area_proportional_to_mass=True,edges=edge_counts))
    tree.write(args.output,encoding='utf-8',xml_declaration=True)
    print(json.dumps(dict(svg=str(args.output),nodes=len(masses),edges=edge_counts,
                          certified_lower=finite['result']['density']['lower'])))


if __name__=='__main__':main()
