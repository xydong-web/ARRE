"""Extract the Figure 5 depictions and Figure 8 example from the original project."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor, rdMolAlign
from scipy.spatial.distance import cdist

RDLogger.DisableLog('rdApp.warning')
DATA = Path(__file__).resolve().parents[1] / 'source' / 'data'
OUT = DATA / 'medicinal_figures'
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def depict_route():
    route=json.loads((DATA/'search/search_route.json').read_text())
    prefix=route['child'][-1]['smiles']
    assert prefix==route['parent'][-1]['smiles'] and route['shared_actions']==2
    assert route['child'][-1]['action'][:2]==route['parent'][-1]['action'][:2]
    full=Chem.MolFromSmiles(prefix)
    at=next(a for a in full.GetAtoms() if not a.GetAtomicNum())
    dummy=at.GetIdx()
    core=Chem.RWMol(full); core.RemoveAtom(dummy); core=core.GetMol()
    Chem.SanitizeMol(core);rdDepictor.Compute2DCoords(core)
    for key,smi in [('prefix',prefix),('parent',route['parent_smiles']),('child',route['child_smiles'])]:
        m=Chem.MolFromSmiles(smi)
        match=m.GetSubstructMatch(core);assert len(match)==core.GetNumAtoms()
        rdDepictor.GenerateDepictionMatching2DStructure(m,core,list(enumerate(match)))
        drawer=Draw.rdMolDraw2D.MolDraw2DSVG(460,210)
        o=drawer.drawOptions();o.useBWAtomPalette();o.clearBackground=False
        o.bondLineWidth=2.25;o.padding=.06;o.baseFontSize=.78
        if key=='prefix':o.atomLabels[dummy]='*'
        extra=set(range(m.GetNumAtoms()))-set(match)
        co=(.61,.33,.19) if key=='parent' else (.07,.43,.45)
        # Supply prepared SVG as transparent vector artwork. Color only the terminal fragment.
        drawer.DrawMolecule(m);drawer.FinishDrawing()
        import xml.etree.ElementTree as ET
        root=ET.fromstring(drawer.GetDrawingText())
        for el in root:
            tags=el.get('class','').split()
            atoms={int(a.split('-')[1]) for a in tags if a.startswith('atom-')}
            if key!='prefix' and atoms and atoms<=extra:
                hexco='#'+''.join(f'{int(x*255):02x}' for x in co)
                for attr in ('style','fill','stroke'):
                    if el.get(attr):el.set(attr,el.get(attr).replace('#000000',hexco))
        (OUT/(key+'.svg')).write_bytes(ET.tostring(root,encoding='utf-8'))

def pose(root):
    audit=root/'docs/diagnostics/gnina_pose_shift_20261025/full/pose_molecules.csv'
    frame=pd.read_csv(audit)
    flag=lambda s:s.astype(str).str.lower().isin(['true','1','1.0'])
    keep=(frame.arm=='F')&(frame.status=='ok')&(frame.pose_rmsd_A>5)
    for col in ['heavy_graph_same','stereo_smiles_same','protonation_signature_same']:keep &=flag(frame[col])
    pool=frame.loc[keep,['target','seed','idx','pose_rmsd_A']].sort_values(['target','seed','idx'])
    pool.to_csv(OUT/'eligible_pose_ids.csv',index=False,float_format='%.10f')
    median=pool.pose_rmsd_A.median()
    order=pool.sort_values(['pose_rmsd_A','target','seed','idx'])
    sel=order.iloc[len(order)//2];target=sel.target;seed=int(sel.seed);idx=int(sel.idx)
    row=frame.loc[sel.name];tag='r2CE4L4'
    r=root/'results'
    originals=[r/'robust'/tag/target/f's{seed}'/'pose_P0.sdf', r/'gnina'/tag/target/f's{seed}'/'best_poses.sdf']
    files=[r/'gnina_poses'/tag/target/f'gnina_poses_s{seed}.csv',r/'gnina'/tag/target/f's{seed}'/'gnina.csv']
    mols=[]
    for key,path,tab,score in zip(['original','redocked'],originals,files,[row.cnn_fixed,row.cnn_redock]):
        df=pd.read_csv(tab);qr=df[df.idx==idx];assert len(qr)==1 and qr.iloc[0].smiles==row.smiles
        assert np.isclose(qr.iloc[0].cnn_aff,score)
        found=[m for m in Chem.SDMolSupplier(str(path),removeHs=False) if m and m.GetProp('_Name')==str(idx)]
        assert len(found)==1
        (OUT/(key+'.mol')).write_text(Chem.MolToMolBlock(found[0]))
        mols.append(Chem.RemoveHs(found[0]))
    a,b=mols
    assert Chem.MolToSmiles(a)==Chem.MolToSmiles(b)
    assert Chem.MolToSmiles(a,isomericSmiles=False)==Chem.MolToSmiles(Chem.MolFromSmiles(row.smiles),isomericSmiles=False)
    pa=a.GetConformer().GetPositions();pb=b.GetConformer().GetPositions()
    # Preserve both coordinate frames. Enumerate graph automorphisms to resolve symmetric indices.
    maps=b.GetSubstructMatches(a,uniquify=False,maxMatches=10000,useChirality=True)
    assert maps
    ix=min(maps,key=lambda ix:np.mean(np.square(pa-pb[list(ix)])))
    pb=pb[list(ix)];rmsd=float(np.sqrt(np.mean(np.sum((pa-pb)**2,axis=1))))
    reported=rdMolAlign.CalcRMS(b,a,maxMatches=128)
    assert np.isclose(reported,row.pose_rmsd_A,atol=.002) and np.isclose(reported,rmsd,atol=.002)
    prot=root/'data/LIT-PCBA'/target/'protein.pdb'
    local_residues={}
    for ln in prot.read_text().splitlines():
        if not ln.startswith('ATOM  ') or ln[16] not in ' A' or ln[76:78].strip() not in ('C','N','O','S'):
            continue
        key=(ln[21],ln[22:27].strip(),ln[17:20].strip())
        coord=[float(ln[u:u+8]) for u in (30,38,46)]
        local_residues.setdefault(key,[]).append((ln,coord))
    lines=[]
    for atoms in local_residues.values():
        if cdist([item[1] for item in atoms],np.vstack([pa,pb])).min()<=14:
            lines.extend(item[0] for item in atoms)
    (OUT/'oprk1_pocket.pdb').write_text('\n'.join(lines)+'\nEND\n')
    residues={}
    for ln in prot.read_text().splitlines():
        if not ln.startswith('ATOM  ') or ln[16] not in ' A' or ln[76:78].strip() not in ('C','O','N','S'):continue
        key=(ln[21].strip(),ln[22:27].strip(),ln[17:20].strip())
        residues.setdefault(key,[]).append(([float(ln[j:j+8]) for j in (30,38,46)],ln[76:78].strip()))
    nearby=[]
    for k,atomlist in residues.items():
        xyz=np.array([v[0] for v in atomlist])
        dist=float(cdist(xyz,np.vstack([pa,pb])).min())
        if dist<6:
            nearby.append(dict(residue=' '.join(k),distance=dist,xyz=xyz.tolist(),elements=[v[1] for v in atomlist]))
    assert nearby
    manifest=dict(target=target,seed=seed,index_zero_based=idx,smiles=row.smiles,method='CGFlow + ARRE',eligible=len(pool),median_A=median,rmsd_A=rmsd,
                  selection='Upper middle order statistic of receptor-frame RMSD among ARRE candidates above 5 A with audited graph, stereo and protonation agreement. Ties in RMSD are ordered by target, seed and index. Scores are excluded from selection.',
                  cnn_original=float(row.cnn_fixed),cnn_redocked=float(row.cnn_redock),mapping_original_to_redocked=list(ix),original_xyz=pa.tolist(),redocked_xyz=pb.tolist(),
                  elements=[at.GetSymbol() for at in a.GetAtoms()],bonds=[[b.GetBeginAtomIdx(),b.GetEndAtomIdx()] for b in a.GetBonds()],receptor=sorted(nearby,key=lambda x:x['distance']),
                  sources={str(p.relative_to(root)):digest(p) for p in [audit,*originals,*files,prot]})
    (OUT/'pose_case.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
    print(target,seed,idx,len(pool),median,rmsd,row.cnn_fixed,row.cnn_redock)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('project',type=Path)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);depict_route();pose(args.project)
