"""Render the two OPRK1 poses in one orthographic receptor frame using a CPU depth buffer."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

DATA = Path(__file__).resolve().parents[1] / 'source/data/medicinal_figures'
BACKBONE = {'N', 'CA', 'C', 'O', 'OXT'}
RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}
POSE_COLORS = {"original": np.array([20, 63, 106]), "redocked": np.array([146, 28, 46])}
PATCH = np.array([154, 102, 44])
WHITE = np.array([255., 255., 255.])
LIGHT = np.array([-.35, .49, .80])
LIGHT /= np.linalg.norm(LIGHT)


def unit(a):
    a = np.asarray(a)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def parse_pocket(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith('ATOM  '):
            rows.append((tuple(float(line[i:i + 8]) for i in (30, 38, 46)),
                         line[21], int(line[22:26]), line[17:20].strip(),
                         line[12:16].strip(), line[76:78].strip()))
    return rows


def landmarks(atoms, poses):
    xyz = np.array([x[0] for x in atoms])
    keys = sorted(set(r[1:4] for r in atoms))
    rows = []
    for key in keys:
        idx = [i for i, r in enumerate(atoms) if r[1:4] == key and r[4] not in BACKBONE]
        if not idx:
            continue
        distances = [cdist(xyz[idx], p) for p in poses]
        rows.append((key, idx, distances))
    selected = []
    for j in range(2):
        valid = [r for r in rows if r[2][j].min() <= 4 and r[2][1-j].min() >= 6]
        key, idx, matrices = min(valid, key=lambda r: (r[2][j].min(), r[0]))
        i, k = np.unravel_index(matrices[j].argmin(), matrices[j].shape)
        selected.append(dict(chain=key[0], number=key[1], residue=key[2],
                             atom=atoms[idx[i]][4], ligand_index=int(k),
                             distance=float(matrices[j][i,k]), other_distance=float(matrices[1-j].min())))
    return selected


def camera(xyz, poses, bonds):
    points = np.vstack(poses)
    center = (points.max(0) + points.min(0)) / 2
    delta = poses[1].mean(0) - poses[0].mean(0)
    rays = xyz[None,:, :] - points[:,None,:]
    dist2 = (rays*rays).sum(2)
    neighbors = [[] for _ in poses[0]]
    for a,b in bonds:
        neighbors[a].append(b);neighbors[b].append(a)
    normals=[]
    for pose in poses:
        local=[np.cross(pose[nb[0]]-pose[i],pose[nb[1]]-pose[i])
               for i,nb in enumerate(neighbors) if len(nb)>=2]
        normals.append(unit(np.array(local)))
    candidates = []
    for k in range(192):
        z = 1 - 2 * (k+.5) / 192
        phi = k * 2.399963229729
        view = np.array([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z])
        along = rays @ view
        lateral = np.maximum(dist2 - along*along, 0)
        blockers = ((along>1) & (along<17) & (lateral<2.2**2)).sum(axis=1)
        faces=min(np.abs(n @ view).mean() for n in normals)
        score = .32*np.exp(-.65*blockers).mean()+.83*faces-.16*abs(unit(delta)@view)
        candidates.append((score,k,view))
    view = max(candidates,key=lambda c:c[:2])[2]
    flat=points-points.mean(0)
    flat-=(flat@view)[:,None]*view
    horizontal=np.linalg.svd(flat,full_matrices=False)[2][0]
    if horizontal@delta<0:horizontal=-horizontal
    up = unit(np.cross(view,horizontal))
    return center, np.stack([horizontal,up,view],axis=1)


class Scene:
    def __init__(self, span, pixels):
        self.n = pixels
        self.scale = pixels/span
        grid = (np.arange(pixels)+.5-pixels/2)/self.scale
        self.x, self.y = np.meshgrid(grid,grid[::-1])
        self.rgb = np.full((pixels,pixels,3),255.,dtype=np.float32)
        self.z = np.full((pixels,pixels),-1e5,dtype=np.float32)

    def region(self, p, pad):
        lo=p.min(0)-pad;hi=p.max(0)+pad
        n=self.n;s=self.scale
        return (slice(max(0,int(n/2-hi[1]*s)-1),min(n,int(n/2-lo[1]*s)+2)),
                slice(max(0,int(n/2+lo[0]*s)-1),min(n,int(n/2+hi[0]*s)+2)))

    def line(self,a,b,r,color):
        # Orthographic ray hits an approximation to a round capped stick.
        sl=self.region(np.stack([a,b]),r)
        x,y=self.x[sl],self.y[sl]
        if not x.size:return
        v=b-a; t=np.clip(((x-a[0])*v[0]+(y-a[1])*v[1])/(v[:2]@v[:2]+1e-9),0,1)
        dx=x-a[0]-v[0]*t;dy=y-a[1]-v[1]*t
        dz2=r*r-dx*dx-dy*dy
        dz=np.sqrt(np.maximum(dz2,0)); z=a[2]+v[2]*t+dz
        mask=(dz2>=0)&(z>self.z[sl])
        illum=np.clip((dx*LIGHT[0]+dy*LIGHT[1]+dz*LIGHT[2])/r,0,1)
        tone=color[None,None,:]*(.64+.36*illum[...,None])+12*illum[...,None]**8
        self.rgb[sl][mask]=np.clip(tone[mask],0,255);self.z[sl][mask]=z[mask]

    def triangle(self,p,normals,color):
        sl=self.region(p,.03);x=self.x[sl];y=self.y[sl]
        if not x.size:return
        a,b,c=p
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-10:return
        u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
        v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
        w=1-u-v;z=u*a[2]+v*b[2]+w*c[2]
        m=(u>=0)&(v>=0)&(w>=0)&(z>self.z[sl])
        if not m.any():return
        vn=unit(u[m,None]*normals[0]+v[m,None]*normals[1]+w[m,None]*normals[2])
        vn*=np.where(vn[:,2:3]<0,-1,1)
        diffuse=np.maximum(vn@LIGHT,0)
        self.rgb[sl][m]=color*(.67+.32*diffuse[:,None])
        self.z[sl][m]=z[m]


def ribbons(sc,atoms,xyz,poses):
    cas=[i for i,r in enumerate(atoms) if r[4]=='CA']
    near=cdist(xyz[cas],np.vstack(poses)).min(1)<10.8
    segments=[];run=[]
    for j,(i,keep) in enumerate(zip(cas,near)):
        if run and (not keep or atoms[i][1]!=atoms[run[-1]][1]
                    or atoms[i][2]!=atoms[run[-1]][2]+1 or np.linalg.norm(xyz[i]-xyz[run[-1]])>4.7):
            segments.append(run);run=[]
        if keep:run.append(i)
    segments.append(run)
    for ids in segments:
        if len(ids)<4:continue
        ca=xyz[ids]; t=np.arange(len(ca))
        curve=CubicSpline(t,ca,axis=0)(np.linspace(0,t[-1],len(ids)*8))
        tangent=unit(np.gradient(curve,axis=0))
        axis=unit(np.gradient(gaussian_filter1d(curve,11,axis=0,mode='nearest'),axis=0))
        wide=axis-(axis*tangent).sum(1)[:,None]*tangent
        # Axis direction is projected across the C-alpha path of each local helix.
        wide=unit(wide);normal=unit(np.cross(tangent,wide))
        theta=np.linspace(0,2*np.pi,9)
        rings=curve[:,None,:]+.92*np.cos(theta)[None,:,None]*wide[:,None,:]+.12*np.sin(theta)[None,:,None]*normal[:,None,:]
        ns=unit(.12*np.cos(theta)[None,:,None]*wide[:,None,:]+.92*np.sin(theta)[None,:,None]*normal[:,None,:])
        for j in range(len(curve)-1):
            for k in range(len(theta)-1):
                for idx in [[(j,k),(j,k+1),(j+1,k)],[(j,k+1),(j+1,k+1),(j+1,k)]]:
                    sc.triangle(np.array([rings[u,v] for u,v in idx]),np.array([ns[u,v] for u,v in idx]),np.array([220,223,224]))


def pocket_surface(sc,atoms,xyz,poses,landmark):
    near=cdist(xyz,np.vstack(poses)).min(1)<6.0
    idx=np.flatnonzero(near);p=xyz[idx]
    # Rolling-probe approximation to a solvent-excluded envelope, evaluated in the view frame.
    step=.24; probe=1.4
    lo=np.floor((p.min(0)-4)/step)*step;hi=p.max(0)+4
    axes=[np.arange(low,high+step,step) for low,high in zip(lo,hi)]
    blocked=np.zeros(tuple(len(axis) for axis in axes),dtype=bool)
    for i,point in zip(idx,p):
        rad=RADII[atoms[i][5]]+probe
        slices=tuple(slice(max(0,int((v-rad-base)/step)),min(len(axis),int((v+rad-base)/step)+2))
                     for axis,v,base in zip(axes,point,lo))
        grid=np.ix_(*(axis[ss] for axis,ss in zip(axes,slices)))
        blocked[slices]|=sum((g-v)**2 for g,v in zip(grid,point))<=rad**2
    inside=gaussian_filter(distance_transform_edt(blocked)*step,sigma=.9)
    solid=inside>=probe
    anyz=solid.any(2);i0=np.maximum(0,solid.shape[2]-1-np.argmax(solid[:,:,::-1],2))
    i1=np.minimum(i0+1,solid.shape[2]-1);i,j=np.indices(anyz.shape)
    a=inside[i,j,i0];b=inside[i,j,i1]
    z=axes[2][i0]+step*(a-probe)/np.maximum(a-b,1e-5)
    z=np.where(anyz,z,0)
    points=np.stack([sc.x,sc.y],axis=-1)
    def interp(field):
        return RegularGridInterpolator(axes[:2],field,bounds_error=False,fill_value=0)(points)
    coverage=interp(anyz.astype(float))
    mask=coverage>.51
    front=interp(z)/np.maximum(coverage,1e-6)
    coords=np.stack([sc.x[mask],sc.y[mask],front[mask]],axis=1)
    grad=np.stack(np.gradient(inside,step),axis=-1)
    normal=-unit(RegularGridInterpolator(axes,grad,bounds_error=False,fill_value=0)(coords))
    light=np.clip(normal@LIGHT,0,1)
    side=np.array([i for i in idx if (atoms[i][1],atoms[i][2])==(landmark['chain'],landmark['number'])
                   and atoms[i][4] not in BACKBONE])
    dist=cKDTree(xyz[side]).query(coords)[0]
    weight=np.clip((4.35-dist)/1.7,0,1)
    rgb=(1-weight[:,None])*np.array([245,245,243])+weight[:,None]*np.array([243,190,108])
    rgb*=.81+.19*light[:,None]
    alpha=(.72+.17*weight)*np.clip(coverage[mask],0,1)
    alpha*=np.where(front[mask]>sc.z[mask],1,.25)
    sc.rgb[mask]=sc.rgb[mask]*(1-alpha[:,None])+rgb*alpha[:,None]


def render(path,pixels=900):
    atoms=parse_pocket(DATA/'oprk1_pocket.pdb')
    q=json.loads((DATA/'pose_case.json').read_text())
    poses=[np.asarray(q[key]) for key in ['original_xyz','redocked_xyz']]
    xyz=np.array([r[0] for r in atoms]);land=landmarks(atoms,poses)
    center,basis=camera(xyz,poses,q['bonds'])
    points=np.vstack(poses)
    ca_idx=[i for i,r in enumerate(atoms) if r[4]=='CA']
    frame=xyz[np.array(ca_idx)[cdist(xyz[ca_idx],points).min(1)<10.8]]@basis
    middle=(frame.max(0)+frame.min(0))/2
    center=center+((middle-center@basis)*np.array([1,1,0]))@basis.T
    span=float(max(np.ptp(frame[:,:2],axis=0))+4.3)
    xyz=(xyz-center)@basis;poses=[(p-center)@basis for p in poses]
    path.mkdir(parents=True,exist_ok=True)
    for j,(pose,name) in enumerate(zip(poses,['original','redocked'])):
        ligand_color=POSE_COLORS[name]
        sc=Scene(span,pixels);ribbons(sc,atoms,xyz,poses)
        pocket_surface(sc,atoms,xyz,poses,land[j])
        # Orthographic cutaway: ligand and landmark sticks stay visible through the translucent envelope.
        sc.z.fill(-1e5)
        ids=[i for i,r in enumerate(atoms) if (r[1],r[2])==(land[j]['chain'],land[j]['number'])]
        for ii,i in enumerate(ids):
            for k in ids[ii+1:]:
                if (atoms[i][4] not in BACKBONE or atoms[k][4] not in BACKBONE) and np.linalg.norm(xyz[i]-xyz[k])<1.91:
                    sc.line(xyz[i],xyz[k],.105,PATCH)
        anchor=next(i for i in ids if atoms[i][4]==land[j]['atom'])
        start=xyz[anchor];end=pose[land[j]['ligand_index']]
        for f in np.arange(.11,.90,.23):
            sc.line(start+(end-start)*f,start+(end-start)*(f+.105),.065,np.array([120,92,54]))
        for k,m in q['bonds']:sc.line(pose[k],pose[m],.194,ligand_color)
        for point in pose:sc.line(point,point,.202,ligand_color)
        Image.fromarray(np.uint8(np.clip(sc.rgb,0,255))).save(path/(name+'_pocket.png'),optimize=True)
    info=dict(rmsd_A=float(np.sqrt(np.mean(np.sum((poses[0]-poses[1])**2,axis=1)))),
              landmarks=land,center=center.tolist(),basis_columns=basis.tolist(),span_A=span,
              selection='Nearest side-chain heavy atom within 4 A in the displayed pose; that residue must be at least 6 A from the other pose by side-chain heavy-atom distance.',
              receptor_sha256=q['sources']['data/LIT-PCBA/OPRK1/protein.pdb'])
    (path/'pocket_view.json').write_text(json.dumps(info,indent=2)+'\n')
    print([(x['residue'],x['number'],round(x['distance'],2),round(x['other_distance'],2)) for x in land])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DATA)
    parser.add_argument('--pixels',type=int,default=900)
    args=parser.parse_args();render(args.output,args.pixels)
