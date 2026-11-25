"""Vector depictions and receptor-frame projections for the two archived cases."""
import xml.etree.ElementTree as ET
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D, Bbox
from fontTools.pens.recordingPen import RecordingPen
from fontTools.svgLib.path import parse_path


def fit_points(ax, points, rect):
    """Isotropic projection into axes coordinates using the printed aspect ratio."""
    p=np.asarray(points)
    x,y,w,h=rect
    low=p.min(0); high=p.max(0); span=np.maximum(high-low,1e-8)
    ratio=ax.bbox.width/ax.bbox.height
    sy=min(h/span[1],w*ratio/span[0]);sx=sy/ratio
    return (sx,sy), np.array((x+w/2,y+h/2))-(high+low)*.5*np.array((sx,sy))

def molecule(ax, path, rect):
    items=[]
    for el in ET.parse(path).getroot():
        if not el.tag.endswith('path'):continue
        pen=RecordingPen();parse_path(el.get('d'),pen)
        verts=[];codes=[]
        for op,points in pen.value:
            if op=='endPath':continue
            if op=='closePath':points=[(0,0)]
            typ={'moveTo':Path.MOVETO,'lineTo':Path.LINETO,'qCurveTo':Path.CURVE3,
                 'curveTo':Path.CURVE4,'closePath':Path.CLOSEPOLY}[op]
            verts.extend(points);codes.extend([typ]*len(points))
        style=dict(t.split(':',1) for t in el.get('style','').split(';') if t)
        items.append((Path(verts,codes),style.get('fill',el.get('fill','none')),style.get('stroke',el.get('stroke','none'))))
    bounds=Bbox.union([p.get_extents() for p,_,_ in items])
    p=np.array([[bounds.x0,-bounds.y1],[bounds.x1,-bounds.y0]])
    (sx,sy),off=fit_points(ax,p,rect)
    trans=Affine2D().scale(sx,-sy).translate(*off)+ax.transData
    for path,fill,stroke in items:
        ax.add_patch(PathPatch(path,facecolor=fill,edgecolor=stroke,lw=.85,transform=trans,capstyle='round',joinstyle='round'))

def pocket_views(ax, root, metadata):
    """Draw a pair of molecular renders at equal scale in a common receptor frame."""
    from PIL import Image
    images = [np.asarray(Image.open(root / f"{pose}_pocket.png").convert("RGB"))
              for pose in ("original", "redocked")]
    # Crop both images identically, preserving their coordinate-to-page scale.
    union = np.any(np.stack(images) < 246, axis=(0, 3))
    rows, cols = np.where(union)
    pad = 11
    yy = slice(max(0, rows.min()-pad), min(union.shape[0], rows.max()+pad+1))
    xx = slice(max(0, cols.min()-pad), min(union.shape[1], cols.max()+pad+1))
    height, width = images[0][yy, xx].shape[:2]
    aspect = ax.bbox.width / ax.bbox.height
    dx = min(.475, .59 * width / (height * aspect))
    dy = dx * aspect * height / width
    for j, (center, title) in enumerate([
            (.255, "Original pose"), (.752, "Redocked pose")]):
        ax.text(center, .842, title, ha="center", va="center", fontsize=7.5, color="#233746", weight="bold")
        cy = .509
        ax.imshow(images[j][yy, xx], extent=[center-dx/2, center+dx/2, cy-dy/2, cy+dy/2],
                  aspect="auto", interpolation="lanczos", resample=True)
        mark = metadata["landmarks"][j]
        name = mark["residue"].title() + str(mark["number"])
        ax.text(center, .175, f"{name}  ·  {mark['distance']:.2f} Å",
                ha="center", va="center", fontsize=7.4, color="#72501F")
    ax.text(.502, .048, f"Paired heavy-atom RMSD  {metadata['rmsd_A']:.2f} Å",
            ha="center", va="center", fontsize=7.1, color="#364B52")
