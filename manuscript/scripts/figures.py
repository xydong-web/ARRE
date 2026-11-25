#!/usr/bin/env python3
"""Rebuild all quantitative panels from the frozen, unrounded evidence."""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET

from fontTools.pens.recordingPen import RecordingPen
from fontTools.svgLib.path import parse_path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from comparison_figures import render as render_comparison
from display_names import ABLATIONS
from local_benchmark import DEV_METHODS, RECEIPT, accepted

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE if (HERE / "data").exists() else HERE / "source"
DATA = PROJECT / "data"
FIG = PROJECT / "figures"
FIG.mkdir(exist_ok=True)
C = dict(arre="#466F87", parent="#78B6C8", control="#F2D98E",
         contrast="#E48578", planning="#98AF1E", ink="#26343B", gray="#737B80",
         grid="#E5E9EB")
METHODS = {"B": ("CGFlow", C["parent"], "o"),
           "F": ("CGFlow + ARRE", C["arre"], "D"),
           "SynGA": ("SynGA", C["control"], "s"),
           "SynGBO": ("SynGBO", C["contrast"], "^")}
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 6.7,
    "xtick.labelsize": 6, "ytick.labelsize": 6.3, "legend.fontsize": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "lines.linewidth": 1.25,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "text.color": C["ink"], "axes.labelcolor": C["ink"],
    "axes.titleweight": "bold", "legend.frameon": False})
QA, PANEL = [], []


def js(name):
    return json.loads((DATA / name).read_text())


def csv(name):
    return pd.read_csv(DATA / name)


def source(fig, panels, files, stats):
    for panel in panels:
        PANEL.append(dict(figure=fig, panel=panel, source_files=files,
                          statistics=stats))


def title(ax, letter, text):
    ax.set_title(f"{letter}  {text}", loc="left", pad=4.5, fontsize=7.5, weight="bold")


def style(ax, x=True):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if x else "y", color=C["grid"], lw=.45)
    ax.tick_params(length=2.5, width=.6, pad=2)


def boot(values):
    a = np.asarray(values, float)
    rng = np.random.RandomState(0)
    z = rng.choice(a, size=(10000, len(a)), replace=True).mean(axis=1)
    return np.percentile(z, [2.5, 97.5])


def estimate(q):
    return q.get("mean", q.get("mean_pp")), q.get("ci95", q.get("boot_ci_pp"))


def point(ax, x, y, ci, color, marker="o", open_=False, size=4):
    ax.plot(ci, [y, y], color=color, lw=1.3, zorder=3)
    ax.plot([x], [y], marker=marker, ms=size, mec=color,
            mfc="white" if open_ else color, zorder=4)


def vertical(ax, x, entry, color, marker="o", open_=False):
    mean, ci = estimate(entry)
    ax.errorbar(x, mean, [[mean-ci[0]], [ci[1]-mean]], fmt=marker, ms=4.2,
                color=color, mfc="white" if open_ else color, mec=color,
                elinewidth=1.25, capsize=2, zorder=4)


def null(ax, value=0, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(value, color=C["gray"], ls=":", lw=.8)


def forest(ax, entries, labels, colors=None, xlabel="", limits=None, annotation=False):
    ax.axvline(0, color=C["gray"], ls=":", lw=.8)
    if colors is None:
        colors = [C["arre"]] * len(entries)
    for i, (q, color) in enumerate(zip(entries, colors)):
        m, ci = estimate(q)
        point(ax, m, i, ci, color, marker="D" if i == len(entries)-1 else "o")
        if annotation:
            ax.annotate(f"{m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]", (m, i),
                        xytext=(0, -11), textcoords="offset points",
                        ha="center", fontsize=5.7)
    ax.set_yticks(range(len(entries)), labels)
    ax.set_ylim(len(entries)-.4, -.7)
    if limits:
        ax.set_xlim(*limits)
    ax.set_xlabel(xlabel)
    style(ax)


def newfig(height=145, rows=3, cols=2, left=.12, right=.97, hspace=.56, wspace=.53,
           height_ratios=None, bottom=.09):
    f, axs = plt.subplots(rows, cols, figsize=(178/25.4, height/25.4),
                          height_ratios=height_ratios)
    f.subplots_adjust(left=left, right=right, bottom=bottom, top=.955,
                      hspace=hspace, wspace=wspace)
    return f, np.asarray(axs).reshape(rows, cols)


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    items=list(fig.texts)
    items.extend(t for legend in fig.legends for t in legend.get_texts())
    for ax in fig.axes:
        if not ax.axison:
            items.extend(ax.texts)
            items.extend([ax.title,ax._left_title])
            continue
        items.extend(ax.texts)
        items.extend([ax.title,ax._left_title,ax._right_title])
        if ax.get_legend() is not None:
            items.extend(ax.get_legend().get_texts())
        for axis,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            if not axis.get_visible():
                continue
            items.append(axis.label)
            low,high=sorted(lim)
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if low-1e-8 <= tick.get_loc() <= high+1e-8:
                    items.extend([tick.label1,tick.label2])
    for item in items:
        if not item.get_visible() or not item.get_text():
            continue
        # Hidden ticks outside the visible axis interval are not printed.
        box = item.get_window_extent(renderer)
        if box.width and (box.x0 < -1 or box.y0 < -1 or
                          box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1):
            outside.append(item.get_text())
    QA.append(dict(figure=name, width_mm=fig.get_figwidth()*25.4,
                   height_mm=fig.get_figheight()*25.4, offcanvas_text=outside))
    for suffix in ("pdf", "svg"):
        fig.savefig(FIG / f"{name}.{suffix}")
    fig.savefig(FIG / f"{name}.png", dpi=240)
    fig.savefig(FIG / f"{name}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("EXPORTED", name, "offcanvas", outside, flush=True)


def method_figure():
    export = js("method_v10/figure_one_export.json")
    assert (FIG / "Figure_01.pdf").is_file(), "Run figure_one.py first"
    QA.append(dict(figure="Figure_01", width_mm=export["width_mm"], height_mm=export["height_mm"],
                   offcanvas_text=[], conversion="PowerPoint slide to PDF"))
    source(1, "abcd", ["method_v10/Figure_01.pptx", "method_v10/figure_one_export.json",
                       "method_v10/selected_trajectory.json", "method_v10/selected_retrosynthesis.json",
                       "method_v10/source_manifest.json"],
           "Editable supplied plate; RHC expansion aligned with text; archived poses, reactions and assessment")


def benchmark():
    fig = plt.figure(figsize=(178/25.4,148/25.4))
    top = fig.add_axes([.16,.725,.81,.225])
    raw_vina = fig.add_axes([.16,.363,.162,.267])
    solved = fig.add_axes([.352,.363,.15,.267], sharey=raw_vina)
    tradeoff = fig.add_axes([.605,.363,.365,.267])
    equal_time = fig.add_axes([.16,.095,.342,.173])
    overhead = fig.add_axes([.605,.095,.365,.173])
    tm = csv("legacy/fig2_target_means.csv")
    local = accepted(DATA)["dev15"]["means"]
    for ax, metric in ((raw_vina, "m1"), (solved, "m2")):
        for i, (key, _) in enumerate(DEV_METHODS):
            color, marker = ((C["arre"], "D") if key in ("F", "RXCE") else
                             (C["parent"], "o") if key in ("B", "RX") else
                             (C["gray"], "s"))
            ax.scatter(local[key][metric], i, s=19, marker=marker,
                       color=color, linewidths=0, zorder=3)
        for boundary in (1.5, 3.5):
            ax.axhline(boundary, color=C["grid"], lw=.65)
        style(ax)
    raw_vina.set(yticks=range(len(DEV_METHODS)),
                 yticklabels=[name for _, name in DEV_METHODS], ylim=(5.6, -.6),
                 xlim=(-12.35,-10.65), xticks=[-12,-11.5,-11],
                 xlabel="Raw Vina (kcal mol$^{-1}$)")
    solved.set(xlim=(-4,100), xticks=[0,25,50,75,100],
               xlabel="AiZynthFinder (%)")
    solved.tick_params(axis="y", left=False, labelleft=False)
    solved.spines["left"].set_visible(False)
    title(raw_vina,"b","Local model comparison")
    ax = top
    for method, (name, color, marker) in METHODS.items():
        q = tm[tm.method==method]
        x, means, los, his = [], [], [], []
        for budget,g in q.groupby("budget"):
            lo,hi = boot(g.vina); x.append(budget/1000); means.append(g.vina.mean()); los.append(lo); his.append(hi)
        ax.plot(x,means,marker=marker,color=color,label=name,ms=3.3)
        ax.fill_between(x,los,his,color=color,alpha=.13,lw=0)
    ax.set(xlabel="New evaluations (thousands)",ylabel="Mean Vina (kcal mol$^{-1}$)",xticks=[16,32,64])
    title(ax,"a","Matched evaluation budget"); style(ax,False)
    ax.legend(loc="upper right",ncol=4,handletextpad=.45,columnspacing=1.1)
    ax = tradeoff
    for method,(name,color,marker) in METHODS.items():
        q = tm[(tm.method==method)&(tm.budget==64000)]
        ax.scatter(q.vina,q.aiz,s=8,color=color,alpha=.4,marker=marker)
        ax.scatter(q.vina.mean(),q.aiz.mean(),s=35,color=color,edgecolor=C["ink"],lw=.5,marker=marker,label=name)
    ax.set(xlabel="Mean Vina (kcal mol$^{-1}$)",ylabel="AiZynthFinder solved (%)")
    title(ax,"c","Docking–planning trade-off"); style(ax,False)
    z = js("extension_data.json")["Figure2"]
    ax = equal_time
    mu,ci = z["delta_mean"],z["delta_ci95"]
    for i,(target,v) in enumerate(zip(z["target"],z["equal_time_delta"])):
        ax.scatter(v,i,s=18,color=C["arre"])
    point(ax,mu,5,ci,C["arre"],"D",size=5)
    ax.axvline(0,color=C["gray"],ls=":",lw=.8)
    ax.set(yticks=range(6),yticklabels=z["target"]+["Mean (95% CI)"],ylim=(5.6,-.6),
           xlabel="ARRE − parent Vina (kcal mol$^{-1}$)")
    style(ax); title(ax,"d","Equal observed elapsed time")
    ax = overhead
    ax.scatter(z["iteration_time_ratio"],range(5),s=19,color=C["parent"])
    ax.axvline(1,color=C["gray"],ls=":",lw=.8)
    ax.set(yticks=range(5),yticklabels=z["target"],ylim=(5.6,-.6),xlim=(.98,1.3),
           xlabel="Iteration time ratio (ARRE / parent)")
    style(ax); title(ax,"e","Measured runtime overhead")
    save(fig,"Figure_02")
    source(2,"a",["legacy/fig2_target_means.csv"],"15 target means; 3 seeds; unchanged pointwise percentile target bootstrap, 10,000 draws, RandomState 0")
    source(2,"b",[RECEIPT],"six archived unadjusted Vina and AiZynthFinder means; same oracle, 64,000 distinct evaluations, 15 development tasks and 3 seeds; descriptive")
    source(2,"c",["legacy/fig2_target_means.csv"],"15 target means at 64,000 evaluations after averaging 3 seeds, with overall means")
    source(2,"de",["walltime.json","extension_data.json"],"5 paired target means, 2 seeds; nominal 95% t interval; measured elapsed time")


def molecule(ax, smiles, title_=None, color="black"):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    opts = Draw.MolDrawOptions()
    opts.bondLineWidth = 2
    opts.baseFontSize = .8
    ax.imshow(Draw.MolToImage(m,size=(1000,600),kekulize=False,options=opts))
    ax.axis("off")
    if title_:
        ax.set_title(title_,fontsize=6.3,color=color,pad=0,weight="normal")


def vector_molecule(ax, smiles, label, color):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(320, 160)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.minFontSize = 19
    opts.maxFontSize = 25
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope():
            opts.atomLabels[atom.GetIdx()] = f"{atom.GetIsotope()}*"
    drawer.DrawMolecule(Draw.rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False))
    drawer.FinishDrawing()
    root = ET.fromstring(drawer.GetDrawingText())
    paths = []
    for element in root:
        assert element.tag == "{http://www.w3.org/2000/svg}path", element.tag
        pen = RecordingPen()
        parse_path(element.attrib["d"], pen)
        vertices, codes = [], []
        for command, points in pen.value:
            if command == "endPath":
                continue
            if command == "closePath":
                points = [(0, 0)]
            code = {"moveTo": MplPath.MOVETO, "lineTo": MplPath.LINETO,
                    "qCurveTo": MplPath.CURVE3, "curveTo": MplPath.CURVE4,
                    "closePath": MplPath.CLOSEPOLY}[command]
            vertices.extend(points)
            codes.extend([code] * len(points))
        path = MplPath(vertices, codes)
        attrs = dict(item.split(":", 1) for item in
                     element.get("style", "").split(";") if item)
        ax.add_patch(PathPatch(path, facecolor=attrs.get("fill", element.get("fill", "none")),
                               edgecolor=attrs.get("stroke", "none"), linewidth=.78,
                               capstyle="round", joinstyle="round"))
        paths.append(path.get_extents())
    box = Bbox.union(paths)
    ax.set(xlim=(box.x0-4,box.x1+4), ylim=(box.y1+4,box.y0-4), aspect="equal")
    ax.axis("off")
    ax.text(.5,1.045,label,ha="center",va="bottom",fontsize=6.3,color=color,
            transform=ax.transAxes)


def band(ax,q,x,color,label=None,scale=1):
    q=q.sort_values(x)
    ax.plot(q[x],scale*q["mean"],color=color,label=label)
    ax.fill_between(q[x],scale*q.lower,scale*q.upper,color=color,alpha=.14,lw=0)


def search():
    fig=plt.figure(figsize=(178/25.4,110/25.4))
    ax=fig.add_axes([.05,.62,.90,.33]);ax.axis("off")
    title(ax,"a","Recorded shared-prefix continuation | ADRB2, seed 0")
    r=js("search/search_route.json")
    for rect,smi,label,col in [
        ([.000,.29,.135,.42],r["child"][1]["smiles"],"Initial block",C["ink"]),
        ([.163,.29,.135,.42],r["child"][1]["action"][2],"Linker block",C["ink"]),
        ([.381,.25,.182,.50],r["child"][2]["smiles"],"Shared prefix",C["ink"]),
        ([.639,.57,.116,.33],r["parent"][2]["action"][2],"Parent block",C["ink"]),
        ([.811,.57,.186,.33],r["parent_smiles"],"Recorded parent",C["ink"]),
        ([.639,.076,.116,.33],r["child"][2]["action"][2],"New block",C["arre"]),
        ([.811,.076,.186,.33],r["child_smiles"],"New completion",C["arre"])]:
        vector_molecule(ax.inset_axes(rect),smi,label,col)
    ax.plot([.571,.601,.601],[.50,.50,.737],lw=.85,color=C["gray"],transform=ax.transAxes)
    ax.plot([.601,.601],[.50,.238],lw=.85,color=C["gray"],transform=ax.transAxes)
    for start,end in [((.300,.50),(.375,.50)),
                      ((.601,.737),(.633,.737)),((.601,.238),(.633,.238)),
                      ((.757,.737),(.808,.737)),((.757,.238),(.808,.238))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle="->",mutation_scale=8,color=C["gray"],
                                     lw=.85,transform=ax.transAxes,shrinkA=0,shrinkB=0))
    ax.text(.148,.50,"+",fontsize=8,ha="center",va="center",transform=ax.transAxes)
    ax.text(.338,.31,r["child"][1]["action"][0],fontsize=6,ha="center",transform=ax.transAxes)
    assert r["parent"][2]["action"][0] == r["child"][2]["action"][0]
    ax.text(.705,.528,r["child"][2]["action"][0],fontsize=6,ha="center",transform=ax.transAxes)
    score = f"Prefix 2 · iteration {r['iteration_zero_based']} · Vina {r['child_vina']:.3f}".replace("Vina -","Vina −")
    ax.text(.997,-.015,score+" kcal mol$^{-1}$",
            fontsize=6,ha="right",va="top",color=C["arre"],transform=ax.transAxes)
    ax=fig.add_axes([.09,.363,.385,.187])
    q=csv("search/search_allocation_summary.csv");q.budget/=1000
    colors={"onpolicy":C["parent"],"prefix1":C["arre"],"prefix2":C["control"]}
    lines=("-", "--", "-.")
    markers=("o","s","^")
    for arm,color,ls,label in zip(colors,colors.values(),lines,("On-policy","Prefix 1","Prefix 2")):
        band(ax,q[q.arm==arm],"budget",color,label,100)
        ax.lines[-1].set_linestyle(ls)
    title(ax,"b","New-evaluation allocation")
    ax.set(xlabel="Docking calls (thousands)",ylabel="Allocation (%)",ylim=(0,105),xlim=(0,64),
           yticks=[0,50,100],xticks=[0,16,32,48,64])
    ax.legend(loc="upper right",fontsize=6,ncol=1,labelspacing=.18,handlelength=1.8)
    style(ax,False)
    ax=fig.add_axes([.58,.363,.38,.187]);q=csv("search/search_yield_summary.csv")
    for j,(arm,color) in enumerate(colors.items()):
        w=q[q.arm==arm].sort_values("phase")
        ax.errorbar(w.phase+(j-1)*.20,w["mean"],
                    [w["mean"]-w.lower,w.upper-w["mean"]],fmt=markers[j],
                    ms=3.3,capsize=1.6,elinewidth=1,color=color,
                    mec=C["gray"] if j==2 else color,mew=.45)
    ax.set_xticks(range(5),["0–6.4","6.4–16","16–32","32–48","48–64"])
    ax.set(xlabel="Phase (thousand calls)",ylabel="Entries / 1,000 calls",ylim=(0,104),yticks=[0,50,100])
    title(ax,"c","Archive entry yield")
    style(ax,False)
    ax=fig.add_axes([.09,.095,.81,.155])
    rho=csv("search/search_critic_window_rho_summary.csv")
    lab=csv("search/search_critic_window_labels_summary.csv")
    rho.window/=1000;lab.window/=1000
    ax.axvspan(.064,1.92,color="#F2F4F5",zorder=0)
    band(ax,rho,"window",C["arre"])
    ax.axhline(0,lw=.6,color=C["gray"])
    ax.text(.078,.1,"Warm-up (30 iterations)",fontsize=6,color=C["gray"])
    ax.set(xlabel="Docking calls (thousands; log scale)",ylabel="Spearman ρ",xlim=(.064,64),ylim=(-.05,1.02),
           yticks=[0,.5,1],xscale="log")
    ax.set_xticks([.064,.256,1,4,16,64],["0.064","0.256","1","4","16","64"])
    other=ax.twinx();other.plot(lab.window,lab["mean"]/1000,"--",color=C["gray"],lw=1)
    other.fill_between(lab.window,lab.lower/1000,lab.upper/1000,color=C["gray"],alpha=.10,lw=0)
    other.set(ylim=(0,60),yticks=[0,25,50])
    other.set_ylabel("Labels (thousands)",fontsize=6);other.tick_params(labelsize=6)
    title(ax,"d","Critic predictions before the incoming batch update")
    save(fig,"Figure_03")
    source(3,"a",["search/search_route.json"],"seven vector structures drawn from archived SMILES; original actions, zero-based iteration and score")
    source(3,"b",["search/search_allocation_summary.csv"],"15 targets × 3 seeds; archived pointwise target-bootstrap 95% intervals")
    source(3,"c",["search/search_yield_summary.csv"],"phase- and arm-specific target counts; archived pointwise target-bootstrap 95% intervals")
    source(3,"d",["search/search_critic_window_rho_summary.csv","search/search_critic_window_labels_summary.csv"],"15 targets × 3 seeds; archived pointwise target-bootstrap 95% intervals; predictive association")


def pocket():
    fig=plt.figure(figsize=(178/25.4,150/25.4))
    outer=fig.add_gridspec(3,4,left=.025,right=.97,top=.917,bottom=.055,
                           width_ratios=[1,1,1,.96],wspace=.23,hspace=.35)
    molecules=csv("search/pocket_molecules.csv");seeds=csv("search/pocket_seeds.csv");fp=csv("search/pocket_fingerprints.csv")
    effects=csv("search/pocket_effects.csv").set_index("target")
    render_manifest=js("pocket_render/render_manifest.json")
    renders={r["id"]:r for r in render_manifest["panels"]}
    review={p["id"]:p for p in js("pocket_review/pose_review.json")["panels"]}
    colors={"B":C["parent"],"ARRE":C["arre"],"reference":C["control"]}
    # Positions use the common cropped view, so the same residue sits in the
    # same place in all three columns for a target.
    label_positions={
        "A_ASP219":(.52,.94), "A_PHE193":(.84,.51), "A_SER275":(.40,.085),
        "A_VAL242":(.165,.84), "B_TYR18":(.85,.865),
        "B_SER212":(.18,.91), "B_VAL127":(.83,.285),
        "A_MET156":(.175,.14),
    }
    label_groups=[]
    for col,label in enumerate(("CGFlow", "CGFlow + ARRE", "Crystal reference", "Contact recovery")):
        box=outer[0,col].get_position(fig)
        fig.text((box.x0+box.x1)/2,.967,label,ha="center",weight="bold",fontsize=7)
    for row,target in enumerate(["NAMPT","MEK1","ROCK1"]):
        box=outer[row,0].get_position(fig)
        fig.text(box.x0,box.y1+.02,f"{chr(97+row)}  {target}",weight="bold",fontsize=8)
        ref_box=outer[row,2].get_position(fig)
        fig.text(ref_box.x0,box.y1+.02,f"PDB {review[f'{target}_reference']['pdb_id']}",
                 fontsize=6.2)
        for col,method in enumerate(["B","ARRE","reference"]):
            gs=outer[row,col].subgridspec(2,1,height_ratios=[1,.12],hspace=.11)
            ax=fig.add_subplot(gs[0,0])
            receipt=renders[f"{target}_{method}"]
            crop=receipt["crop_pixels"]
            width,height=receipt["image_size"]
            with Image.open(DATA/"pocket_render"/receipt["png"]) as im:
                ax.imshow(im.crop(crop),interpolation="none")
            labels=[]
            for item in receipt["projected_residue_labels"]:
                x=(item["x"]*width-crop[0])/(crop[2]-crop[0])
                y=(crop[3]-(1-item["y"])*height)/(crop[3]-crop[1])
                residue=item["residue"]
                chain,name=residue.split("_")
                assert 0<=x<=1 and 0<=y<=1
                labels.append(ax.annotate(f"{chain}:{name.title()}",xy=(x,y),
                    xytext=label_positions[residue],xycoords="axes fraction",
                    textcoords="axes fraction",fontsize=6.2,ha="center",va="center",
                    bbox=dict(boxstyle="round,pad=.14",fc="white",ec="none",alpha=.94),
                    arrowprops=dict(arrowstyle="-",lw=.53,color=C["gray"],shrinkA=2,shrinkB=1)))
            choice=review[receipt["id"]]
            if choice.get("rank",1)>1:
                labels.append(ax.text(.83,.92,f"Rank {choice['rank']}",ha="center",va="center",
                    fontsize=6.2,weight="bold",color=C["arre"],transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=.18",fc="white",ec=C["arre"],lw=.5)))
            label_groups.append((ax,labels))
            ax.set_axis_off()
            if method=="reference":
                f=fp[(fp.target==target)&(fp.method==method)].sort_values("rule")["matched"].tolist()
            else:
                f=[r["matched"] for r in choice["rule_details"]]
            bx=fig.add_subplot(gs[1,0]);bx.axis("off")
            for j,matched in enumerate(f):
                bx.text(.11+j*.19,.45,f"R{j+1}",fontsize=5.8,ha="center",va="center",
                        color="white" if matched and method=="ARRE" else C["ink"],
                        bbox=dict(boxstyle="square,pad=.19",fc=colors[method] if matched else "white",
                                  ec=C["gray"],lw=.6))
            bx.text(.97,.45,f"{sum(f)}/{len(f)}",ha="right",va="center",fontsize=6.4)
        rg=outer[row,3].subgridspec(2,1,hspace=.52)
        ax=fig.add_subplot(rg[0,0])
        for method in ("B","ARRE"):
            vals=np.sort(molecules[(molecules.target==target)&(molecules.method==method)].matched_rate_all.to_numpy())*100
            ax.step(np.r_[0,vals,100],np.r_[0,np.arange(1,len(vals)+1)/len(vals),1],
                    where="post",color=colors[method],ls="--" if method=="B" else "-")
        ax.set(xlim=(-2,102),ylim=(-.02,1.04),yticks=[0,1],xticks=[0,50,100])
        ax.set_title(f"ECDF · Δ {effects.loc[target,'delta_pp']:+.2f} pp",fontsize=6.5,pad=2)
        ax=fig.add_subplot(rg[1,0])
        for j,method in enumerate(["B","ARRE"]):
            vals=seeds[(seeds.target==target)&(seeds.method==method)].sort_values("seed").mean_key_all.to_numpy()*100
            ax.scatter(j+np.array([-.1,0,.1]),vals,s=17,marker="D",color=colors[method])
            ax.plot([j-.23,j+.23],[vals.mean()]*2,color=colors[method],lw=1.4)
        ax.set(ylim=(0,100),xticks=[0,1],xticklabels=["CGFlow","CGFlow + ARRE"],yticks=[0,50,100])
        ax.set_title("Seed mean (%)",fontsize=6,pad=1,weight="normal")
        ax.grid(axis="y",color=C["grid"],lw=.5)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax,labels in label_groups:
        extents=[Text.get_window_extent(label,renderer).padded(1.5) for label in labels]
        for i,extent in enumerate(extents):
            assert ax.bbox.contains(extent.x0,extent.y0) and ax.bbox.contains(extent.x1,extent.y1), labels[i].get_text()
            assert not any(extent.overlaps(other) for other in extents[:i]), labels[i].get_text()
    save(fig,"Figure_04")
    for panel,target in zip("abc",("NAMPT","MEK1","ROCK1")):
        source(4,panel,["pocket_render/render_manifest.json",
                        f"pocket_render/{render_manifest['contact_geometry_file']}",
                        "pocket_review/pose_review.json",
                        *[f"pocket_render/{renders[f'{target}_{m}']['png']}" for m in ("B","ARRE","reference")],
                        "search/pocket_molecules.csv","search/pocket_seeds.csv",
                        "search/pocket_fingerprints.csv","search/pocket_effects.csv"],
               "300 candidates per method and target; 3 seeds; separate rank-based pose display screen; common camera and crop")


def comparison_plate(number):
    qa, sources = render_comparison(number, FIG, PROJECT / "plate_qa")
    QA.append(qa)
    PANEL.extend(sources)


def mechanism():
    comparison_plate(5)


def transfer():
    comparison_plate(6)


def properties():
    fig=plt.figure(figsize=(178/25.4,132/25.4))
    axes=np.array([[fig.add_axes([x,y,w,h]) for x,w in [(.084,.390),(.569,.400)]]
                   for y,h in [(.725,.223),(.420,.181),(.106,.175)]])
    q=csv("property/source_data_size_target.csv")
    names=["CGFlow-B","ARRE","ARRE frozen","SynGA","SynGBO"]
    colors=[C["parent"],C["arre"],C["arre"],C["control"],C["contrast"]]
    ax=axes[0,0]
    for i,(name,col,marker) in enumerate(zip(names,colors,["o","D","s","v","^"])):
        vals=q[q.method==name].heavy_atom_mean.to_numpy()
        ax.scatter(i+np.linspace(-.17,.17,len(vals)),vals,s=10,color=col,alpha=.65,marker=marker)
        ax.scatter(i,np.mean(vals),s=26,color=col,marker="D",edgecolor=C["ink"],lw=.55)
    ax.set(xticks=range(5),xticklabels=["CGFlow","CGFlow +\nARRE","ARRE\n(frozen policy)","SynGA","SynGBO"],
           ylabel="Mean heavy atoms",xlim=(-.49,4.49),yticks=[26,30,34,38])
    title(ax,"a","Molecular-size distribution");style(ax,False)
    ax=axes[0,1];q=csv("property/source_data_correction_summary.csv")
    for dx,opp,col,marker in [(-.15,"CGFlow-B",C["arre"],"o"),(.15,"SynGBO",C["contrast"],"s")]:
        for i,est in enumerate(["raw","R1","ANCOVA"]):
            z=q[(q.opponent==opp)&(q.estimator==est)].iloc[0]
            vertical(ax,i+dx,dict(mean=z.mean_difference,ci95=[z.ci95_low,z.ci95_high]),col,marker)
    null(ax)
    ax.set(xticks=range(3),xticklabels=["Raw","R1","ANCOVA"],xlim=(-.52,2.52),
           ylim=(-1.39,.2),yticks=[-1.2,-.8,-.4,0],
           ylabel="ARRE − comparator Vina\n(kcal mol$^{-1}$)")
    ax.legend(handles=[Line2D([],[],marker=m,ls="",color=c,label=l) for m,c,l in
                       [("o",C["arre"],"vs CGFlow"),("s",C["contrast"],"vs SynGBO")]],
              loc="upper left",ncol=2,handletextpad=.25,columnspacing=.8,borderaxespad=.2)
    style(ax,False)
    title(ax,"b","Distinct size-adjusted estimands")
    q=csv("property/source_data_threshold_summary.csv")
    for pos,metric,letter,ttl,ylabel in [(0,"DS_percent","c","Joint docking + planning","Joint success (%)"),
                                        (1,"C_percent","d","Conditional planning","Conditional success (%)")]:
        ax=axes[1,pos]
        for name,col,marker in zip(names,colors,["o","D","s","v","^"]):
            z=q[(q.method==name)&(q.metric==metric)].sort_values("threshold_q")
            ax.plot(z.threshold_q*100,z["mean"],color=col,marker=marker,ms=3,label=name.replace("CGFlow-B","Parent").replace("ARRE frozen","ARRE (frozen policy)"))
            ax.fill_between(z.threshold_q*100,z.ci95_low,z.ci95_high,color=col,alpha=.10,lw=0)
        ax.set(xlabel="Baseline docking percentile",ylabel=ylabel,ylim=(0,105))
        title(ax,letter,ttl);style(ax,False)
        if pos==0:
            handles,legend_labels=ax.get_legend_handles_labels()
    fig.legend(handles,[{"Parent":"CGFlow","ARRE":"CGFlow + ARRE"}.get(label,label)
                        for label in legend_labels],
               ncol=5,fontsize=6,loc="center",bbox_to_anchor=(.53,.327),
               handletextpad=.35,columnspacing=1.2)
    z=js("extension_data.json")["Figure7"]
    ax=axes[2,0]
    ax.scatter(z["fc_minus_synGBO_vina"],[0]*5,s=10,color=C["arre"],zorder=4)
    point(ax,z["mean_delta_vina"],1,z["ci95_delta_vina"],C["ink"],"D",size=4.8)
    null(ax,horizontal=False)
    ax.set(yticks=[0,1],yticklabels=["Five tasks","Mean"],ylim=(1.58,-.58),
           xlabel="Capped ARRE − SynGBO Vina (kcal mol$^{-1}$)",
           xlim=(-.85,.68),xticks=[-.8,-.4,0,.4])
    title(ax,"e","Generation cap: ≤30 heavy atoms");style(ax)
    ax=axes[2,1]
    for name,x,y in zip(z["target"],z["fc_minus_synGBO_heavy_atoms"],z["fc_minus_synGBO_vina"]):
        ax.scatter(x,y,s=20,color=C["arre"])
        offsets={"FEN1":(4,-1),"MAPK1":(-4,4)}
        ax.annotate(name,(x,y),xytext=offsets.get(name,(4,4)),textcoords="offset points",
                    ha="right" if name=="MAPK1" else "left",fontsize=6)
    ax.axhline(0,color=C["gray"],ls=":",lw=.7);ax.axvline(0,color=C["gray"],ls=":",lw=.7)
    ax.set(xlim=(-3.1,1.2),ylim=(-.85,.68),xlabel="Heavy-atom difference",
           ylabel="Vina difference\n(kcal mol$^{-1}$)",yticks=[-.8,-.4,0,.4])
    title(ax,"f","Size-capped target contrasts")
    save(fig,"Figure_07")
    source(7,"a",["property/source_data_size_target.csv"],"15 target-level means, 3 seeds")
    source(7,"b",["property/source_data_correction_summary.csv","property/source_data_correction_target.csv"],"original 95% target-bootstrap intervals; raw/R1/ANCOVA are separate estimands")
    source(7,"cd",["property/source_data_threshold_summary.csv","property/source_data_target_threshold_long.csv"],"pointwise target-bootstrap 95% intervals, 15 targets; multiple thresholds exploratory")
    source(7,"ef",["sizecap.json","extension_data.json"],"five target means × 3 seeds, nominal paired t interval; prespecified size cap")


def validation():
    comparison_plate(8)


def supplementary():
    fig,axes=newfig(120,rows=2,left=.173,wspace=.68,hspace=.53)
    q=csv("legacy/ablation_target_summary.csv")
    tags=["r4CE4","r4CE4L1","r2CE4L2","r2CE4L4","r4CE4L8"]
    for j,(metric,ylabel) in enumerate([("vina","Mean Vina (kcal mol$^{-1}$)"),("aiz","AiZynth solved (%)")]):
        ax=axes[0,j]
        for k,tag in zip([0,1,2,4,8],tags):
            vals=q[(q.tag==tag)&(q.metric==metric)]["mean"]
            lo,hi=boot(vals);m=vals.mean()
            ax.errorbar(k,m,yerr=[[m-lo],[hi-m]],fmt="D",color=C["planning"],ms=4,capsize=2)
        ax.set(xticks=[0,1,2,4,8],xlabel="SPP strength κ",ylabel=ylabel)
        title(ax,chr(97+j),"Planning-prior sweep");style(ax,False)
        ax=axes[1,j]
        for i,(tag,label,col) in enumerate([("r4B0","Parent frozen",C["parent"]),("base_ep30","Parent learning",C["parent"]),("r4CE0","ARRE (frozen policy)",C["arre"]),("r2CE4L4","Standard ARRE",C["arre"])]):
            vals=q[(q.tag==tag)&(q.metric==metric)]["mean"];lo,hi=boot(vals)
            point(ax,vals.mean(),i,[lo,hi],col,open_="frozen" in label)
        ax.set(yticks=range(4),yticklabels=["Parent frozen","Parent learning","ARRE (frozen policy)","Standard ARRE"],ylim=(3.6,-.6),xlabel=ylabel)
        title(ax,chr(99+j),"Generator-policy intervention");style(ax)
    save(fig,"Figure_S01")
    source("S1","abcd",["legacy/ablation_target_summary.csv"],"15 target means × 3 seeds; percentile bootstrap 95% intervals; descriptive sweep and generator controls")
    fig,axes=newfig(145,rows=2,left=.22,wspace=.96,hspace=.55)
    fb=js("feedback_summary.json")
    for j,(metric,xlabel) in enumerate([("vina","Control − online Vina (kcal mol$^{-1}$)"),("aizynth","Control − online AiZynth (pp)")]):
        entries=[];labels=[];colors=[]
        for cohort,label in [("P1","Pilot 5"),("P2","Extension 10"),("pooled15","Pooled 15")]:
            for arm,al in [("Fr","critic frozen"),("Fx","fixed allocation")]:
                entries.append(fb[cohort][metric][arm+"-On"]);labels.append(label+" · "+al)
                colors.append(C["contrast"] if arm=="Fr" else C["control"])
        forest(axes[0,j],entries,labels,colors,xlabel)
        title(axes[0,j],chr(97+j),"Feedback docking" if metric=="vina" else "Feedback planning")
    target=csv("feedback_target_means.csv")
    for j,arm in enumerate(["Fr","Fx"]):
        piv=target.pivot(index="target",columns="arm",values="vina")
        vals=piv[arm]-piv.On
        ax=axes[1,j];ax.scatter(vals,range(15),s=13,color=C["contrast"] if arm=="Fr" else C["control"])
        ax.axvline(0,color=C["gray"],ls=":",lw=.7)
        ax.set(yticks=range(15),yticklabels=vals.index,ylim=(14.6,-.6),xlabel="Target-mean control − online Vina")
        ax.tick_params(axis="y",labelsize=5.5)
        title(ax,chr(99+j),"Frozen critic" if arm=="Fr" else "Fixed allocation");style(ax)
    save(fig,"Figure_S02")
    source("S2","abcd",["feedback_runs.csv","feedback_target_means.csv","feedback_summary.json"],"135 complete branch runs; exploratory paired t intervals by cohort; complete budget and checkpoint audit")
    p=DATA/"legacy/fig4_transfer_targets.csv"
    if p.exists():
        q=pd.read_csv(p)
        fig,axes=newfig(145,rows=3,left=.19,wspace=.65,hspace=.63)
        for ax,(substrate,metric,letter) in zip(axes.flat,[("New pockets","vina","a"),("New pockets","aiz","b"),("RxnFlow full","vina","c"),("RxnFlow E+P+LA","vina","d"),("PMO","auc_top10","e"),("SynFlowNet","auc_top10","f")]):
            z=q[(q.substrate==substrate)&(q.metric==metric)]
            if z.empty:
                ax.axis("off");continue
            ax.scatter(z.delta,range(len(z)),color=C["planning"] if metric=="aiz" else C["arre"],s=15)
            ax.axvline(0,color=C["gray"],ls=":",lw=.7)
            xlabel={"vina":"Candidate − baseline Vina (kcal mol$^{-1}$)",
                    "aiz":"AiZynth gain (pp)","auc_top10":"Candidate − baseline AUC (top-10)"}[metric]
            ax.set(yticks=range(len(z)),yticklabels=z.target,ylim=(len(z)-.4,-.7),xlabel=xlabel)
            title(ax,letter,substrate);style(ax)
        save(fig,"Figure_S03")
        source("S3","abcdef",["legacy/fig4_transfer_targets.csv","legacy/transfer_runs.csv"],"earlier exploratory transfer, retained without adding independent confirmatory pocket counts")


def toc():
    fig=plt.figure(figsize=(83/25.4,36/25.4))
    ax=fig.add_axes([0,0,1,1]);ax.axis("off")
    # Use actual archived branch chemistry, with readable strategy-level labels.
    r=js("search/search_route.json")
    molecule(ax.inset_axes([.02,.23,.25,.65]),r["child"][2]["smiles"])
    molecule(ax.inset_axes([.73,.23,.25,.65]),r["child_smiles"])
    for x in [.275,.685]:
        ax.add_patch(FancyArrowPatch((x,.56),(x+.04,.56),arrowstyle="->",mutation_scale=9,color=C["arre"],lw=1.3))
    for y,text,col in [(.73,"Productive prefix",C["arre"]),(.54,"Reactant feedback",C["contrast"]),(.35,"Planning prior",C["planning"])]:
        ax.text(.5,y,text,ha="center",va="center",fontsize=8,color=col,weight="bold")
    ax.text(.15,.12,"Shared prefix",ha="center",fontsize=6.5)
    ax.text(.85,.12,"New completion",ha="center",fontsize=6.5)
    save(fig,"Graphical_Abstract")
    source("ToC","all",["search/search_route.json"],"authentic archived shared-prefix chemistry, schematic strategy labels")


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--pocket-only",action="store_true")
    group.add_argument("--layout-revision",action="store_true")
    group.add_argument("--narrative-revision",action="store_true")
    args=parser.parse_args()
    qa_path=PROJECT/"figure_qa.json"
    panel_path=PROJECT/"panel_sources.json"
    if args.layout_revision or args.narrative_revision:
        revised={2,3} if args.narrative_revision else {2,5,6,7,8}
        QA.extend(q for q in json.loads(qa_path.read_text())
                  if q["figure"] not in {f"Figure_{i:02d}" for i in revised})
        PANEL.extend(p for p in json.loads(panel_path.read_text()) if p["figure"] not in revised)
        benchmark()
        if args.narrative_revision:
            search()
        else:
            mechanism();transfer();properties();validation()
    elif args.pocket_only:
        QA.extend(q for q in json.loads(qa_path.read_text()) if q["figure"]!="Figure_04")
        PANEL.extend(p for p in json.loads(panel_path.read_text()) if p["figure"]!=4)
        pocket()
    else:
        method_figure();benchmark();search();pocket();mechanism();transfer();properties();validation();supplementary();toc()
    qa_path.write_text(json.dumps(QA,indent=2))
    panel_path.write_text(json.dumps(PANEL,indent=2))


if __name__=="__main__":
    main()
