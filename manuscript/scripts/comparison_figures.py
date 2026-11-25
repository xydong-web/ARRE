"""Render the three comparison figures from the archived target-level analyses."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from figure_qc.audit_panel_alignment import require_matplotlib_panel_alignment
from display_names import ABLATIONS
from chem_art import molecule, pocket_views

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE if (HERE / "data").exists() else HERE / "source"
DATA = PROJECT / "data"
BLUE, LIGHT, GREEN, CORAL = "#466F87", "#78B6C8", "#748720", "#B46255"
INK, GRAY, RULE = "#26343B", "#596970", "#D7E1E4"
WIDTH_MM = 178
STYLE = {"font.family":"sans-serif", "font.sans-serif":["Liberation Sans","DejaVu Sans"],
         "image.composite_image":False, "svg.hashsalt":"arre-comparison-v1", "font.size":8, "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none",
         "figure.facecolor":"white", "savefig.facecolor":"white"}

def read(name):
    return json.loads((DATA / name).read_text())

def fmt(v,d=2):
    return (f"{v:+.{d}f}" if abs(v) >= .5*10**(-d) else f"{0:.{d}f}").replace("-","−")

def bounds(interval,d=2):
    return "["+", ".join(fmt(n,d) for n in interval)+"]"

def label(ax,x,y,txt,size=7.3,ha="left",color=INK,bold=False,**kwargs):
    return ax.text(x,y,txt,fontsize=size,ha=ha,va="center",color=color,
                   weight="bold" if bold else "normal",**kwargs)

def cell(ax,left,right,bottom,top,fill):
    ax.add_patch(Rectangle((left,bottom),right-left,top-bottom,facecolor=fill,edgecolor="none",zorder=0))

def panel(fig,slot,letter,title):
    ax=fig.add_subplot(slot,label=letter)
    ax.set(xlim=(0,1),ylim=(0,1)); ax.set_axis_off()
    label(ax,0,.962,letter,10,bold=True)
    label(ax,.058 if slot.colspan.stop-slot.colspan.start==1 else .031,.962,title,9,bold=True)
    return ax

def grid(height,ratios):
    fig=plt.figure(figsize=(WIDTH_MM/25.4,height/25.4))
    gs=fig.add_gridspec(len(ratios),2,left=.025,right=.975,bottom=.023,top=.982,
                        wspace=.12,hspace=.10,height_ratios=ratios)
    return fig,gs

def tint(value,limit,pos=BLUE):
    w=.56*min(abs(value)/limit,1)
    return (1-w)*np.array(to_rgb("#f4f6f6"))+w*np.array(to_rgb(pos if value>=0 else CORAL))

def scale(value,limits,left,right):
    return left+(right-left)*(value-limits[0])/(limits[1]-limits[0])

def ruler(ax,limits,left,right,base,top,ticks,reference=0):
    x=scale(reference,limits,left,right)
    ax.plot([x,x],[base,top],":",lw=.85,c=GRAY,zorder=1)
    ax.plot([left,right],[base,base],color=RULE,lw=.7)
    for tick in ticks:
        x=scale(tick,limits,left,right)
        ax.plot([x,x],[base,base-.014],color=GRAY,lw=.6)
        label(ax,x,base-.053,f"{tick:g}".replace("-","−"),6.6,"center",GRAY)

def interval(ax,mean,ci,y,limits,left,right,color=BLUE,marker="D"):
    assert limits[0]<=ci[0]<=mean<=ci[1]<=limits[1],(mean,ci,limits)
    x,a,b=[scale(v,limits,left,right) for v in [mean,*ci]]
    ax.errorbar(x,y,xerr=[[x-a],[b-x]],fmt=marker,ms=3.6,color=color,
                elinewidth=1.5,capsize=2,capthick=1)

def number_pair(ax,x,y,q,color=INK,d=2):
    label(ax,x,y+.018,fmt(q["mean"],d),8.2,"center",color,True)
    label(ax,x,y-.027,bounds(q["ci95"],d),6.65,"center",INK)

def grouped_gains(ax,metric,color,limits,ticks):
    cells={r["bits"]:r for r in read("factorial_summary.json")}
    order=["000","001","100","101","010","011","110","111"]
    pairs=np.array([.18,.40,.62,.84])
    y=lambda val:scale(val,limits,.21,.765)
    for tick in ticks:
        yy=y(tick)
        if metric != "docking_gain" or tick < 1.5:
            ax.plot([.09,.98],[yy,yy],color=GRAY if tick==0 else RULE,lw=.65)
        label(ax,.072,yy,str(tick).replace("-","−"),6.7,"right",GRAY)
    for j,key in enumerate(order):
        q=cells[key][metric]; assert q["n"]==15
        x=pairs[j//2]+(.039 if j%2 else -.039)
        if key=="000":
            ax.plot([x-.032,x+.032],[y(0)]*2,color=color,lw=1.5)
        else:
            m=q["mean"];lo,hi=q["ci95"];yy=y(m)
            ax.bar(x,yy-y(0),width=.064,bottom=y(0),color=color if j%2 else "white",edgecolor=color,lw=.95)
            ax.errorbar(x,yy,yerr=[[yy-y(lo)],[y(hi)-yy]],color=INK,fmt="none",elinewidth=.77,capsize=1.7)
        if key in (("100","111") if metric=="docking_gain" else ("001","111")):
            label(ax,x,y(q['ci95'][1])+.056,fmt(q['mean']),6.7,"center",color,True)
    for x,name in zip(pairs,["Neither","RHC","RFG","Both"]):label(ax,x,.115,name,7,"center")
    label(ax,.51,.015,"RHC / RFG background",6.7,"center",GRAY)

def component_plate():
    fig,gs=grid(166,[1.3,1.0,.98])
    a=panel(fig,gs[0,:],"a","Reaction-history continuation at a shared ADRB2 intermediate")
    root=DATA/"medicinal_figures"
    molecule(a,root/"prefix.svg",(.008,.283,.292,.459))
    for key,y0,caption in [("parent",.525,"Earlier completion"),("child",.055,"RHC continuation")]:
        molecule(a,root/(key+".svg"),(.479,y0,.34,.39))
        label(a,.837,y0+.20,caption,7.25,bold=True)
    for yy in [.716,.246]:
        a.annotate("",xy=(.467,yy),xytext=(.389,yy),arrowprops={"arrowstyle":"-|>","mutation_scale":8,"lw":1,"color":GRAY})
    a.plot([.314,.390],[.477,.477],lw=.9,c=GRAY);a.plot([.390,.390],[.246,.716],lw=.9,c=GRAY)
    label(a,.15,.149,"Shared two-action prefix",7.1,"center",GRAY)
    b=panel(fig,gs[1,0],"b","Top-100 Vina gain (kcal/mol)")
    c=panel(fig,gs[1,1],"c","AiZynthFinder solved-fraction gain (pp)")
    grouped_gains(b,"docking_gain",BLUE,(-.15,1.58),[0,.5,1,1.5])
    grouped_gains(c,"planner_gain",GREEN,(-19,21),[-15,0,15])
    for ax in (b,c):
        label(ax,.52,.837,"Open = SPP off     Filled = SPP on",6.65,"center",GRAY)
    b=panel(fig,gs[2,0],"d","Full ARRE versus component removals")
    data=pd.read_csv(DATA/"legacy/ablation_display_effects.csv")
    names=["CGFlow","Without RHC","Without RFG","Without SPP¹","No look-ahead"]
    for j,(metric,ttl,lim,ticks,l,r) in enumerate([
        ("vina","Vina (kcal/mol)",(-.35,1.40),[0,.6,1.2],.36,.62),
        ("aiz","Solved (pp)",(-8,32),[0,15,30],.72,.985)]):
        label(b,(l+r)/2,.811,ttl,7.1,"center",bold=True)
        ruler(b,lim,l,r,.215,.75,ticks)
        for i,(tag,name) in enumerate(zip(["base_ep30"]+[v[0] for v in ABLATIONS],names)):
            sub=data[(data.tag=="r2CE4L4")&(data.reference_tag==tag)&(data.metric==metric)&
                     (data.panel==("a" if i==0 else "bc"))]
            assert len(sub)==1
            row=sub.iloc[0];y=.70-i*.097
            if j==0:label(b,.001,y,name,7.05)
            interval(b,row["mean"],[row.lo,row.hi],y,lim,l,r,BLUE if j==0 else GREEN)
    label(b,.01,.053,"¹ Removes the prior and look-ahead.",6.7,color=GRAY)
    c=panel(fig,gs[2,1],"e","Alternative search controls")
    ctrl=read("controls.json")["contrasts"]["m1_adj"]
    keys=["R-B","P-B","E-B","C-B","C-E"]
    names=["Random prefix","Prioritized replay","Elite last block","RHC","RHC − elite last block"]
    lim=(-1.5,.30); ruler(c,lim,.45,.975,.215,.78,[-1.2,-.6,0])
    for i,(key,name) in enumerate(zip(keys,names)):
        q=ctrl[key];y=.72-i*.10
        label(c,0,y,name,7.0)
        interval(c,q["mean"],q["ci95"],y,lim,.45,.975,BLUE if i>2 else GRAY)
    label(c,.70,.044,"Adjusted Vina difference (kcal/mol)",6.8,"center")
    return fig

def cohort_intervals(ax,rows,limits,ticks,color):
    left,right=.55,.96
    ruler(ax,limits,left,right,.18,.81,ticks)
    for i,(name,mean,ci) in enumerate(rows):
        y=.716-i*.109
        label(ax,.004,y,name,6.85)
        interval(ax,mean,ci,y,limits,left,right,color)

def transfer_plate():
    fig,gs=grid(158,[1.05,1.12,1.03])
    a=panel(fig,gs[0,0],"a","Docking superiority")
    b=panel(fig,gs[0,1],"b","Independent planning non-inferiority")
    tr=read("trial.json"); rows=[tr["P1_m1_adj_R1"][k] for k in ("F-B","F-SynGBO","FZ-SynGBO")]
    lim=(-1.2,.10)
    ruler(a,lim,.52,.97,.255,.82,[-1,-.5,0])
    for i,(q,t) in enumerate(zip(rows,["A   ARRE vs CGFlow","B   ARRE vs SynGBO","C   Policy-frozen ARRE\n      vs SynGBO"])):
        y=.76-i*.174;label(a,0,y,t,7.15);interval(a,q["mean"],q["ci95"],y,lim,.52,.97)
    label(a,.68,.095,"Adjusted Vina difference (kcal/mol)",6.8,"center")
    ni=tr["P2"]["FZ_vs_SynGBO_noninferiority"];lim=(-6,4)
    left,right=.187,.97; margin=-abs(tr["constants"]["margin_pp"]);xm=scale(margin,lim,left,right)
    cell(b,left,xm,.255,.765,"#F5E6E1");b.plot([xm,xm],[.255,.765],ls="--",color=CORAL,lw=1)
    ruler(b,lim,left,right,.255,.765,[-5,0,3])
    for y,key,txt,co in [(.68,"ci90","90% CI",GREEN),(.51,"ci95","95% CI",GRAY)]:
        label(b,0,y,txt,7.0); interval(b,ni["mean"],ni[key],y,lim,left,right,co)
    label(b,.51,.825,f"Policy-frozen ARRE vs SynGBO   ·   {fmt(ni['mean'])} pp",7.0,"center")
    label(b,.55,.095,"Difference in Retro* success (pp)",6.8,"center")
    label(b,.50,.008,f"Margin {fmt(margin,0)} pp   ·   90% lower bound {fmt(ni['ci90'][0],3)} pp",6.75,"center",GREEN,True)
    c=panel(fig,gs[1,:],"c","Policy-frozen ARRE versus SynGBO across 20 held-out pockets")
    q=rows[2];pockets=np.asarray(q["pockets"]);vals=np.asarray(q["diffs"])
    assert len(set(pockets))==20 and (vals<0).sum()==18
    order=np.argsort(vals,kind="stable")
    x=np.linspace(.09,.965,20); y=lambda v:scale(v,(-.89,.25),.28,.79)
    for tick in [-.8,-.4,0]:
        c.plot([.06,.988],[y(tick)]*2,c=GRAY if tick==0 else RULE,lw=.65)
        label(c,.048,y(tick),str(tick).replace("-","−"),6.65,"right",GRAY)
    for xx,i in zip(x,order):
        v=vals[i]; c.bar(xx,y(v)-y(0),bottom=y(0),width=.035,color=BLUE if v<0 else CORAL,lw=0)
        label(c,xx,.208,pockets[i].split("_")[0],6.4,"right",GRAY,rotation=49,rotation_mode="anchor")
    label(c,.08,.038,"Adjusted Vina difference (kcal/mol)",6.85,color=GRAY)
    label(c,.975,.879,"18 of 20 pockets favor policy-frozen ARRE",7.05,"right",BLUE,True)
    d=panel(fig,gs[2,0],"d","Docking transfer across generators")
    port=read("portability.json")
    dq=[("CGFlow · held-out (20)",rows[0]),
        ("RxnFlow · development (15)",port["dev15"]["contrasts"]["m1_adj"]["RXCE-RX"]),
        ("RxnFlow · held-out (20)",port["held20"]["contrasts"]["m1_adj"]["RXCE-RX"])]
    cohort_intervals(d,[(name,q["mean"],q["ci95"]) for name,q in dq],(-1.4,.1),[-1.2,-.6,0],BLUE)
    label(d,.005,.022,"Guided − parent, adjusted Vina (kcal/mol)",6.7,color=GRAY)
    e=panel(fig,gs[2,1],"e","Independent planning by cohort")
    full=read("planner_full.json")["contrasts"];extra=read("planner_extra.json")["contrasts"]
    qc=[("CGFlow · development (15)",full["F-B"]),("CGFlow · TarPass (20)",extra["TP_F-TP_B"]),
        ("CGFlow · new pockets (10)",extra["NP_F-NP_B"]),
        ("RxnFlow · development (15)",read("planner_dev.json")["contrasts"]["S1_RXCE-S1_RX"]),
        ("RxnFlow · held-out (20)",read("planner_held.json")["contrasts"]["S1H_RXCE-S1H_RX"])]
    cohort_intervals(e,[(name,q["independent"]["mean_pp"],q["independent"]["boot_ci_pp"]) for name,q in qc],(-3.7,8),[-3,0,3,6],GREEN)
    label(e,.005,.022,"Guided − parent, solved fraction (pp)",6.7,color=GRAY)
    return fig

def validation_plate():
    fig,gs=grid(180,[1.14,1.43,.97,.90])
    a=panel(fig,gs[0,:],"a","Independent score gains before and after redocking")
    data=pd.read_csv(DATA/"independent/statistics.csv").set_index("condition")
    effects=pd.read_csv(DATA/"independent/source_target_effects.csv")
    configs=[("Vina gain (kcal/mol)","vina_original","vina_redocked",(-.1,1.75),[0,.75,1.5]),
             ("CNNaffinity, raw","cnn_original_raw","cnn_redocked_raw",(-.30,.35),[-.2,0,.2]),
             ("CNNaffinity, size-adjusted","cnn_original_adjusted","cnn_redocked_adjusted",(-.30,.35),[-.2,0,.2]),
             ("RTMScore gain","rtm_original","rtm_redocked",(-1.1,3.6),[-1,1,3])]
    for i,(name,one,two,lim,ticks) in enumerate(configs):
        x0=.25*i;l=x0+.084;r=x0+.216
        label(a,x0+.138,.803,name,7.1,"center",bold=True)
        y=lambda v:scale(v,lim,.225,.724)
        for tick in ticks:
            a.plot([l-.008,r+.008],[y(tick)]*2,c=GRAY if tick==0 else RULE,lw=.55,ls=":" if tick==0 else "-")
            label(a,l-.017,y(tick),f"{tick:g}".replace("-","−"),6.45,"right",GRAY)
        raw=effects[effects.condition.isin([one,two])].pivot(index="target",columns="condition",values="effect")[[one,two]]
        assert len(raw)==15 and not raw.isnull().values.any()
        for v,w in raw.to_numpy():a.plot([l,r],[y(v),y(w)],lw=.57,c="#B6C8CC")
        co=CORAL if i==2 else BLUE
        means=data.loc[[one,two],"mean"].values;assert np.allclose(means,raw.mean().values)
        a.plot([l,r],[y(v) for v in means],c=co,lw=1.8)
        for x,k in [(l,one),(r,two)]:
            z=data.loc[k];m=y(z["mean"])
            a.errorbar(x,m,yerr=[[m-y(z.lo)],[y(z.hi)-m]],fmt="o",ms=3,color=co,elinewidth=.9,capsize=2)
        label(a,l,.13,"Original",6.5,"center");label(a,r,.13,"Redocked",6.5,"center")
    b=panel(fig,gs[1,0],"b","CNNaffinity change by pose displacement")
    raw=pd.read_csv(DATA/"independent/source_pose_strata.csv")
    groups=[("<=2 A","≤2 Å"),("2-5 A",">2–5 Å"),(">5 A",">5 Å")]
    lim=(-.46,.14); yy=lambda v:scale(v,lim,.20,.79)
    for tick in [-.4,-.2,0,.1]:
        b.plot([.14,.98],[yy(tick)]*2,c=GRAY if tick==0 else RULE,lw=.65,ls=":" if tick==0 else "-")
        label(b,.125,yy(tick),f"{tick:g}".replace("-","−"),6.65,"right",GRAY)
    for xx,(skey,name) in zip([.29,.57,.85],groups):
        values=np.sort(raw[raw.stratum==skey].effect.values)
        assert len(values)==15 and min(values)>lim[0] and max(values)<lim[1],(skey,min(values),max(values))
        for n,v in enumerate(values):
            b.plot(xx+((n%5)-2)*.016,yy(v),marker="o",ms=2.0,color="#AEBFC1",ls="none")
        z=data.loc["pose-shift:"+skey];assert np.isclose(z["mean"],values.mean())
        mean=yy(z["mean"])
        b.errorbar(xx,mean,yerr=[[mean-yy(z.lo)],[yy(z.hi)-mean]],fmt="D",ms=3.5,color=CORAL,elinewidth=1.3,capsize=2.4)
        label(b,xx,.12,name,7.05,"center")
    label(b,.55,.033,"Receptor-frame heavy-atom RMSD",6.8,"center",GRAY)
    c=panel(fig,gs[1,1],"c","OPRK1 binding pocket")
    pocket_views(c, DATA/"medicinal_figures", read("medicinal_figures/pocket_view.json"))
    b=panel(fig,gs[2,:],"d","Original and alternative receptor structures")
    rec=read("xrec.json")["contrasts"];ori=rec["orig/m1/F-B"];alt=rec["alt/m1/F-B"]
    pockets=sorted(ori["per_target"]);assert set(pockets)==set(alt["per_target"]) and len(pockets)==10
    lo,hi=-1.9,.27
    y=lambda v:scale(v,(lo,hi),.24,.84)
    for tick in [-1.5,-1,-.5,0]:
        yy=y(tick);b.plot([.057,.975],[yy,yy],color=RULE if tick else GRAY,lw=.62,ls=":" if tick==0 else "-")
        label(b,.046,yy,f"{tick:g}".replace("-","−"),6.7,"right",GRAY)
    for j,p in enumerate(pockets+["Mean"]):
        x=.09+j*.087
        v1=ori["per_target"][p] if j<10 else ori["mean"]
        v2=alt["per_target"][p] if j<10 else alt["mean"]
        b.plot([x-.012,x+.012],[y(v1),y(v2)],lw=1.15,color="#8E9DA2")
        for dx,v,q,mark,face in [(-.012,v1,ori,"o","white"),(.012,v2,alt,"D",BLUE)]:
            if j==10:b.plot([x+dx]*2,[y(t) for t in q["ci95"]],lw=1.1,color=BLUE)
            b.plot(x+dx,y(v),marker=mark,mec=BLUE,mfc=face,ms=4,mew=.85)
        label(b,x,.143,p.split("_")[0],6.6,"center")
    for x,sym,fc,txt in [(.59,"o","white","Original"),(.742,"D",BLUE,"Alternative")]:
        b.plot(x,.944,marker=sym,ms=4,mec=BLUE,mfc=fc);label(b,x+.016,.944,txt,7,color=GRAY)
    label(b,.057,.015,"ARRE − parent Vina (kcal/mol) across ten source pockets and 24 alternative structures.",6.8,color=GRAY)
    c=panel(fig,gs[3,0],"e","Shared-cohort discrimination")
    lim=(.46,.675);ruler(c,lim,.33,.97,.277,.775,[.5,.55,.6,.65],reference=.5)
    for i,(k,t) in enumerate([("Vina","Vina"),("CNNaffinity","CNNaffinity"),("Heavy atoms","Heavy atoms")]):
        q=data.loc["shared_AUROC:"+k];y0=.72-i*.16
        label(c,.01,y0,t,7.35)
        interval(c,q["mean"],[q.lo,q.hi],y0,lim,.33,.97,BLUE if i<2 else GRAY)
    label(c,.65,.087,"AUROC  ·  chance = 0.50",6.9,"center")
    d=panel(fig,gs[3,1],"f","Held-out PoseBusters pass rates")
    held=read("pb_held.json")["methods"]
    for i,(back,pair) in enumerate([("CGFlow",("B","F")),("RxnFlow",("RX","RXCE"))]):
        xx=.22+i*.49
        y=lambda v:scale(v,(98.55,99.60),.28,.77)
        vals=[held[k]["pass_all_pct"] for k in pair]
        d.plot([xx-.11,xx+.11],[y(v) for v in vals],c=LIGHT,lw=1.7)
        for dx,val,full in zip([-.11,.11],vals,[False,True]):
            d.plot(xx+dx,y(val),marker="o",ms=4,mfc=BLUE if full else "white",mec=BLUE)
            label(d,xx+dx,y(val)+.103,f"{val:.2f}%",6.65,"center")
        label(d,xx,.23,back,7.2,"center")
    label(d,.5,.063,"Open = parent     Filled = guided",6.8,"center",GRAY)
    return fig

SOURCES={
5:{"a":(["factorial_summary.json","factorial_target_means.csv","factorial_runs.csv"],"Nominal 95% paired target t intervals after seed averaging; eight factorial configurations"),
   "b":(["legacy/ablation_display_effects.csv","legacy/ablation_primary_statistics.csv"],"Original nominal 95% target-bootstrap intervals; eight Holm-corrected ablation tests"),
   "c":(["controls.json"],"Five paired task means; 95% paired t intervals")},
6:{"a":(["trial.json"],"Three prespecified contrasts from the six-test Holm docking family, paired 95% t intervals"),
   "b":(["trial.json"],"Preregistered -5 pp independent-planning margin; paired 90% and 95% t intervals"),
   "c":(["trial.json","Heldout20.csv"],"Twenty named pocket effects after averaging three seeds; same contrasts as panel a"),
   "d":(["trial.json","portability.json"],"Held-out matched pockets and developmental RxnFlow comparison; paired t intervals"),
   "e":(["planner_full.json","planner_extra.json","planner_dev.json","planner_held.json"],"Independent planner, full-run pocket-bootstrap intervals")},
8:{"a":(["independent/statistics.csv","independent/source_target_effects.csv"],"Native score units; original 95% target-bootstrap intervals"),
   "b":(["xrec.json"],"Ten paired source pockets and 24 alternatives; mean and paired t intervals within receptor condition"),
   "c":(["independent/statistics.csv","independent/source_AUROC.csv"],"Shared cohort, archived 95% bootstrap intervals"),
   "d":(["pb_dev.json","pb_held.json"],"22,500 development and 24,000 held-out poses; energy-ratio check excluded")}}

SOURCES[5]={
    "a":(["search/search_route.json",*[f"medicinal_figures/{n}.svg" for n in ("prefix","parent","child")]],"Recorded two-action ADRB2 prefix and two terminal completions, drawn on a common core."),
    "b":SOURCES[5]["a"],"c":SOURCES[5]["a"],"d":SOURCES[5]["b"],"e":SOURCES[5]["c"]}
SOURCES[6]["c"]=(["trial.json","Heldout20.csv"],"Ordered policy-frozen minus SynGBO paired docking differences across 20 pockets.")
SOURCES[8]={"a":SOURCES[8]["a"],"b":(["independent/source_pose_strata.csv","independent/statistics.csv"],"Unadjusted redocking-minus-original CNNaffinity change, ARRE minus CGFlow, 15 target means and archived 95% target-bootstrap intervals."),
            "c":(["medicinal_figures/"+s for s in ("pose_case.json","eligible_pose_ids.csv","original.mol","redocked.mol","oprk1_pocket.pdb","original_pocket.png","redocked_pocket.png","pocket_view.json")],"Graph, stereo and protonation matched original and redocked coordinates of the upper median eligible ARRE candidate above 5 A; same receptor camera and scale, pose-specific side-chain proximity and no ligand-only alignment."),
            "d":SOURCES[8]["b"],"e":SOURCES[8]["c"],"f":(["pb_held.json"],"Method-level rates among 24,000 held-out poses; development-arm rates are recorded in the Supporting Information.")}

def render(number,output,qa_dir):
    output=Path(output);qa_dir=Path(qa_dir)
    output.mkdir(parents=True,exist_ok=True);qa_dir.mkdir(parents=True,exist_ok=True)
    name=f"Figure_{number:02d}"
    with mpl.rc_context({**mpl.rcParamsDefault, **STYLE}):
        fig={5:component_plate,6:transfer_plate,8:validation_plate}[number]()
        # Select the distributed typefaces explicitly so typography is stable on headless hosts.
        fonts = Path(__file__).resolve().parent / "fonts"
        for axis in fig.axes:
            for item in axis.texts:
                props = item.get_fontproperties().copy()
                face = "Bold" if props.get_weight() == "bold" else "Regular"
                props.set_file(str(fonts / f"LiberationSans-{face}.ttf"))
                item.set_fontproperties(props)
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig,json_out=qa_dir/(name+".alignment.json"),
                                           panel_ids=[ax.get_label() for ax in fig.axes],tolerance_pt=1.5)
        fig.savefig(output/f"{name}.pdf",dpi=600,metadata={"CreationDate":None})
        fig.savefig(output/f"{name}.svg",dpi=600,metadata={"Date":None})
        fig.savefig(output/f"{name}.tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"})
        qa=dict(figure=name,width_mm=WIDTH_MM,height_mm=round(fig.get_figheight()*25.4),
                offcanvas_text=[])
        renderer=fig.canvas.get_renderer()
        for ax in fig.axes:
            for t in ax.texts:
                if not fig.bbox.padded(1).contains(*t.get_window_extent(renderer).get_points()[0]) or not fig.bbox.padded(1).contains(*t.get_window_extent(renderer).get_points()[1]):
                    qa["offcanvas_text"].append(t.get_text())
        plt.close(fig)
    if qa["offcanvas_text"]:raise ValueError(qa)
    sources=[dict(figure=number,panel=k,source_files=v[0],statistics=v[1]) for k,v in SOURCES[number].items()]
    return qa,sources

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PROJECT/"figures")
    parser.add_argument("--qa-dir",type=Path,default=PROJECT/"plate_qa")
    args=parser.parse_args()
    for n in (5,6,8):
        render(n,args.output,args.qa_dir)

if __name__=="__main__":
    main()
