from __future__ import annotations
"""
visualize_zigzag_geometry.py
============================
3-D visualisation of the zig-zag pneumatic soft bending actuator.

Generates four figures saved to results/figures/:
  zigzag_2d_side_view.png      — annotated 2-D side-view diagram
  zigzag_3d_undeformed.png     — 3-D undeformed body
  zigzag_3d_deformed.png       — 3-D deformed shapes at several pressures
  zigzag_geometry_summary.png  — 6-panel geometry summary

Run:  python src/visualize_zigzag_geometry.py
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D          # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import cumulative_trapezoid

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from actuator_config import default_actuator_config, compute_zigzag_geometry
from reduced_model   import solve_shape, get_default_physical_config

plt.rcParams.update({"font.family": "serif", "axes.titlesize": 10,
                     "axes.labelsize": 9, "legend.fontsize": 8})

C_BODY  = "#5B9BD5"; C_TOOTH = "#ED7D31"; C_CAVITY = "#F2F2F2"
C_INLET = "#70AD47"; C_TIP   = "#A9D18E"; C_WALL   = "#404040"

# ─────────────────────────────────────────────────────────────────────────────
def _top_profile(geo):
    L=geo["geo_L_mm"]; h=geo["geo_h_mm"]; l1=geo["geo_l1_mm"]
    pitch=geo["tooth_pitch_mm"]; depth=geo["tooth_depth_mm"]
    N=geo["geo_N_teeth"]
    xs,zs=[0.0,l1],[h,h]
    x=l1
    for _ in range(N):
        xs+=[x+pitch/2, x+pitch]; zs+=[h-depth, h]; x+=pitch
    xs.append(L); zs.append(h)
    return np.array(xs), np.array(zs)

def _outline_2d(geo):
    L=geo["geo_L_mm"]; h=geo["geo_h_mm"]; t=geo["geo_t_mm"]
    l1=geo["geo_l1_mm"]; N=geo["geo_N_teeth"]; pitch=geo["tooth_pitch_mm"]
    depth=geo["tooth_depth_mm"]; ih=geo["inner_h_mm"]
    xt,zt = _top_profile(geo)
    outer_x = np.concatenate([[0.0],xt,[L,0.0]])
    outer_z = np.concatenate([[0.0],zt,[0.0,0.0]])
    cav_x = np.array([t, L-t, L-t, t, t])
    cav_z = np.array([t, t, h-t, h-t, t])
    teeth=[]
    xc=l1
    for _ in range(N):
        teeth.append((np.array([xc,xc+pitch/2,xc+pitch,xc]),
                      np.array([h-t,h-depth,h-t,h-t])))
        xc+=pitch
    return dict(outer=(outer_x,outer_z), cavity=(cav_x,cav_z), teeth=teeth)

def _box_faces(x0,x1,w,h,xo=0,yo=0,zo=0):
    x0+=xo;x1+=xo;y0=yo;y1=yo+w;z0=zo;z1=zo+h
    return [[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
            [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0]],
            [[x1,y0,z0],[x1,y0,z1],[x1,y1,z1],[x1,y1,z0]],
            [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
            [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]]]

# ─────────────────────────────────────────────────────────────────────────────
def fig1_side_view(geo, path):
    out = _outline_2d(geo)
    L=geo["geo_L_mm"]; h=geo["geo_h_mm"]; t=geo["geo_t_mm"]
    l1=geo["geo_l1_mm"]; l2=geo["geo_l2_mm"]; N=geo["geo_N_teeth"]
    pitch=geo["tooth_pitch_mm"]; depth=geo["tooth_depth_mm"]
    active=geo["active_length_mm"]

    fig,ax=plt.subplots(figsize=(15,5))
    cx,cz=out["cavity"]
    ax.fill(cx,cz,color=C_CAVITY,zorder=1,label="Air cavity")
    ox,oz=out["outer"]
    ax.fill(ox,oz,color=C_BODY,alpha=0.55,zorder=2,label="Silicone body")
    ax.fill([0,l1,l1,0],[0,0,h,h],color=C_INLET,alpha=0.35,zorder=3,
            label=f"Inlet plain (l₁={l1:.0f} mm)")
    ax.fill([L-l2,L,L,L-l2],[0,0,h,h],color=C_TIP,alpha=0.35,zorder=3,
            label=f"Free-tip plain (l₂={l2:.0f} mm)")
    for i,(tx,tz) in enumerate(out["teeth"]):
        ax.fill(tx,tz,color=C_TOOTH,alpha=0.85,zorder=4,
                label="Zig-zag teeth" if i==0 else None)
    ax.plot(ox,oz,color=C_WALL,lw=1.5,zorder=5)
    ax.plot(cx,cz,color=C_WALL,lw=1.0,ls="--",zorder=5,alpha=0.6)

    ak=dict(arrowstyle="<->",color="black",lw=1.2)
    bk=dict(fontsize=8,ha="center",va="center",
            bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="none",alpha=0.8))
    ya=-h*0.18
    ax.annotate("",xy=(L,ya),xytext=(0,ya),arrowprops=ak)
    ax.text(L/2,ya,f"L = {L:.0f} mm",**bk)
    ya2=h*1.12
    ax.annotate("",xy=(l1,ya2),xytext=(0,ya2),arrowprops=ak)
    ax.text(l1/2,ya2,f"l₁={l1:.0f}",**bk)
    ax.annotate("",xy=(L,ya2),xytext=(L-l2,ya2),arrowprops=ak)
    ax.text(L-l2/2,ya2,f"l₂={l2:.0f}",**bk)
    xt0=l1; xt1=l1+pitch
    yta=h+h*0.25
    ax.annotate("",xy=(xt1,yta),xytext=(xt0,yta),arrowprops=ak)
    ax.text((xt0+xt1)/2,yta,f"p={pitch:.1f} mm",**bk)
    xm=l1+pitch/2
    ax.annotate("",xy=(xm,h-depth),xytext=(xm,h),arrowprops=ak)
    ax.text(xm+pitch*0.18,h-depth/2,f"d={depth:.1f}",fontsize=8)
    ax.annotate("",xy=(-L*0.04,h),xytext=(-L*0.04,0),arrowprops=ak)
    ax.text(-L*0.07,h/2,f"h={h:.0f} mm",fontsize=8,ha="center",va="center",
            rotation=90,bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="none",alpha=0.8))
    ax.text(l1+active/2,-h*0.36,
            f"N = {N} zig-zag teeth  (active = {active:.0f} mm)",
            fontsize=8.5,ha="center",style="italic",
            bbox=dict(boxstyle="round,pad=0.3",fc="#FFF2CC",ec="#D6B656",alpha=0.9))
    ax.annotate("",xy=(0,h/2),xytext=(-L*0.07,h/2),
                arrowprops=dict(arrowstyle="->",color="royalblue",lw=2,mutation_scale=18))
    ax.text(-L*0.08,h/2,"p",color="royalblue",fontsize=13,ha="right",va="center",fontweight="bold")
    ax.text(L+L*0.01,h/2,"free\ntip →",fontsize=8,ha="left",va="center",color="grey")

    ax.set_xlim(-L*0.12,L*1.07); ax.set_ylim(-h*0.5,h*1.48)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Zig-Zag Pneumatic Soft Actuator — 2-D Side-View\n"
                 f"L={L:.0f} mm  w={geo['geo_w_mm']:.0f} mm  h={h:.0f} mm  "
                 f"t={t:.0f} mm  N={N} teeth  pitch={pitch:.1f} mm",fontsize=11,pad=14)
    ax.legend(loc="lower right",fontsize=8,framealpha=0.9)
    plt.tight_layout(); plt.savefig(path,dpi=200,bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
def fig2_3d_undeformed(geo, path):
    L=geo["geo_L_mm"]; w=geo["geo_w_mm"]; h=geo["geo_h_mm"]; t=geo["geo_t_mm"]
    l1=geo["geo_l1_mm"]; l2=geo["geo_l2_mm"]; N=geo["geo_N_teeth"]
    pitch=geo["tooth_pitch_mm"]; depth=geo["tooth_depth_mm"]
    iw=geo["inner_w_mm"]; ih=geo["inner_h_mm"]

    fig=plt.figure(figsize=(13,6))
    ax=fig.add_subplot(111,projection="3d")

    def add(x0,x1,w_,h_,color,alpha=0.3,lw=0.3):
        poly=Poly3DCollection(_box_faces(x0,x1,w_,h_),alpha=alpha,
                               facecolor=color,edgecolor=C_WALL,linewidth=lw)
        ax.add_collection3d(poly)

    add(0,L,w,h,C_BODY,alpha=0.15,lw=0.25)
    add(0,l1,w,h,C_INLET,alpha=0.5)
    add(L-l2,L,w,h,C_TIP,alpha=0.5)
    add(t,L-t,iw,ih,C_CAVITY,alpha=0.5,lw=0.25)
    xc=l1
    for i in range(N):
        poly=Poly3DCollection(_box_faces(xc,xc+pitch,iw,depth,yo=t,zo=h-t-depth),
                               alpha=0.9,facecolor=C_TOOTH,edgecolor=C_WALL,linewidth=0.5)
        ax.add_collection3d(poly); xc+=pitch

    ax.set_xlabel("x [mm]",labelpad=8); ax.set_ylabel("y [mm]",labelpad=8)
    ax.set_zlabel("z [mm]",labelpad=8)
    ax.set_xlim(0,L); ax.set_ylim(0,w); ax.set_zlim(0,h*1.3)
    ax.set_box_aspect([L,w,h*1.3])
    ax.text(L/2,w/2,-h*0.18,f"L={L:.0f} mm",fontsize=8,ha="center")
    ax.text(l1/2,w*1.05,h,f"l₁={l1:.0f}",fontsize=7,color=C_INLET,fontweight="bold")
    ax.text(L-l2/2,w*1.05,h,f"l₂={l2:.0f}",fontsize=7,color="#3A7D2C",fontweight="bold")
    ax.text(l1+geo["active_length_mm"]/2,-w*0.1,0,
            f"N={N} teeth  pitch={pitch:.1f} mm",fontsize=7.5,ha="center",color=C_TOOTH,fontweight="bold")
    ax.set_title(f"Zig-Zag Soft Actuator — 3-D Undeformed\n"
                 f"L={L:.0f}×w={w:.0f}×h={h:.0f} mm  t={t:.0f} mm  "
                 f"N={N}  pitch={pitch:.1f} mm  depth={depth:.1f} mm",fontsize=10,pad=10)
    ax.view_init(elev=22,azim=-55)
    plt.tight_layout(); plt.savefig(path,dpi=200,bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
def _extrude(x_cl,y_cl,theta,w,h):
    hw=w/2; hh=h/2
    on=np.array([-hh,hh,hh,-hh]); oz=np.array([-hw,-hw,hw,hw])
    pts=np.zeros((4,len(x_cl),3))
    for i in range(len(x_cl)):
        ct=np.cos(theta[i]); st=np.sin(theta[i])
        for k in range(4):
            pts[k,i,0]=x_cl[i]-on[k]*st
            pts[k,i,1]=y_cl[i]+on[k]*ct
            pts[k,i,2]=oz[k]
    faces=[]
    for k in range(4):
        kn=(k+1)%4
        for i in range(len(x_cl)-1):
            faces.append([pts[k,i],pts[k,i+1],pts[kn,i+1],pts[kn,i]])
    for ci in [0,len(x_cl)-1]:
        faces.append([pts[k,ci] for k in range(4)])
    return faces

def fig3_3d_deformed(geo, phys_cfg, pressures, path):
    L=geo["geo_L_mm"]; w=geo["geo_w_mm"]; h=geo["geo_h_mm"]
    EI=phys_cfg["EI_eff_Nmm2"]; kp=phys_cfg["k_p_N_per_kPa"]
    nc=min(len(pressures),4); nr=(len(pressures)+nc-1)//nc
    fig=plt.figure(figsize=(5.5*nc,5*nr))
    cmap=plt.cm.plasma; norm=Normalize(min(pressures),max(pressures))
    for idx,p in enumerate(pressures):
        ax=fig.add_subplot(nr,nc,idx+1,projection="3d")
        sol=solve_shape(L_mm=L,p_input_kpa=p,EI_eff_Nmm2=EI,k_p_N_per_kPa=kp,n_points=80)
        faces=_extrude(sol["x_mm"],sol["y_mm"],sol["theta_rad"],w,h)
        ax.add_collection3d(Poly3DCollection(faces,alpha=0.55,
                             facecolor=cmap(norm(p)),edgecolor="none"))
        ax.plot(sol["x_mm"],sol["y_mm"],np.zeros(len(sol["x_mm"])),"k-",lw=1.5)
        ax.set_xlabel("x",fontsize=7); ax.set_ylabel("y",fontsize=7); ax.set_zlabel("z",fontsize=7)
        ax.set_zlim(-w/2,w/2); ax.set_xlim(-5,L+5); ax.set_ylim(-5,L+5)
        ax.set_box_aspect([1,1,0.15]); ax.view_init(elev=18,azim=-60); ax.tick_params(labelsize=6)
        ax.set_title(f"p={p:.0f} kPa  tip_y={sol['tip_y_mm']:.1f} mm  "
                     f"θ={np.degrees(sol['tip_angle_rad']):.1f}°",fontsize=8,pad=4)
    sm=ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
    fig.colorbar(sm,ax=fig.get_axes(),label="Pressure [kPa]",fraction=0.015,pad=0.04)
    fig.suptitle(f"Zig-Zag Soft Actuator — 3-D Deformed  EI={EI:.0f} N·mm²  kp={kp:.4f} N/kPa",
                 fontsize=11,y=1.01)
    plt.tight_layout(); plt.savefig(path,dpi=180,bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
def fig4_summary(geo, phys_cfg, path):
    import matplotlib.gridspec as gridspec
    L=geo["geo_L_mm"]; w=geo["geo_w_mm"]; h=geo["geo_h_mm"]; t=geo["geo_t_mm"]
    l1=geo["geo_l1_mm"]; l2=geo["geo_l2_mm"]; N=geo["geo_N_teeth"]
    pitch=geo["tooth_pitch_mm"]; depth=geo["tooth_depth_mm"]
    active=geo["active_length_mm"]; iw=geo["inner_w_mm"]; ih=geo["inner_h_mm"]
    EI=phys_cfg["EI_eff_Nmm2"]; kp=phys_cfg["k_p_N_per_kPa"]
    factor=geo["zigzag_compliance_factor"]
    plo=phys_cfg["p_min_kpa"]; phi=phys_cfg["p_max_kpa"]
    p_sw=np.linspace(plo,phi,60)

    fig=plt.figure(figsize=(16,9))
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.35)

    # (0,0) Cross-section
    ax=fig.add_subplot(gs[0,0])
    ax.add_patch(mpatches.Rectangle((0,0),w,h,lw=1.5,ec=C_WALL,fc=C_BODY,alpha=0.55,label="Body"))
    ax.add_patch(mpatches.Rectangle((t,t),iw,ih,lw=1.0,ec=C_WALL,fc=C_CAVITY,label="Cavity"))
    ax.set_xlim(-w*0.25,w*1.25); ax.set_ylim(-h*0.25,h*1.4); ax.set_aspect("equal")
    ak=dict(arrowstyle="<->",color="k",lw=1.2)
    ax.annotate("",xy=(w,-h*0.15),xytext=(0,-h*0.15),arrowprops=ak)
    ax.text(w/2,-h*0.22,f"w={w:.0f} mm",ha="center",fontsize=8)
    ax.annotate("",xy=(-w*0.18,h),xytext=(-w*0.18,0),arrowprops=ak)
    ax.text(-w*0.28,h/2,f"h={h:.0f} mm",ha="center",fontsize=8,rotation=90)
    ax.legend(fontsize=8,loc="upper right"); ax.grid(True,alpha=0.25)
    ax.set_xlabel("y [mm]"); ax.set_ylabel("z [mm]"); ax.set_title("(a) Cross-section (y-z)")

    # (0,1) Single tooth profile
    ax=fig.add_subplot(gs[0,1])
    tx=np.array([0,pitch/2,pitch]); tz=np.array([h,h-depth,h])
    ax.fill_between(tx,h-t,tz,color=C_TOOTH,alpha=0.8,label="Tooth")
    ax.plot(tx,tz,color=C_WALL,lw=1.5)
    ax.axhline(h-t,color="grey",ls="--",lw=1,label=f"Inner top (z={h-t:.0f})")
    ax.annotate("",xy=(0,h*1.1),xytext=(pitch,h*1.1),arrowprops=ak)
    ax.text(pitch/2,h*1.16,f"pitch p={pitch:.1f} mm",ha="center",fontsize=8)
    ax.annotate("",xy=(pitch/2+pitch*0.06,h),xytext=(pitch/2+pitch*0.06,h-depth),arrowprops=ak)
    ax.text(pitch/2+pitch*0.15,h-depth/2,f"d={depth:.1f}",fontsize=7.5,color="darkorange")
    ax.set_xlim(-pitch*0.1,pitch*1.3); ax.set_ylim(-h*0.1,h*1.28)
    ax.legend(fontsize=8); ax.grid(True,alpha=0.25)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]"); ax.set_title("(b) Tooth profile (x-z)")

    # (0,2) Tip response
    ax=fig.add_subplot(gs[0,2]); ax2=ax.twinx()
    tys=[]; ths=[]
    for p in p_sw:
        sol=solve_shape(L_mm=L,p_input_kpa=p,EI_eff_Nmm2=EI,k_p_N_per_kPa=kp)
        tys.append(sol["tip_y_mm"]); ths.append(np.degrees(sol["tip_angle_rad"]))
    ax.plot(p_sw,tys,color=C_BODY,lw=2,label="Tip y [mm]")
    ax2.plot(p_sw,ths,color=C_TOOTH,lw=2,ls="--",label="Tip θ [°]")
    ax.set_xlabel("Pressure [kPa]"); ax.set_ylabel("Tip y [mm]",color=C_BODY)
    ax2.set_ylabel("Tip angle [°]",color=C_TOOTH)
    ax.tick_params(axis="y",labelcolor=C_BODY); ax2.tick_params(axis="y",labelcolor=C_TOOTH)
    lines=ax.get_lines()+ax2.get_lines()
    ax.legend(lines,[l.get_label() for l in lines],fontsize=8,loc="upper left")
    ax.set_title("(c) Tip response"); ax.grid(True,alpha=0.3)

    # (1,0) Axial layout
    ax=fig.add_subplot(gs[1,0])
    ax.barh(0,l1,left=0,height=0.5,color=C_INLET,label=f"Inlet l₁={l1:.0f}mm")
    ax.barh(0,active,left=l1,height=0.5,color=C_TOOTH,alpha=0.7,label=f"Teeth {active:.0f}mm")
    ax.barh(0,l2,left=l1+active,height=0.5,color=C_TIP,label=f"Tip l₂={l2:.0f}mm")
    for k in range(N): ax.axvline(l1+k*pitch,color=C_WALL,lw=0.6,ls=":",alpha=0.5)
    ax.set_xlim(0,L); ax.set_ylim(-0.5,1.0); ax.set_yticks([])
    ax.set_xlabel("x [mm]"); ax.legend(fontsize=8,loc="lower right")
    ax.set_title(f"(d) Axial layout  (L={L:.0f} mm)"); ax.grid(True,axis="x",alpha=0.3)

    # (1,1) Centerline family
    ax=fig.add_subplot(gs[1,1])
    p_plt=np.linspace(plo,phi,7)
    cmap_p=plt.cm.viridis; norm_p=plt.Normalize(p_plt.min(),p_plt.max())
    for p in p_plt:
        sol=solve_shape(L_mm=L,p_input_kpa=p,EI_eff_Nmm2=EI,k_p_N_per_kPa=kp,n_points=150)
        ax.plot(sol["x_mm"],sol["y_mm"],color=cmap_p(norm_p(p)),lw=1.8)
    plt.colorbar(ScalarMappable(cmap=cmap_p,norm=norm_p),ax=ax,label="p [kPa]",fraction=0.05)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect("equal")
    ax.set_title("(e) Centerline family"); ax.grid(True,alpha=0.3)

    # (1,2) Parameter table
    ax=fig.add_subplot(gs[1,2]); ax.axis("off")
    rows=[["Parameter","Value","Unit"],
          ["L",f"{L:.0f}","mm"],["w",f"{w:.0f}","mm"],["h",f"{h:.0f}","mm"],
          ["t",f"{t:.0f}","mm"],[f"Cavity {iw:.0f}×{ih:.0f}","mm×mm","—"],
          ["l₁",f"{l1:.0f}","mm"],["l₂",f"{l2:.0f}","mm"],
          ["Active",f"{active:.0f}","mm"],["N teeth",f"{N}","—"],
          ["Pitch",f"{pitch:.1f}","mm"],["Depth",f"{depth:.1f}","mm"],
          ["Compliance",f"{factor:.2f}","—"],
          ["EI_eff",f"{EI:.1f}","N·mm²"],["k_p",f"{kp:.5f}","N/kPa"]]
    tbl=ax.table(cellText=[[r[0],r[1],r[2]] for r in rows[1:]],
                 colLabels=rows[0],cellLoc="center",loc="center",bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (i,j),cell in tbl.get_celld().items():
        if i==0: cell.set_facecolor("#2F5496"); cell.set_text_props(color="white",fontweight="bold")
        elif i%2==0: cell.set_facecolor("#D9E1F2")
        cell.set_edgecolor("#BFBFBF")
    ax.set_title("(f) Parameters",pad=10)

    fig.suptitle("Zig-Zag Pneumatic Soft Actuator — Geometry Summary",
                 fontsize=13,fontweight="bold",y=1.01)
    plt.savefig(path,dpi=180,bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
def run_all(config_overrides=None, save_dir="results/figures"):
    os.makedirs(save_dir,exist_ok=True)
    geo      = compute_zigzag_geometry(config_overrides)
    phys_cfg = get_default_physical_config()
    if config_overrides: phys_cfg.update(config_overrides)

    print("\n── Zig-Zag Actuator Geometry ──────────────────────────────")
    print(f"  L={geo['geo_L_mm']:.0f}mm  w={geo['geo_w_mm']:.0f}mm  h={geo['geo_h_mm']:.0f}mm  t={geo['geo_t_mm']:.0f}mm")
    print(f"  l1={geo['geo_l1_mm']:.0f}mm  l2={geo['geo_l2_mm']:.0f}mm  N={geo['geo_N_teeth']} teeth")
    print(f"  active={geo['active_length_mm']:.1f}mm  pitch={geo['tooth_pitch_mm']:.2f}mm  depth={geo['tooth_depth_mm']:.2f}mm")
    print(f"  EI_eff={phys_cfg['EI_eff_Nmm2']:.2f}N·mm²  k_p={phys_cfg['k_p_N_per_kPa']:.5f}N/kPa")
    print("────────────────────────────────────────────────────────────\n")

    plo=phys_cfg["p_min_kpa"]; phi=phys_cfg["p_max_kpa"]
    deform_ps=[plo, plo+(phi-plo)*0.33, plo+(phi-plo)*0.66, phi]

    print("Fig 1 — 2-D side view ...")
    fig1_side_view(geo, os.path.join(save_dir,"zigzag_2d_side_view.png"))
    print("Fig 2 — 3-D undeformed ...")
    fig2_3d_undeformed(geo, os.path.join(save_dir,"zigzag_3d_undeformed.png"))
    print("Fig 3 — 3-D deformed ...")
    fig3_3d_deformed(geo, phys_cfg, deform_ps, os.path.join(save_dir,"zigzag_3d_deformed.png"))
    print("Fig 4 — Geometry summary ...")
    fig4_summary(geo, phys_cfg, os.path.join(save_dir,"zigzag_geometry_summary.png"))
    print(f"\n✓  All geometry figures saved to: {save_dir}\n")


if __name__ == "__main__":
    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "results", "figures")
    run_all(save_dir=SAVE_DIR)
