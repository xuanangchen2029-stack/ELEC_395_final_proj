from __future__ import annotations
"""
experiments.py
==============
Two ablation experiments that build directly on the ORIGINAL model/training code.
No architecture or hyperparameter changes — identical to train_pinn.py / train_mlp.py.

Experiment 1 — Effect of blocked-force loss
--------------------------------------------
  Version A: remove blocked-force loss term (w_block = 0)
  Version B: keep blocked-force loss (original, w_block = 10)
  Metric comparison on test split:
    - k_p error (%), EI_eff error (%)
    - tip displacement RMSE (mm)
    - blocked force RMSE (N)

Experiment 2 — PINN vs MLP under different data regimes
---------------------------------------------------------
  Three data budgets:
    low    :  4 training pressures  (p_min, p_max, and 2 intermediate)
    medium : 6 training pressures   (original split)
    high   : 8 training pressures   (denser sampling)
  For each budget: train PINN + MLP, record test metrics.
  Generates a bar-chart comparison.

Run:
    python src/experiments.py
Results written to:  results/exp1/  and  results/exp2/
Summary figures:     results/exp1_blocked_force_ablation.png
                     results/exp2_data_regime_comparison.png
"""

import copy, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch, torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pinn_model    import SoftActuatorPINN, MLP, physics_residuals, gradients, blocked_force_hat_linear
from reduced_model import solve_shape, blocked_force_constrained, get_default_physical_config
from data_generation import generate_synthetic_dataset

plt.rcParams.update({"font.family":"serif","axes.titlesize":10,
                     "axes.labelsize":9,"legend.fontsize":8,"figure.dpi":150})

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers  (mirror train_pinn.py / train_mlp.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def to_t(arr, dev):
    return torch.tensor(np.asarray(arr, dtype=np.float32), device=dev).view(-1, 1)

def split_df(df, sp): return df[df["split"] == sp].reset_index(drop=True)

def make_bundle(sdf, tdf, bdf, sp, dev):
    sdf=split_df(sdf,sp); tdf=split_df(tdf,sp); bdf=split_df(bdf,sp)
    return dict(shape_df=sdf, tip_df=tdf, block_df=bdf,
                s_d=to_t(sdf["s_mm"],dev),    p_d=to_t(sdf["p_kpa"],dev),
                x_d=to_t(sdf["x_mm"],dev),    y_d=to_t(sdf["y_mm"],dev),
                p_t=to_t(tdf["p_kpa"],dev),   tip_y=to_t(tdf["tip_y_mm"],dev),
                tip_angle=to_t(tdf["tip_angle_rad"],dev),
                p_bf=to_t(bdf["p_kpa"],dev),  F_b=to_t(bdf["F_b_N"],dev))

def sample_coll(nc, dev, L, plo, phi):
    s = L*torch.rand(nc,1,device=dev,requires_grad=True)
    p = plo+(phi-plo)*torch.rand(nc,1,device=dev,requires_grad=True)
    return s, p

def sup_losses(model, bun, mse, L, Fsc, w_block=10.0):
    xh,yh,_ = model(bun["s_d"],bun["p_d"])
    ls = mse(xh/L,bun["x_d"]/L) + mse(yh/L,bun["y_d"]/L)
    nt = len(bun["tip_df"])
    st = L*torch.ones(nt,1,dtype=torch.float32,device=bun["p_t"].device)
    _,yth,thth = model(st,bun["p_t"])
    lt = mse(yth/L,bun["tip_y"]/L) + mse(thth,bun["tip_angle"])
    Fbh = blocked_force_hat_linear(model,bun["p_bf"])
    lb  = mse(Fbh/Fsc,bun["F_b"]/Fsc)
    return ls, lt, lb

def unpack(model, s, p):
    out=model(s,p)
    return (out[0],out[1],out[2]) if isinstance(out,tuple) else \
           (out[:,0:1],out[:,1:2],out[:,2:3])

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a).ravel()-np.asarray(b).ravel())**2)))


# ─────────────────────────────────────────────────────────────────────────────
# Core PINN trainer  (original hyperparameters, w_block controllable)
# ─────────────────────────────────────────────────────────────────────────────

def train_pinn_core(shape_df, tip_df, block_df, meta, device,
                    out_dir, n_iters=4000, lr=1e-3, seed=0,
                    w_phys=1.0, w_bc=10.0, w_shape=20.0,
                    w_tip=20.0, w_block=10.0, tag="pinn"):
    """
    Exact replica of original train_pinn.py with controllable w_block.
    All other hyperparameters match the original exactly.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    L    = float(meta.loc[0,"L_mm"])
    plo  = float(meta.loc[0,"p_min_kpa"])
    phi  = float(meta.loc[0,"p_max_kpa"])
    EI_tr= float(meta.loc[0,"EI_eff_true_Nmm2"])
    kp_tr= float(meta.loc[0,"k_p_true_N_per_kPa"])
    Fsc  = max((8/3)*kp_tr*phi, 1e-6)

    tr = make_bundle(shape_df,tip_df,block_df,"train",device)
    va = make_bundle(shape_df,tip_df,block_df,"val",  device)

    model = SoftActuatorPINN(width=64,depth=4,length_scale_mm=L,
                              pressure_scale_kpa=phi,
                              EI_scale_Nmm2=EI_tr,
                              kp_scale_N_per_kPa=kp_tr).to(device)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist=[]; best={"val_total":np.inf,"iter":-1,"state_dict":None}

    for it in range(n_iters):
        opt.zero_grad()
        sf,pf = sample_coll(512,device,L,plo,phi)
        rx,ry,rth = physics_residuals(model,sf,pf)
        lp = rx.pow(2).mean()+ry.pow(2).mean()+(rth/(kp_tr*phi)).pow(2).mean()

        pbc = plo+(phi-plo)*torch.rand(128,1,device=device)
        s0  = torch.zeros_like(pbc,requires_grad=True)
        sL  = L*torch.ones_like(pbc,requires_grad=True)
        x0,y0,th0 = model(s0,pbc); _,_,thL = model(sL,pbc)
        thLs = gradients(thL,sL)
        lbc = (x0/L).pow(2).mean()+(y0/L).pow(2).mean()+th0.pow(2).mean()+(thLs*L).pow(2).mean()

        ls,lt,lb = sup_losses(model,tr,mse,L,Fsc)
        loss = w_phys*lp + w_bc*lbc + w_shape*ls + w_tip*lt + w_block*lb
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            ls_v,lt_v,lb_v = sup_losses(model,va,mse,L,Fsc)
            val_total = 20*ls_v+20*lt_v+10*lb_v
        model.train()

        hist.append(dict(iter=it,loss=float(loss),phys=float(lp),bc=float(lbc),
                         shape=float(ls),tip=float(lt),block=float(lb),
                         val_total=float(val_total),
                         EI_eff_Nmm2=float(model.EI_eff),
                         k_p_N_per_kPa=float(model.k_p)))
        if float(val_total) < best["val_total"]:
            best={"val_total":float(val_total),"iter":it,
                  "state_dict":copy.deepcopy(model.state_dict())}
        if it%500==0 or it==n_iters-1:
            EIe=abs(float(model.EI_eff)-EI_tr)/EI_tr*100
            print(f"    [{tag}] it={it:4d}  loss={float(loss):.3e}  "
                  f"EI={float(model.EI_eff):.1f}({EIe:.1f}%)  "
                  f"kp={float(model.k_p):.5f}  val={float(val_total):.3e}")

    if best["state_dict"]: model.load_state_dict(best["state_dict"])
    pd.DataFrame(hist).to_csv(os.path.join(out_dir,f"{tag}_history.csv"),index=False)
    torch.save(model.state_dict(),os.path.join(out_dir,f"{tag}_model_best.pt"))
    return model, pd.DataFrame(hist), EI_tr, kp_tr


def train_mlp_core(shape_df, tip_df, meta, device,
                   out_dir, n_iters=3000, lr=1e-3, seed=0, tag="mlp"):
    """Exact replica of original train_mlp.py."""
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    L   = float(meta.loc[0,"L_mm"])
    phi = float(meta.loc[0,"p_max_kpa"])
    mse = nn.MSELoss()

    def mk(sp):
        sd=split_df(shape_df,sp); td=split_df(tip_df,sp)
        return dict(shape_df=sd,tip_df=td,
                    s_d=to_t(sd["s_mm"],device),p_d=to_t(sd["p_kpa"],device),
                    x_d=to_t(sd["x_mm"],device),y_d=to_t(sd["y_mm"],device),
                    p_t=to_t(td["p_kpa"],device),tip_y=to_t(td["tip_y_mm"],device),
                    tip_angle=to_t(td["tip_angle_rad"],device))
    tr=mk("train"); va=mk("val")

    model=MLP(2,3,64,4,L,phi).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr)

    def losses(b):
        out=model(b["s_d"],b["p_d"])
        ls=mse(out[:,0:1]/L,b["x_d"]/L)+mse(out[:,1:2]/L,b["y_d"]/L)
        nt=len(b["tip_df"])
        st=L*torch.ones(nt,1,device=device)
        ot=model(st,b["p_t"])
        lt=mse(ot[:,1:2]/L,b["tip_y"]/L)+mse(ot[:,2:3],b["tip_angle"])
        return ls,lt

    hist=[]; best={"v":np.inf,"sd":None}
    for it in range(n_iters):
        model.train(); opt.zero_grad()
        ls,lt=losses(tr); loss=20*ls+20*lt
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vls,vlt=losses(va); v=20*vls+20*vlt
        model.train()
        hist.append(dict(iter=it,loss=float(loss),val_total=float(v)))
        if float(v)<best["v"]: best={"v":float(v),"sd":copy.deepcopy(model.state_dict())}
        if it%500==0 or it==n_iters-1:
            print(f"    [{tag}] it={it:4d}  loss={float(loss):.3e}  val={float(v):.3e}")
    if best["sd"]: model.load_state_dict(best["sd"])
    pd.DataFrame(hist).to_csv(os.path.join(out_dir,f"{tag}_history.csv"),index=False)
    torch.save(model.state_dict(),os.path.join(out_dir,f"{tag}_model_best.pt"))
    return model, pd.DataFrame(hist)


def eval_metrics(pinn, mlp, shape_df, tip_df, block_df, meta, device, split="test"):
    """
    Compute all comparison metrics on a given split.
    mlp may be None (e.g. in Exp 1 where only PINN variants are compared);
    MLP metrics will be set to NaN in that case.
    """
    L    = float(meta.loc[0,"L_mm"])
    EI_tr= float(meta.loc[0,"EI_eff_true_Nmm2"])
    kp_tr= float(meta.loc[0,"k_p_true_N_per_kPa"])
    plo  = float(meta.loc[0,"p_min_kpa"])
    phi  = float(meta.loc[0,"p_max_kpa"])
    p_sw = np.linspace(plo,phi,100)

    sdf=split_df(shape_df,split); tdf=split_df(tip_df,split); bdf=split_df(block_df,split)

    with torch.no_grad():
        s=to_t(sdf["s_mm"],device); p=to_t(sdf["p_kpa"],device)
        _,py,_ = unpack(pinn,s,p)
        my = unpack(mlp, s,p)[1] if mlp is not None else None

        st=L*torch.ones(len(tdf),1,device=device)
        pt=to_t(tdf["p_kpa"],device)
        _,py_tip,_ = unpack(pinn,st,pt)
        my_tip = unpack(mlp, st,pt)[1] if mlp is not None else None

        Fbh = blocked_force_hat_linear(pinn,to_t(bdf["p_kpa"],device))

    ref_ty = np.array([solve_shape(L_mm=L,p_input_kpa=p_,EI_eff_Nmm2=EI_tr,
                                    k_p_N_per_kPa=kp_tr)["tip_y_mm"] for p_ in p_sw])
    ref_bl = np.array([blocked_force_constrained(L_mm=L,p_input_kpa=p_,
                        EI_eff_Nmm2=EI_tr,k_p_N_per_kPa=kp_tr) for p_ in p_sw])

    with torch.no_grad():
        stt=L*torch.ones(len(p_sw),1,device=device)
        ptt=to_t(p_sw,device)
        _,py_sw,_ = unpack(pinn,stt,ptt)
        my_sw = unpack(mlp, stt,ptt)[1] if mlp is not None else None
        Fb_sw = blocked_force_hat_linear(pinn,ptt).cpu().numpy().ravel()

    py_sw = py_sw.cpu().numpy().ravel()
    my_sw = my_sw.cpu().numpy().ravel() if my_sw is not None else None

    EI_id=float(pinn.EI_eff); kp_id=float(pinn.k_p)
    return {
        "EI_pct_err":       abs(EI_id-EI_tr)/EI_tr*100,
        "kp_pct_err":       abs(kp_id-kp_tr)/kp_tr*100,
        "pinn_tip_rmse":    rmse(py_sw, ref_ty),
        "mlp_tip_rmse":     rmse(my_sw, ref_ty) if my_sw is not None else float("nan"),
        "pinn_shape_y_rmse":rmse(py.cpu().numpy(), sdf["y_mm"].values),
        "mlp_shape_y_rmse": rmse(my.cpu().numpy(), sdf["y_mm"].values) if my is not None else float("nan"),
        "pinn_block_rmse":  rmse(Fb_sw, ref_bl),
        "EI_id": EI_id, "kp_id": kp_id, "EI_tr": EI_tr, "kp_tr": kp_tr,
    }


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Blocked-force loss ablation
# ═════════════════════════════════════════════════════════════════════════════

def exp1_blocked_force(base_data_dir="data/synthetic",
                       out_root="results/exp1",
                       n_iters=4000, seed=0):
    print("\n" + "="*60)
    print("EXPERIMENT 1: Blocked-force loss ablation")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape_df = pd.read_csv(os.path.join(base_data_dir,"shape_data.csv"))
    tip_df   = pd.read_csv(os.path.join(base_data_dir,"tip_data.csv"))
    block_df = pd.read_csv(os.path.join(base_data_dir,"blocked_force_data.csv"))
    meta     = pd.read_csv(os.path.join(base_data_dir,"metadata.csv"))

    results = {}

    # Version A: w_block = 0  (no blocked-force loss)
    print("\n── Version A: w_block = 0 (no blocked-force loss) ──")
    dir_a = os.path.join(out_root, "vA_no_block")
    pinn_a, hist_a, EI_tr, kp_tr = train_pinn_core(
        shape_df, tip_df, block_df, meta, device, dir_a,
        n_iters=n_iters, w_block=0.0, tag="pinn_A", seed=seed)
    results["A"] = eval_metrics(pinn_a, None, shape_df, tip_df, block_df, meta, device)
    results["A"]["label"] = "PINN-A\n(no block loss)"
    results["A"]["hist"]  = hist_a

    # Version B: w_block = 10  (original)
    print("\n── Version B: w_block = 10 (original, with blocked-force loss) ──")
    dir_b = os.path.join(out_root, "vB_with_block")
    pinn_b, hist_b, _, _ = train_pinn_core(
        shape_df, tip_df, block_df, meta, device, dir_b,
        n_iters=n_iters, w_block=10.0, tag="pinn_B", seed=seed)
    results["B"] = eval_metrics(pinn_b, None, shape_df, tip_df, block_df, meta, device)
    results["B"]["label"] = "PINN-B\n(with block loss)"
    results["B"]["hist"]  = hist_b

    # ── Save results ──────────────────────────────────────────────────────────
    summary = []
    for k, v in results.items():
        summary.append({
            "version": k, "label": v["label"].replace("\n"," "),
            "EI_pct_err": v["EI_pct_err"], "kp_pct_err": v["kp_pct_err"],
            "pinn_tip_rmse": v["pinn_tip_rmse"],
            "pinn_block_rmse": v["pinn_block_rmse"],
            "pinn_shape_y_rmse": v["pinn_shape_y_rmse"],
        })
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(os.path.join(out_root,"exp1_summary.csv"),index=False)
    print("\n── Exp 1 results ──────────────────────────────")
    print(df_sum.to_string(index=False))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    fig.suptitle("Experiment 1: Effect of Blocked-Force Loss\n"
                 "Version A = no block loss  |  Version B = with block loss (original)",
                 fontsize=11,fontweight="bold",y=1.01)

    labels = [v["label"] for v in results.values()]
    colors = ["#ED7D31","#5B9BD5"]

    # (0,0) EI_eff error
    ax=axes[0,0]
    bars=ax.bar(labels,[v["EI_pct_err"] for v in results.values()],color=colors,
                edgecolor="k",lw=0.8,width=0.5)
    for bar,val in zip(bars,[v["EI_pct_err"] for v in results.values()]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,
                f"{val:.2f}%",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_ylabel("EI_eff error (%)"); ax.set_title("(a) EI_eff identification error")
    ax.grid(True,axis="y",alpha=0.3); ax.axhline(5,color="k",ls="--",lw=0.8,alpha=0.5)
    ax.text(1.5,5.2,"5% threshold",fontsize=7,color="k",ha="center")

    # (0,1) k_p error
    ax=axes[0,1]
    bars=ax.bar(labels,[v["kp_pct_err"] for v in results.values()],color=colors,
                edgecolor="k",lw=0.8,width=0.5)
    for bar,val in zip(bars,[v["kp_pct_err"] for v in results.values()]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,
                f"{val:.2f}%",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_ylabel("k_p error (%)"); ax.set_title("(b) k_p identification error")
    ax.grid(True,axis="y",alpha=0.3)

    # (0,2) Tip RMSE
    ax=axes[0,2]
    bars=ax.bar(labels,[v["pinn_tip_rmse"] for v in results.values()],color=colors,
                edgecolor="k",lw=0.8,width=0.5)
    for bar,val in zip(bars,[v["pinn_tip_rmse"] for v in results.values()]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,
                f"{val:.1f}mm",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_ylabel("Tip displacement RMSE (mm)"); ax.set_title("(c) Tip displacement RMSE")
    ax.grid(True,axis="y",alpha=0.3)

    # (1,0) Blocked force RMSE
    ax=axes[1,0]
    bars=ax.bar(labels,[v["pinn_block_rmse"]*1e3 for v in results.values()],color=colors,
                edgecolor="k",lw=0.8,width=0.5)
    for bar,val in zip(bars,[v["pinn_block_rmse"]*1e3 for v in results.values()]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.05,
                f"{val:.2f}mN",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_ylabel("Blocked force RMSE (mN)"); ax.set_title("(d) Blocked force RMSE")
    ax.grid(True,axis="y",alpha=0.3)

    # (1,1) EI convergence comparison
    ax=axes[1,1]
    for (k,v),c in zip(results.items(),colors):
        ax.plot(v["hist"]["iter"],v["hist"]["EI_eff_Nmm2"],color=c,lw=2,label=v["label"].replace("\n"," "))
    ax.axhline(EI_tr,color="k",ls="--",lw=1.5,label=f"True EI={EI_tr:.0f}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("EI_eff [N·mm²]")
    ax.set_title("(e) EI_eff convergence comparison"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    # (1,2) k_p convergence comparison
    ax=axes[1,2]
    for (k,v),c in zip(results.items(),colors):
        ax.plot(v["hist"]["iter"],v["hist"]["k_p_N_per_kPa"],color=c,lw=2,label=v["label"].replace("\n"," "))
    ax.axhline(kp_tr,color="k",ls="--",lw=1.5,label=f"True kp={kp_tr:.5f}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("k_p [N/kPa]")
    ax.set_title("(f) k_p convergence comparison"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    plt.tight_layout()
    out=os.path.join(out_root,"exp1_blocked_force_ablation.png")
    plt.savefig(out,dpi=200,bbox_inches="tight"); plt.close()
    print(f"\n  Saved: {out}")
    return df_sum


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — PINN vs MLP under different data regimes
# ═════════════════════════════════════════════════════════════════════════════

def exp2_data_regimes(out_root="results/exp2", n_iters_pinn=4000,
                      n_iters_mlp=3000, n_seeds=3):
    """
    Three data regimes × (PINN + MLP) × n_seeds → mean ± std bar chart.

    Data regimes (number of training pressure levels):
      low    : 3 train + 1 val + 1 test  =  5 pressures total  [20, 50, 80 for train]
      medium : 6 train + 2 val + 2 test  = 10 pressures (original)
      high   : 8 train + 3 val + 3 test  = 14 pressures (denser)
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: PINN vs MLP under different data regimes")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_root, exist_ok=True)

    cfg = get_default_physical_config()
    L   = cfg["L_mm"]; plo=cfg["p_min_kpa"]; phi=cfg["p_max_kpa"]

    regimes = {
        "low"   : np.array([20.0, 40.0, 60.0, 80.0,               # train
                             30.0, 70.0,                            # val
                             50.0, 100.0]),                         # test
        "medium": np.arange(10.0, 101.0, 10.0),                    # original 10 levels
        "high"  : np.arange(10.0, 101.0,  7.0),                    # ~14 levels
    }
    # Train fractions tuned so roughly 60/20/20 split
    regime_splits = {
        "low":    (0.50, 0.25),
        "medium": (0.60, 0.20),
        "high":   (0.60, 0.20),
    }

    all_records = []

    for regime_name, pressures in regimes.items():
        print(f"\n── Regime: {regime_name}  ({len(pressures)} pressure levels) ──")

        for seed in range(n_seeds):
            print(f"  Seed {seed} ...")
            data_dir = os.path.join(out_root, f"data_{regime_name}_seed{seed}")
            tf, vf   = regime_splits[regime_name]
            generate_synthetic_dataset(
                save_dir=data_dir, pressures_kpa=pressures,
                n_shape_points=25, seed=seed,
                train_frac=tf, val_frac=vf,
            )
            sdf = pd.read_csv(os.path.join(data_dir,"shape_data.csv"))
            tdf = pd.read_csv(os.path.join(data_dir,"tip_data.csv"))
            bdf = pd.read_csv(os.path.join(data_dir,"blocked_force_data.csv"))
            meta= pd.read_csv(os.path.join(data_dir,"metadata.csv"))

            run_dir = os.path.join(out_root, f"run_{regime_name}_seed{seed}")

            # PINN
            pinn,_,EI_tr,kp_tr = train_pinn_core(
                sdf,tdf,bdf,meta,device,run_dir,
                n_iters=n_iters_pinn,seed=seed,tag="pinn")

            # MLP
            mlp,_ = train_mlp_core(
                sdf,tdf,meta,device,run_dir,
                n_iters=n_iters_mlp,seed=seed,tag="mlp")

            m = eval_metrics(pinn,mlp,sdf,tdf,bdf,meta,device,split="test")
            m.update({"regime":regime_name,"seed":seed,
                      "n_train_pressures": int((sdf["split"]=="train")
                                               .groupby(sdf["p_kpa"]).any().sum()
                                               if False else
                                               len(sdf[sdf["split"]=="train"]["p_kpa"].unique()))})
            all_records.append(m)
            print(f"    → tip RMSE: PINN={m['pinn_tip_rmse']:.2f}mm  "
                  f"MLP={m['mlp_tip_rmse']:.2f}mm  "
                  f"EI err={m['EI_pct_err']:.2f}%")

    df_all = pd.DataFrame(all_records)
    df_all.to_csv(os.path.join(out_root,"exp2_all_runs.csv"),index=False)

    # Aggregate: mean ± std per regime
    agg = df_all.groupby("regime").agg(
        pinn_tip_mean=("pinn_tip_rmse","mean"), pinn_tip_std=("pinn_tip_rmse","std"),
        mlp_tip_mean =("mlp_tip_rmse","mean"),  mlp_tip_std =("mlp_tip_rmse","std"),
        pinn_shape_mean=("pinn_shape_y_rmse","mean"),pinn_shape_std=("pinn_shape_y_rmse","std"),
        mlp_shape_mean =("mlp_shape_y_rmse","mean"), mlp_shape_std =("mlp_shape_y_rmse","std"),
        EI_err_mean=("EI_pct_err","mean"),EI_err_std=("EI_pct_err","std"),
    ).reindex(["low","medium","high"]).reset_index()
    agg.to_csv(os.path.join(out_root,"exp2_summary.csv"),index=False)

    print("\n── Exp 2 aggregated results ────────────────────────────────────")
    print(agg[["regime","pinn_tip_mean","mlp_tip_mean","pinn_shape_mean",
               "mlp_shape_mean","EI_err_mean"]].to_string(index=False))

    # ── Figure ────────────────────────────────────────────────────────────────
    x   = np.arange(len(agg))
    w_  = 0.35
    regime_labels = [r.capitalize() for r in agg["regime"]]
    colors_pinn="#5B9BD5"; colors_mlp="#ED7D31"

    fig,axes=plt.subplots(1,3,figsize=(15,5))
    fig.suptitle(
        f"Experiment 2: PINN vs MLP Under Different Data Regimes\n"
        f"(averaged over {n_seeds} seeds, error bars = 1 std)",
        fontsize=11,fontweight="bold",y=1.01)

    # (0) Tip displacement RMSE
    ax=axes[0]
    b1=ax.bar(x-w_/2,agg["pinn_tip_mean"],w_,yerr=agg["pinn_tip_std"],
              color=colors_pinn,label="PINN",capsize=5,edgecolor="k",lw=0.8)
    b2=ax.bar(x+w_/2,agg["mlp_tip_mean"], w_,yerr=agg["mlp_tip_std"],
              color=colors_mlp, label="MLP", capsize=5,edgecolor="k",lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(regime_labels)
    ax.set_ylabel("Tip y RMSE (mm)"); ax.set_title("(a) Tip displacement RMSE")
    ax.legend(); ax.grid(True,axis="y",alpha=0.3)
    for bar,v in zip(list(b1)+list(b2),
                     list(agg["pinn_tip_mean"])+list(agg["mlp_tip_mean"])):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
                f"{v:.1f}",ha="center",va="bottom",fontsize=7.5)

    # (1) Shape y RMSE
    ax=axes[1]
    b1=ax.bar(x-w_/2,agg["pinn_shape_mean"],w_,yerr=agg["pinn_shape_std"],
              color=colors_pinn,label="PINN",capsize=5,edgecolor="k",lw=0.8)
    b2=ax.bar(x+w_/2,agg["mlp_shape_mean"], w_,yerr=agg["mlp_shape_std"],
              color=colors_mlp, label="MLP", capsize=5,edgecolor="k",lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(regime_labels)
    ax.set_ylabel("Shape y RMSE (mm)"); ax.set_title("(b) Shape y RMSE")
    ax.legend(); ax.grid(True,axis="y",alpha=0.3)
    for bar,v in zip(list(b1)+list(b2),
                     list(agg["pinn_shape_mean"])+list(agg["mlp_shape_mean"])):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,
                f"{v:.2f}",ha="center",va="bottom",fontsize=7.5)

    # (2) EI_eff identification error
    ax=axes[2]
    b1=ax.bar(x,agg["EI_err_mean"],0.5,yerr=agg["EI_err_std"],
              color=colors_pinn,label="PINN EI err",capsize=5,edgecolor="k",lw=0.8)
    ax.axhline(5,color="k",ls="--",lw=1,alpha=0.6,label="5% threshold")
    ax.set_xticks(x); ax.set_xticklabels(regime_labels)
    ax.set_ylabel("EI_eff identification error (%)"); ax.set_title("(c) EI_eff error")
    ax.legend(); ax.grid(True,axis="y",alpha=0.3)
    for bar,v in zip(b1,agg["EI_err_mean"]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,
                f"{v:.1f}%",ha="center",va="bottom",fontsize=7.5)

    plt.tight_layout()
    out=os.path.join(out_root,"exp2_data_regime_comparison.png")
    plt.savefig(out,dpi=200,bbox_inches="tight"); plt.close()
    print(f"\n  Saved: {out}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["1","2","all"], default="all",
                        help="Which experiment(s) to run (default: all)")
    parser.add_argument("--n_iters", type=int, default=4000,
                        help="PINN training iterations (default: 4000, original)")
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="Number of seeds for Exp 2 (default: 3)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.exp in ("1","all"):
        df1 = exp1_blocked_force(n_iters=args.n_iters)
        # Also copy summary figure to results/
        import shutil
        shutil.copy("results/exp1/exp1_blocked_force_ablation.png",
                    "results/exp1_blocked_force_ablation.png")

    if args.exp in ("2","all"):
        df2 = exp2_data_regimes(n_iters_pinn=args.n_iters,
                                n_iters_mlp=3000,
                                n_seeds=args.n_seeds)
        import shutil
        shutil.copy("results/exp2/exp2_data_regime_comparison.png",
                    "results/exp2_data_regime_comparison.png")

    print("\n✓  All experiments complete.")


if __name__ == "__main__":
    main()
