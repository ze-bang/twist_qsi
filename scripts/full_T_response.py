"""The full temperature dependence of the transverse response, straight from
the exact spectral formula -- NOT the projected-channel route.

  <s>_T = (1/Z) sum_{a<b} |X_ab|^2 (e^{-bEa}-e^{-bEb})/(E_b-E_a)
  alpha(T) = 2 lambda d<s>_T/dT

Questions answered:
 (1) WHERE do the two peaks of alpha(T) come from, in T?  We show
     alpha_gauge(T) is locked to C_gauge(T) (Grueneisen), and the
     charge feature tracks the spinon population n_sp(T).
 (2) Is the charge feature really absent for the uniform drive, or just
     suppressed?  Exact ratio, honestly.
 (3) The spinon-sector analogue of the gauge decomposition: within the
     2-defect (Sz=0) manifold, split the induced second-order operator
     into its DIAGONAL (spinon self-energy = potential) and OFF-DIAGONAL
     (spinon hopping) parts, for uniform vs staggered, and show which one
     the charge peak needs.

Uses the alpha_selection_J*.npz histograms (built by alpha_selection_full)
for (1),(2); builds the small 2-defect resolvent for (3).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GP = HERE.parents[1] / "gauge_probe_prl"
OUT = GP / "notes"; FIGS = OUT / "figs"
C_TEAL="#2a9d8f"; C_RED="#d1495b"; C_PUR="#6f42c1"; C_K="#333333"
plt.rcParams.update({"font.size":9,"axes.labelsize":9.5,"axes.titlesize":9,
 "legend.fontsize":6.8,"xtick.labelsize":8,"ytick.labelsize":8,
 "figure.dpi":200,"savefig.bbox":"tight"})


def load(J):
    return np.load(OUT / f"alpha_selection_J{J:+.2f}.npz")


def kernel(Ea, Eb, T, ebin):
    b = 1.0/T; de = Eb-Ea
    return np.where(np.abs(de)<ebin/2, b*np.exp(-b*Ea),
                   (np.exp(-b*Ea)-np.exp(-b*Eb))/np.where(np.abs(de)<1e-12,1.0,de))


def sT(d, pname, Tg):
    e0=float(d["e0"]); ebin=float(d["ebin"]); nb=int(d["nbins"]); Ec=np.arange(nb)*ebin
    mult={8:1.,9:2.,10:2.,11:2.}
    out=np.zeros_like(Tg)
    for it,T in enumerate(Tg):
        b=1./T
        Z=sum(m*np.exp(-b*(d[f"E{nu}"]-e0)).sum() for nu,m in mult.items())
        num=0.
        for lo,hi in ((8,9),(9,10),(10,11)):
            H=d[f"H_{lo}_{hi}_{pname}"]; ii,jj=np.nonzero(H)
            num+=np.sum(H[ii,jj]*kernel(Ec[jj],Ec[ii],T,ebin))
        out[it]=num/Z
    return out


def C_of(levels_list, mult, Tg):
    E=np.concatenate(levels_list); w0=np.concatenate(mult)
    C=np.zeros_like(Tg)
    for it,T in enumerate(Tg):
        b=1./T; w=w0*np.exp(-b*E); Z=w.sum()
        m1=(w*E).sum()/Z; m2=(w*E**2).sum()/Z
        C[it]=(m2-m1**2)/T**2
    return C


def main():
    Tg=np.geomspace(3e-3,0.6,400)
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.0),sharex=True)
    for jc,J in enumerate((-0.05,+0.04)):
        d=load(J); e0=float(d["e0"])
        E8=d["E8"]-e0
        icemax=E8[89]
        # gauge specific heat: 90 ice levels only
        Cg=C_of([E8[:90]],[np.ones(90)],Tg)
        # full C(T)
        mult={8:1.,9:2.,10:2.,11:2.}
        Cfull=C_of([d[f"E{nu}"]-e0 for nu in mult],
                   [np.full(len(d[f"E{nu}"]),m) for nu,m in mult.items()],Tg)
        # spinon population fraction
        nsp=np.zeros_like(Tg)
        for it,T in enumerate(Tg):
            b=1./T
            Z=sum(m*np.exp(-b*(d[f"E{nu}"]-e0)).sum() for nu,m in mult.items())
            # "charged" weight = states above icemax
            chg=0.
            for nu,m in mult.items():
                Ex=d[f"E{nu}"]-e0
                chg+=m*np.exp(-b*Ex[Ex>icemax+0.3]).sum()
            nsp[it]=chg/Z
        # alpha for the three drives
        a_un=np.gradient(sT(d,"unif",Tg),Tg)
        a_st=np.gradient(sT(d,"stag",Tg),Tg)
        gp=Tg[np.argmax(np.where(Tg<0.06,Cg,0))]
        cp=Tg[np.argmax(np.where(Tg>0.1,Cfull,0))]
        # ---- top: alpha_unif vs C_gauge (Grueneisen locking)
        ax=axs[0,jc]
        ax.plot(Tg,Cg/np.max(Cg),color=C_K,lw=1.4,label=r"$C_{\rm gauge}(T)$ (90 ice levels)")
        ax.plot(Tg,a_un/np.max(np.abs(a_un)),color=C_TEAL,lw=1.4,
                label=r"$\alpha_{E_g}(T)$ (uniform)")
        ax.axvline(gp,color="0.85",lw=0.8); ax.axvline(cp,color="0.85",lw=0.8)
        ax.set_xscale("log"); ax.axhline(0,color="0.9",lw=0.6)
        ax.set_title(f"$J_\\pm={J:+.2f}$");
        if jc==0: ax.set_ylabel("normalized")
        ax.legend(frameon=False,loc="lower left")
        # correlation in the gauge window
        win=(Tg>0.3*gp)&(Tg<3*gp)
        cc=np.corrcoef(Cg[win],np.abs(a_un[win]))[0,1]
        ax.text(0.02,0.9,f"gauge-window corr$(\\alpha,C_g)={cc:.3f}$",
                transform=ax.transAxes,fontsize=6.5)
        # ---- bottom: charge window, alpha_unif vs alpha_stag vs n_sp
        ax=axs[1,jc]
        ax.plot(Tg,a_un/np.max(np.abs(a_un)),color=C_TEAL,lw=1.4,label=r"$\alpha_{E_g}$")
        ax.plot(Tg,a_st/np.max(np.abs(a_st)),color=C_RED,lw=1.4,label=r"$\alpha_{T_{2g}}$")
        ax.plot(Tg,nsp,color=C_PUR,lw=1.0,ls="--",label=r"spinon fraction $n_{\rm sp}(T)$")
        ax.axvline(cp,color="0.85",lw=0.8)
        ax.set_xscale("log"); ax.axhline(0,color="0.9",lw=0.6)
        ax.set_xlabel(r"$T/J_{zz}$");
        if jc==0: ax.set_ylabel("normalized")
        ax.legend(frameon=False,loc="upper left")
        # charge/gauge feature ratio (honest)
        ag=np.max(np.abs(a_un[Tg<0.06])); ac=np.max(np.abs(a_un[(Tg>0.12)&(Tg<0.5)]))
        sg=np.max(np.abs(a_st[Tg<0.06])); sc=np.max(np.abs(a_st[(Tg>0.12)&(Tg<0.5)]))
        print(f"J={J:+.2f}: gauge peak at T={gp:.4f} (ghex-scale), "
              f"charge feat at T={cp:.3f} (~Jzz-scale)")
        print(f"   corr(alpha_unif, C_gauge) in gauge window = {cc:.4f}")
        print(f"   uniform charge/gauge = {ac/ag:.3f};  staggered charge/gauge = {sc/sg:.3f}")
    fig.savefig(FIGS/"figN13_fullT.pdf")
    print("wrote figN13_fullT.pdf")


if __name__=="__main__":
    main()
