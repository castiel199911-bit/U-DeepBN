import argparse
import os
from pathlib import Path
import json
import math
import random

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageEnhance

import time

# =========================
# Folder & listing
# =========================
import json

# -----------------------
# Local fidelity metrics
# -----------------------
def _predict_proba_bundle(bundle, X):
    """Return (yhat, proba) for any supported BN bundle."""
    if bundle['type']=='naive':
        proba = bundle['model'].predict_proba(X)
        yhat = np.argmax(proba, axis=1)
        return yhat, proba
    elif bundle['type']=='chowliu':
        yhat, proba = predict_chowliu(bundle, X)
        return yhat, proba
    elif bundle['type']=='tan':
        # Use the same TAN predictor used elsewhere in this script
        yhat, proba = predict_tan_uncertainty(bundle, X)
        return yhat, proba
    else:
        yhat, proba = predict_k2(bundle, X)
        return yhat, proba


def _estimate_importance_by_resampling(bundle, x0, p_full, y_pred, X_ref, n_mc=20, rng=None):
    """Approximate per-feature local importance by resampling that feature from X_ref."""
    if rng is None:
        rng = np.random.default_rng(0)
    d = x0.shape[0]
    imps = np.zeros(d, dtype=float)
    for j in range(d):
        probs = []
        idx = rng.integers(0, X_ref.shape[0], size=n_mc)
        for t in idx:
            x = x0.copy()
            x[j] = X_ref[t, j]
            _, p = _predict_proba_bundle(bundle, x[None, :])
            probs.append(p[0])
        p_marg = np.mean(np.vstack(probs), axis=0)
        # Importance = expected drop in predicted-class probability when feature j is "unknown"
        imps[j] = max(0.0, float(p_full[y_pred] - p_marg[y_pred]))
    return imps


# =========================
# MI & selection helpers
# =========================

FEATURE_EXCLUDE = {"label","image_path"}


# =========================
# Discretization & Info
# =========================

def quantile_bins_fit(X, n_bins):
    d = X.shape[1]
    edges = []
    qs = np.linspace(0,1,n_bins+1); qs[0],qs[-1]=0.0,1.0
    for j in range(d):
        col = X[:,j]
        e = np.unique(np.quantile(col, qs))
        if len(e)<=2:
            e = np.array([col.min()-1e-6, col.max()+1e-6])
        edges.append(e)
    return edges

def quantize_with_edges(X, edges):
    Xb = np.zeros_like(X, dtype=np.int16)
    for j,e in enumerate(edges):
        Xb[:,j] = np.clip(np.digitize(X[:,j], e[1:-1], right=False), 0, len(e)-2)
    return Xb

def _mi_discrete(x, y, smooth=1.0):
    kx = int(x.max())+1
    ky = int(y.max())+1
    joint = np.zeros((kx,ky), dtype=float)
    for a,b in zip(x,y): joint[a,b]+=1.0
    joint += smooth; joint/=joint.sum()
    px = joint.sum(axis=1, keepdims=True); py=joint.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        mi = np.nansum(joint * np.log(joint/(px@py)))
    return float(max(mi,0.0))

def _cmi_given_class(x,z,cls,smooth=1.0):
    res=0.0
    for c in np.unique(cls):
        m=(cls==c)
        if m.sum()==0: continue
        res += m.mean()*_mi_discrete(x[m], z[m], smooth=smooth)
    return float(max(res,0.0))

# =========================
# Chow–Liu / TAN / K2
# =========================

def train_chowliu(df, n_bins=8, alpha=1.0):
    feat_cols=[c for c in df.columns if c not in FEATURE_EXCLUDE]
    X=df[feat_cols].values.astype(float)
    y,class_names = pd.factorize(df['label'])
    C=len(class_names); d=X.shape[1]
    edges=quantile_bins_fit(X, n_bins)
    Xb=quantize_with_edges(X, edges)

    models={}  # per-class parents & CPTs
    arities=[len(e)-1 for e in edges]

    for c in range(C):
        m=(y==c)
        Xc = Xb[m]
        W=np.zeros((d,d),dtype=float)
        for i in range(d):
            for j in range(i+1,d):
                W[i,j]=W[j,i]=_mi_discrete(Xc[:,i], Xc[:,j], smooth=alpha)
        selected=[0]; parents=[-1]+[None]*(d-1); remaining=set(range(1,d))
        while remaining:
            best=None; bw=-1.0
            for u in selected:
                for v in remaining:
                    if W[u,v]>bw:
                        bw=W[u,v]; best=(u,v)
            u,v=best; parents[v]=u; selected.append(v); remaining.remove(v)
        cpts={}
        for i in range(d):
            ki=arities[i]
            if parents[i]==-1:
                tab=np.zeros((ki,),dtype=float)+alpha
                for xi in Xc[:,i]: tab[xi]+=1.0
                tab /= tab.sum()
                cpts[f"{i}"]= {"parents":[], "table":tab}
            else:
                kp=arities[parents[i]]
                tab=np.zeros((ki,kp),dtype=float)+alpha
                for xi,xp in zip(Xc[:,i], Xc[:,parents[i]]): tab[xi,xp]+=1.0
                den=tab.sum(axis=0, keepdims=True); den=np.where(den==0,1.0,den)
                tab/=den
                cpts[f"{i}"]={"parents":[parents[i]], "table":tab}
        models[c]={"parents":parents, "cpts":cpts}

    priors=(np.bincount(y)/len(y)).astype(float)
    return {
        "type":"chowliu",
        "feature_names":feat_cols,
        "class_names":list(class_names),
        "bin_edges":edges,
        "arities":arities,
        "models":models,
        "priors":priors,
    }

def predict_chowliu(bundle, X):
    edges=bundle['bin_edges']; Xb=quantize_with_edges(X, edges)
    n=Xb.shape[0]; C=len(bundle['class_names']); d=Xb.shape[1]
    logp=np.log(bundle['priors']+1e-12)[None,:].repeat(n,axis=0)
    for c in range(C):
        parents=bundle['models'][c]['parents']
        cpts=bundle['models'][c]['cpts']
        ll=np.zeros(n)
        for i in range(d):
            if parents[i]==-1:
                tab=cpts[f"{i}"]['table']  # (ki,)
                ki=tab.shape[0]
                xi=np.clip(Xb[:,i], 0, ki-1)
                ll += np.log(tab[xi]+1e-12)
            else:
                p=parents[i]
                tab=cpts[f"{i}"]['table']  # (ki,kp)
                ki,kp=tab.shape
                xi=np.clip(Xb[:,i],0,ki-1)
                xp=np.clip(Xb[:,p],0,kp-1)
                ll += np.log(tab[xi, xp]+1e-12)
        logp[:,c]+=ll
    maxlog=logp.max(axis=1,keepdims=True); prob=np.exp(logp-maxlog); prob/=prob.sum(axis=1,keepdims=True)
    yhat=np.argmax(prob,axis=1)
    return yhat, prob

# TAN

def train_tan(df, bins=8, alpha=1.0):
    feat_cols=[c for c in df.columns if c not in FEATURE_EXCLUDE]
    X=df[feat_cols].values.astype(float)
    y,class_names = pd.factorize(df['label'])
    C=len(class_names); d=X.shape[1]
    edges=quantile_bins_fit(X, bins)
    Xb=quantize_with_edges(X, edges)
    arities=[len(e)-1 for e in edges]
    W=np.zeros((d,d),dtype=float)
    for i in range(d):
        for j in range(i+1,d):
            W[i,j]=W[j,i]=_cmi_given_class(Xb[:,i], Xb[:,j], y, smooth=alpha)
    selected=[0]; parents_tree=[-1]+[None]*(d-1); rem=set(range(1,d))
    while rem:
        best=None; bw=-1.0
        for u in selected:
            for v in rem:
                if W[u,v]>bw:
                    bw=W[u,v]; best=(u,v)
        u,v=best; parents_tree[v]=u; selected.append(v); rem.remove(v)
    cpts={}
    for i in range(d):
        ki=arities[i]
        if parents_tree[i]==-1:
            tab=np.zeros((ki,C),dtype=float)+alpha
            for xi,ci in zip(Xb[:,i], y): tab[xi,ci]+=1.0
            tab/=tab.sum(axis=0,keepdims=True)
            cpts[f"{i}|C"]=tab
        else:
            kp=arities[parents_tree[i]]
            tab=np.zeros((ki,C,kp),dtype=float)+alpha
            for xi,ci,pi in zip(Xb[:,i], y, Xb[:,parents_tree[i]]): tab[xi,ci,pi]+=1.0
            den=tab.sum(axis=0,keepdims=True); den=np.where(den==0,1.0,den)
            tab/=den
            cpts[f"{i}|C,{parents_tree[i]}"]=tab
    priors=(np.bincount(y)/len(y)).astype(float)
    return {
        'type':'tan', 'feature_names':feat_cols, 'class_names':list(class_names),
        'bin_edges':edges, 'arities':arities, 'parents_tree':parents_tree, 'cpts':cpts, 'priors':priors
    }

def predict_tan(model, X):
    edges=model['bin_edges']
    Xb=quantize_with_edges(X, edges)
    n=Xb.shape[0]; C=len(model['class_names']); d=Xb.shape[1]
    logp=np.log(model['priors']+1e-12)[None,:].repeat(n,axis=0)
    for i in range(d):
        parent=model['parents_tree'][i]
        if parent==-1:
            tab=model['cpts'][f"{i}|C"]  # (ki,C)
            ki=tab.shape[0]
            xi=np.clip(Xb[:,i],0,ki-1)
            for c in range(C):
                logp[:,c]+=np.log(tab[xi,c]+1e-12)
        else:
            tab=model['cpts'][f"{i}|C,{parent}"]  # (ki,C,kp)
            ki,_,kp=tab.shape
            xi=np.clip(Xb[:,i],0,ki-1)
            xp=np.clip(Xb[:,parent],0,kp-1)
            for c in range(C):
                logp[:,c]+=np.log(tab[xi,c,xp]+1e-12)
    maxlog=logp.max(axis=1,keepdims=True); prob=np.exp(logp-maxlog); prob/=prob.sum(axis=1,keepdims=True)
    yhat=np.argmax(prob,axis=1)
    return yhat, prob
    
def _tan_build_children(parents_tree):
    d = len(parents_tree)
    children = [[] for _ in range(d)]
    root = None
    for i, p in enumerate(parents_tree):
        if p == -1:
            root = i
        else:
            children[p].append(i)
    if root is None:
        raise ValueError("TAN parents_tree has no root (-1).")
    return root, children

def tan_posterior_given_subset(bundle, x_continuous, observed_idx):
    """
    Exact posterior p(y | x_S) for TAN by summing out unobserved features.

    bundle: TAN dict from train_tan()
    x_continuous: shape (d,) float features
    observed_idx: iterable of feature indices to KEEP (observe)
    returns: posterior over classes shape (C,)
    """
    # Discretize
    edges = bundle["bin_edges"]
    x_b = quantize_with_edges(np.asarray(x_continuous, dtype=float)[None, :], edges)[0]  # (d,)

    parents = bundle["parents_tree"]
    arities = bundle["arities"]
    priors = np.asarray(bundle["priors"], dtype=float)
    C = len(bundle["class_names"])
    d = len(arities)

    observed = np.zeros(d, dtype=bool)
    observed[list(observed_idx)] = True

    root, children = _tan_build_children(parents)

    # phi_i: evidence indicator; if observed -> one-hot; else -> all ones
    # We'll avoid allocating full matrices; just handle observed/unobserved cases in sums.

    # Messages: for each node i != root we store msg_i_to_parent with shape (C, k_parent)
    msg = {}

    # Postorder traversal to compute child->parent messages
    stack = [(root, 0)]  # (node, state) where 0=enter, 1=exit
    postorder = []
    while stack:
        node, st = stack.pop()
        if st == 0:
            stack.append((node, 1))
            for ch in children[node]:
                stack.append((ch, 0))
        else:
            postorder.append(node)

    # Compute messages bottom-up
    for i in postorder:
        p = parents[i]
        if p == -1:
            continue  # root handled later

        kp = arities[p]
        ki = arities[i]

        tab = bundle["cpts"][f"{i}|C,{p}"]  # (ki, C, kp)

        # product of incoming messages from i's children -> depends on x_i state
        # incoming_prod[c, x_i]
        incoming_prod = np.ones((C, ki), dtype=float)
        for ch in children[i]:
            # child message is m_{ch->i}(c, x_i)
            incoming_prod *= msg[ch]  # (C, ki)

        out = np.zeros((C, kp), dtype=float)

        if observed[i]:
            xi = int(np.clip(x_b[i], 0, ki - 1))
            # sum over x_i collapses to that single state
            # m_{i->p}(c, x_p) = phi_i(xi)*P(xi|c,x_p)*prod_children
            out = tab[xi, :, :] * incoming_prod[:, xi][:, None]  # (C, kp)
        else:
            # sum over all x_i states
            # out[c, x_p] = Σ_xi P(xi|c,x_p) * incoming_prod[c, xi]
            # Do it as a loop over ki (ki usually small)
            for xi in range(ki):
                out += tab[xi, :, :] * incoming_prod[:, xi][:, None]

        msg[i] = out  # (C, kp)

    # Now compute class likelihood at root: p(x_S | y=c)
    kr = arities[root]
    root_tab = bundle["cpts"][f"{root}|C"]  # (kr, C)

    incoming_prod_root = np.ones((C, kr), dtype=float)
    for ch in children[root]:
        incoming_prod_root *= msg[ch]  # msg[ch] has shape (C, kr)

    like = np.zeros(C, dtype=float)

    if observed[root]:
        xr = int(np.clip(x_b[root], 0, kr - 1))
        like = root_tab[xr, :] * incoming_prod_root[:, xr]
    else:
        # like[c] = Σ_xr P(xr|c) * incoming_prod_root[c, xr]
        for xr in range(kr):
            like += root_tab[xr, :] * incoming_prod_root[:, xr]

    unnorm = priors * np.clip(like, 1e-300, None)
    s = unnorm.sum()
    if s <= 0 or not np.isfinite(s):
        # fallback: uniform
        return np.ones(C, dtype=float) / C
    return unnorm / s

def _kl_div(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, 1.0); p = p / p.sum()
    q = np.clip(q, eps, 1.0); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

def tan_local_fidelity_exact(bundle, X_te, subsets_by_k, proba_full=None, n_instances=None, seed=0):
    """
    Exact TAN local fidelity for provided subsets.

    subsets_by_k: dict like {k: list_of_idx_arrays}
      - For each k, you provide a list length N where each element is
        an iterable of feature indices kept for that instance.

      Example: subsets_by_k[5][i] = top-5 indices for instance i

    proba_full: optional (N,C) from predict_tan(bundle, X_te)
    n_instances: optional limit (random sample of instances)
    """
    rng = np.random.default_rng(seed)
    X_te = np.asarray(X_te, dtype=float)
    N = X_te.shape[0]

    if proba_full is None:
        _, proba_full = predict_tan(bundle, X_te)
    proba_full = np.asarray(proba_full, dtype=float)

    if n_instances is None or n_instances >= N:
        idxs = np.arange(N)
    else:
        idxs = rng.choice(N, size=int(n_instances), replace=False)

    out = {}
    for k, subsets in subsets_by_k.items():
        kl_list, l1_list, tv_fid_list = [], [], []
        agree = 0
        used = 0

        for i in idxs:
            S = subsets[i]
            p_full = proba_full[i]
            if not np.isfinite(p_full).all() or p_full.sum() <= 0:
                continue
            p_full = np.clip(p_full, 1e-12, 1.0); p_full = p_full / p_full.sum()
            y_pred = int(np.argmax(p_full))

            p_sub = tan_posterior_given_subset(bundle, X_te[i], S)
            p_sub = np.clip(p_sub, 1e-12, 1.0); p_sub = p_sub / p_sub.sum()

            kl = _kl_div(p_full, p_sub)
            l1 = float(np.sum(np.abs(p_full - p_sub)))
            # fidelity in [0,1] = 1 - total variation distance
            fid = float(max(0.0, 1.0 - 0.5 * l1))
            agree += int(np.argmax(p_sub) == y_pred)

            kl_list.append(kl)
            l1_list.append(l1)
            tv_fid_list.append(fid)
            used += 1

        out[int(k)] = {
            "n_instances": int(used),
            "mean_local_fidelity_norm": float(np.mean(tv_fid_list)) if used else float("nan"),
            "mean_KL": float(np.mean(kl_list)) if used else float("nan"),
            "mean_L1": float(np.mean(l1_list)) if used else float("nan"),
            "label_agreement": float(agree / used) if used else float("nan"),
        }

    return out

# K2

def _bic_score(counts):
    totals=counts.sum(axis=0,keepdims=True)
    probs=counts/np.maximum(totals,1e-12)
    ll=np.nansum(counts*np.log(np.maximum(probs,1e-12)))
    N=counts.sum(); r,q=counts.shape; num_free=(r-1)*q; penalty=0.5*num_free*math.log(max(N,1.0))
    return float(ll-penalty)

def train_k2(df, bins=8, alpha=1.0, max_parents=2):
    feat_cols=[c for c in df.columns if c not in FEATURE_EXCLUDE]
    X=df[feat_cols].values.astype(float)
    y,class_names = pd.factorize(df['label'])
    C=len(class_names); d=X.shape[1]
    edges=quantile_bins_fit(X,bins); Xb=quantize_with_edges(X, edges)
    arities=[len(e)-1 for e in edges]
    parents=[[] for _ in range(d)]; include_class=[False]*d

    def make_counts(i, pa, incl_class):
        ki=arities[i]
        if incl_class:
            par_ar=[C]+[arities[p] for p in pa]
            multip=np.concatenate(([1], np.cumprod(par_ar[:-1])))
            q=int(np.prod(par_ar)) if par_ar else 1
            counts=np.zeros((ki,q),dtype=float)+alpha
            for n in range(len(y)):
                state=[y[n]]+[Xb[n,p] for p in pa]
                idx=int(np.dot(state, multip))
                counts[Xb[n,i], idx]+=1.0
            return counts
        else:
            par_ar=[arities[p] for p in pa]
            multip=np.concatenate(([1], np.cumprod(par_ar[:-1]))) if par_ar else np.array([1])
            q=int(np.prod(par_ar)) if par_ar else 1
            counts=np.zeros((ki,q),dtype=float)+alpha
            for n in range(len(y)):
                idx=int(np.dot([Xb[n,p] for p in pa], multip)) if pa else 0
                counts[Xb[n,i], idx]+=1.0
            return counts

    for i in range(d):
        cand=list(range(i)); incl=True; cur=[]
        best=_bic_score(make_counts(i, cur, incl)); improved=True
        while improved and (len(cur)<max_parents or incl):
            improved=False; best_add=None
            if incl:
                sc=_bic_score(make_counts(i, cur, True))
                if sc>best: best=sc; best_add='CLASS'
            for p in [p for p in cand if p not in cur]:
                sc=_bic_score(make_counts(i, cur+[p], incl))
                if sc>best: best=sc; best_add=p
            if best_add is not None:
                improved=True
                if best_add=='CLASS': incl=False; include_class[i]=True
                else: cur.append(best_add)
        parents[i]=cur

    cpts={}
    for i in range(d):
        ki=arities[i]; pa=parents[i]
        if include_class[i]:
            par_ar=[C]+[arities[p] for p in pa]
            multip=np.concatenate(([1], np.cumprod(par_ar[:-1])))
            q=int(np.prod(par_ar)) if par_ar else 1
            counts=np.zeros((ki,q),dtype=float)+alpha
            for n in range(len(y)):
                state=[y[n]]+[Xb[n,p] for p in pa]
                idx=int(np.dot(state,multip))
                counts[Xb[n,i], idx]+=1.0
            totals=counts.sum(axis=0,keepdims=True); totals=np.where(totals==0,1.0,totals)
            table=counts/totals
            cpts[str(i)]={"parents":["CLASS"]+pa, "table":table}
        else:
            par_ar=[arities[p] for p in pa]
            multip=np.concatenate(([1], np.cumprod(par_ar[:-1]))) if pa else np.array([1])
            q=int(np.prod(par_ar)) if pa else 1
            counts=np.zeros((ki,q),dtype=float)+alpha
            for n in range(len(y)):
                idx=int(np.dot([Xb[n,p] for p in pa], multip)) if pa else 0
                counts[Xb[n,i], idx]+=1.0
            totals=counts.sum(axis=0,keepdims=True); totals=np.where(totals==0,1.0,totals)
            table=counts/totals
            cpts[str(i)]={"parents":pa, "table":table}

    priors=(np.bincount(y)/len(y)).astype(float)
    return {
        'type':'k2','feature_names':feat_cols,'class_names':list(class_names),
        'bin_edges':edges,'arities':[len(e)-1 for e in edges],'parents':cpts,'priors':priors
    }

def predict_k2(model, X):
    edges=model['bin_edges']; Xb=quantize_with_edges(X, edges)
    n=Xb.shape[0]; C=len(model['class_names']); d=Xb.shape[1]
    logp=np.log(model['priors']+1e-12)[None,:].repeat(n,axis=0)
    for c in range(C):
        for i in range(d):
            spec=model['parents'][str(i)]
            tab=spec['table']  # (ki,q)
            ki=tab.shape[0]
            pa=spec['parents']
            if 'CLASS' in pa:
                par_feats=[p for p in pa if p!='CLASS']
                par_ar=[C]+[model['arities'][p] for p in par_feats]
                multip=np.concatenate(([1], np.cumprod(par_ar[:-1])))
                idx_parent=np.zeros(n,dtype=int)+c*multip[0]
                for k,p in enumerate(par_feats):
                    idx_parent += np.clip(Xb[:,p],0,par_ar[k+1]-1)*multip[k+1]
            else:
                par_feats=pa
                if len(par_feats)==0:
                    idx_parent=np.zeros(n,dtype=int)
                else:
                    par_ar=[model['arities'][p] for p in par_feats]
                    multip=np.concatenate(([1], np.cumprod(par_ar[:-1])))
                    idx_parent=np.zeros(n,dtype=int)
                    for k,p in enumerate(par_feats):
                        idx_parent += np.clip(Xb[:,p],0,par_ar[k]-1)*multip[k]
            xi=np.clip(Xb[:,i],0,ki-1)
            logp[:,c]+=np.log(tab[xi, idx_parent]+1e-12)
    maxlog=logp.max(axis=1,keepdims=True); prob=np.exp(logp-maxlog); prob/=prob.sum(axis=1,keepdims=True)
    yhat=np.argmax(prob,axis=1)
    return yhat, prob

# =========================
# Plot BN + class node
# =========================

def save_full_bn_png(structure, bundle, out_path, mi_df=None, k=12, class_index=0):
    feat_names=bundle['feature_names']
    if k is None:
        show_feats=feat_names[:]
    else:
        if mi_df is not None and len(mi_df)>0:
            ranked=[f for f in mi_df['feature'].tolist() if f in feat_names]
            show_feats=ranked[:k]
        else:
            show_feats=feat_names[:k]

    G=nx.DiGraph(); DISEASE='Disease'
    G.add_node(DISEASE)
    for f in show_feats:
        G.add_node(f); G.add_edge(DISEASE,f)

    solid_edges=[]
    stype=bundle.get('type',structure).lower()
    if stype=='chowliu':
        models=bundle['models']; class_id=class_index
        parents=models[class_id]['parents']
        fn=bundle['feature_names']
        for child_idx,parent_idx in enumerate(parents):
            if parent_idx==-1: continue
            cf,pf=fn[child_idx], fn[parent_idx]
            if cf in show_feats and pf in show_feats:
                solid_edges.append((pf,cf))
    elif stype=='tan':
        parents=bundle['parents_tree']; fn=bundle['feature_names']
        for child_idx,parent_idx in enumerate(parents):
            if parent_idx==-1: continue
            cf,pf=fn[child_idx], fn[parent_idx]
            if cf in show_feats and pf in show_feats:
                solid_edges.append((pf,cf))
    elif stype=='k2':
        specs=bundle['parents']; fn=bundle['feature_names']
        d=len(fn)
        for i in range(d):
            spec=specs.get(str(i))
            if spec is None: continue
            for p in spec['parents']:
                if p=='CLASS': continue
                pf,cf=fn[p], fn[i]
                if pf in show_feats and cf in show_feats:
                    solid_edges.append((pf,cf))

    pos=nx.spring_layout(G, seed=42, k=0.6)
    

# =========================
# Main
# =========================

def train_tan_uncertainty(
    df,
    bins=8,
    alpha=1.0,
    uncertainty_csv_path="feature_uncertainty_per_class_dog_test.csv",
    uncertainty_col="total",
    bins_min=3,
    bins_max=16,
    alpha_beta=2.0,
):
    """
    TAN with uncertainty-aware:
      - per-feature binning (coarser bins for uncertain features)
      - CMI scaled by reliability for MST
      - per-feature Dirichlet smoothing (alpha_i)
    Returns a model dict compatible with predict_tan_uncertainty (and similar to train_tan).
    """
    import numpy as np
    import pandas as pd

    # ---- data & feature names (same as train_tan) ----
    feat_cols = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    X = df[feat_cols].values.astype(float)
    y, class_names = pd.factorize(df["label"])
    C = len(class_names)
    d = X.shape[1]
    #print('IN uncertainty ------------ A')

    # ---- load per-feature uncertainty -> reliability r in [0,1] ----
    def _load_feature_reliability(csv_path, col):
        try:
            u = pd.read_csv(csv_path)
        except Exception:
            return None
        if "feature" not in u.columns or col not in u.columns:
            return None
        g = u.groupby("feature")[col].mean()  # per-feature mean uncertainty
        vals = g.values
        order = np.argsort(vals)  # low uncertainty first
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = (np.arange(len(vals)) + 1) / len(vals)  # 0..1
        u_rank = pd.Series(ranks, index=g.index)  # larger = more uncertain
        r = (1.0 - u_rank).clip(0.0, 1.0)        # reliability
        return r.to_dict()

    rmap = _load_feature_reliability(uncertainty_csv_path, uncertainty_col)
    if rmap is None:
        r = np.ones(d, dtype=float)
    else:
        r = np.array([rmap.get(f, 1.0) for f in feat_cols], dtype=float)
        r = np.nan_to_num(r, nan=1.0).clip(0.0, 1.0)

    # ---- per-feature quantile binning (coarser for uncertain features) ----
    def _quantile_bins_fit_per_feature(X, base_bins, reliability,
                                       bins_min=3, bins_max=16):
        d = X.shape[1]
        bins_i = np.clip(
            np.round(bins_min + (bins_max - bins_min) * reliability).astype(int),
            bins_min, bins_max
        )
        edges = []
        for j in range(d):
            qs = np.linspace(0, 1, bins_i[j] + 1); qs[0], qs[-1] = 0.0, 1.0
            e = np.unique(np.quantile(X[:, j], qs))
            if len(e) <= 2:
                # fallback to two bins if feature is (near-)constant
                e = np.array([X[:, j].min() - 1e-6, X[:, j].max() + 1e-6])
            edges.append(e)
        return edges

    edges = _quantile_bins_fit_per_feature(X, bins, r, bins_min=bins_min, bins_max=bins_max)
    Xb = quantize_with_edges(X, edges)
    arities = [len(e) - 1 for e in edges]

    # ---- CMI matrix scaled by reliability -> maximum spanning tree (same prim-style as your code) ----
    W = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(i + 1, d):
            cmi = _cmi_given_class(Xb[:, i], Xb[:, j], y, smooth=alpha)
            W[i, j] = W[j, i] = cmi * (0.5 * (r[i] + r[j]))  # reliability scaling

    selected = [0]
    parents_tree = [-1] + [None] * (d - 1)
    rem = set(range(1, d))
    while rem:
        best = None
        bw = -1.0
        for u in selected:
            for v in rem:
                if W[u, v] > bw:
                    bw = W[u, v]
                    best = (u, v)
        u, v = best
        parents_tree[v] = u
        selected.append(v)
        rem.remove(v)

    # ---- CPTs with per-feature alpha_i ----
    alpha_feat = alpha * (1.0 + alpha_beta * (1.0 - r))
    cpts = {}
    for i in range(d):
        ki = arities[i]
        if parents_tree[i] == -1:
            tab = np.zeros((ki, C), dtype=float) + alpha_feat[i]
            for xi, ci in zip(Xb[:, i], y):
                tab[xi, ci] += 1.0
            tab /= tab.sum(axis=0, keepdims=True)
            cpts[f"{i}|C"] = tab
        else:
            kp = arities[parents_tree[i]]
            tab = np.zeros((ki, C, kp), dtype=float) + alpha_feat[i]
            for xi, ci, pi in zip(Xb[:, i], y, Xb[:, parents_tree[i]]):
                tab[xi, ci, pi] += 1.0
            den = tab.sum(axis=0, keepdims=True)
            den = np.where(den == 0, 1.0, den)
            tab /= den
            cpts[f"{i}|C,{parents_tree[i]}"] = tab

    # ---- class priors (same as your train_tan) ----
    priors = (np.bincount(y) / len(y)).astype(float)

    return {
        "type": "tan",
        "feature_names": feat_cols,
        "class_names": list(class_names),
        "bin_edges": edges,
        "arities": arities,
        "parents_tree": parents_tree,
        "cpts": cpts,
        "priors": priors,
        "feature_weights": r.tolist(),  # used by predict_tan_uncertainty
    }


def predict_tan_uncertainty(model, X, feature_weights=None):
    """
    Uncertainty-tempered TAN inference.
    Multiples each feature's log-likelihood by w_i in [0,1].
    Falls back to uniform w_i=1 if no weights provided.
    """
    import numpy as np

    edges = model["bin_edges"]
    Xb = quantize_with_edges(X, edges)
    n = Xb.shape[0]
    C = len(model["class_names"])
    d = Xb.shape[1]
    #print('IN uncertainty -----------------B')

    logp = np.log(model["priors"] + 1e-12)[None, :].repeat(n, axis=0)
    if feature_weights is None:
        w = np.array(model.get("feature_weights", np.ones(d)), dtype=float)
    else:
        w = np.array(feature_weights, dtype=float)
        if w.shape[0] != d:
            raise ValueError(f"feature_weights has length {w.shape[0]} but model has {d} features")

    for i in range(d):
        parent = model["parents_tree"][i]
        if parent == -1:
            tab = model["cpts"][f"{i}|C"]  # (ki, C)
            ki = tab.shape[0]
            xi = np.clip(Xb[:, i], 0, ki - 1)
            for c in range(C):
                logp[:, c] += w[i] * np.log(tab[xi, c] + 1e-12)
        else:
            tab = model["cpts"][f"{i}|C,{parent}"]  # (ki, C, kp)
            ki, _, kp = tab.shape
            xi = np.clip(Xb[:, i], 0, ki - 1)
            xp = np.clip(Xb[:, parent], 0, kp - 1)
            for c in range(C):
                logp[:, c] += w[i] * np.log(tab[xi, c, xp] + 1e-12)

    maxlog = logp.max(axis=1, keepdims=True)
    prob = np.exp(logp - maxlog)
    prob /= prob.sum(axis=1, keepdims=True)
    yhat = np.argmax(prob, axis=1)
    return yhat, prob



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--test-dir', type=str, required=True)
    ap.add_argument('--out-dir', type=str, required=True)
    ap.add_argument('--features', choices=['deep','classic','hybrid'], default='deep')
    ap.add_argument('--structure', choices=['naive','chowliu','tan','k2'], default='naive')
    ap.add_argument('--deep-backbone', choices=['resnet18','resnet50'], default='resnet18')
    ap.add_argument('--deep-dim', type=int, default=128)
    ap.add_argument('--pca-whiten', action='store_true')
    ap.add_argument('--topk', type=int, default=0)
    ap.add_argument('--bins', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--max-parents', type=int, default=2)
    ap.add_argument('--uniform-prior', action='store_true', help='Use uniform class priors instead of empirical frequencies')
    ap.add_argument('--aug', type=int, default=0, help='Number of augmented copies per TRAIN image (0=no augmentation)')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--mi', action='store_true')
    ap.add_argument(
    '--aug-map', type=str, default=None,
    help='JSON dict of per-class augmentation copies, e.g. {"Hypersensitivity":5,"Fungal_infections":4,"default":2}. Overrides --aug if provided.'
)

    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    folder = "runs_new"
    files = os.listdir(folder)

    # Filter and normalize only CSVs starting with train/test
    acc_all = []

    pairs = {}

    for f in files:
        if f.endswith('.csv') and (f.startswith('train') or f.startswith('test')):
            # Extract the "key" after first underscore: e.g. tabular_42.csv
            key = "_".join(f.split("_")[1:])  
            pairs.setdefault(key, {}).update({
                'train' if f.startswith('train') else 'test': f
            })

    count = 0
    # Now iterate through confirmed train/test pairs
    for key, pair in pairs.items():
        if 'train' in pair and 'test' in pair and count<10:
            count+=1
            train_path = os.path.join(folder, pair['train'])
            test_path = os.path.join(folder, pair['test'])
            
            df_tr = pd.read_csv(train_path)
            df_te  = pd.read_csv(test_path)
            
            print('Data:', train_path)
            print(f"Loaded pair: {pair['train']} & {pair['test']}")

            #df_tr = pd.read_csv('art_tan_dog/train_tabular_42.csv')
            # df_te = pd.read_csv('art_tan_dog/test_tabular_42.csv')

            # print('Data loaded.')

            selected_feats = [c for c in df_tr.columns if c not in FEATURE_EXCLUDE]

            # -------- Encode labels (TRAIN fit only)
            le = LabelEncoder().fit(df_tr['label'])
            y_tr = le.transform(df_tr['label']); y_te = le.transform(df_te['label'])
            class_names = le.classes_.tolist(); n_classes=len(class_names)

            # -------- Train model
            print('[3/6] Training BN...')
            if args.structure=='naive':
                X_tr = df_tr[selected_feats].values.astype(float)
                if args.uniform_prior:
                    priors = np.ones(n_classes, dtype=float) / n_classes
                    gnb = GaussianNB(priors=priors)
                else:
                    gnb = GaussianNB()
                gnb.fit(X_tr, y_tr)
                bundle = { 'type':'naive', 'feature_names':selected_feats, 'class_names':class_names, 'model':gnb }
            elif args.structure=='chowliu':
                df_tr_bn = pd.concat([df_tr[selected_feats], pd.Series(le.inverse_transform(y_tr), name='label')], axis=1)
                bundle = train_chowliu(df_tr_bn, n_bins=args.bins, alpha=args.alpha)
                bundle['feature_names']=selected_feats; bundle['class_names']=class_names
                if args.uniform_prior:
                    bundle['priors'] = np.ones(n_classes, dtype=float) / n_classes
            elif args.structure=='tan':
                df_tr_bn = pd.concat([df_tr[selected_feats], pd.Series(le.inverse_transform(y_tr), name='label')], axis=1)
                #bundle = train_tan(df_tr_bn, bins=args.bins, alpha=args.alpha)
                bundle = train_tan_uncertainty(df_tr_bn, bins=args.bins, alpha=args.alpha)
                bundle['feature_names']=selected_feats; bundle['class_names']=class_names
                if args.uniform_prior:
                    bundle['priors'] = np.ones(n_classes, dtype=float) / n_classes
            elif args.structure=='k2':
                df_tr_bn = pd.concat([df_tr[selected_feats], pd.Series(le.inverse_transform(y_tr), name='label')], axis=1)
                bundle = train_k2(df_tr_bn, bins=args.bins, alpha=args.alpha, max_parents=args.max_parents)
                bundle['feature_names']=selected_feats; bundle['class_names']=class_names
                if args.uniform_prior:
                    bundle['priors'] = np.ones(n_classes, dtype=float) / n_classes
            else:
                raise ValueError('Unknown structure')

            # -------- Predict on TEST
            print('[4/6] Predicting on TEST...')
            X_te = df_te[selected_feats].values.astype(float)
            if bundle['type']=='naive':
                yp = bundle['model'].predict(X_te)
                proba = bundle['model'].predict_proba(X_te)
            elif bundle['type']=='chowliu':
                yp, proba = predict_chowliu(bundle, X_te)
            elif bundle['type']=='tan':
                # yp, proba = predict_tan(bundle, X_te)
                yp, proba = predict_tan_uncertainty(bundle, X_te)
            else:
                yp, proba = predict_k2(bundle, X_te)

            # -------- Metrics
            print('[5/6] Metrics...')
            cm = confusion_matrix(y_te, yp, labels=list(range(len(class_names))))
            per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

            for name, acc in zip(class_names, per_class_accuracy):
                print(f"{name}: {acc:.3f}")

            print(accuracy_score(y_te, yp))
            print(classification_report(y_te, yp, target_names=class_names, digits=3))

            acc_all.append(accuracy_score(y_te, yp))

            #print(X_tr)
            X_tr = df_tr[selected_feats].values.astype(float)


            # Save CM plots
            

            out_dir.mkdir(parents=True, exist_ok=True)
            

            # -------- BN graph (top-12 by MI, if available)
            print('[6/6] BN graph...')
            # mi_for_plot = mi_df if mi_df is not None else pd.DataFrame({'feature':selected_feats, 'MI':[0]*len(selected_feats)})
            # save_full_bn_png(args.structure, bundle, out_dir/'bn_graph.png', mi_df=mi_for_plot, k=min(12,len(selected_feats)), class_index=0)

            # -------- Save meta
            (out_dir/'bundle_meta.json').write_text(json.dumps({
                'structure':bundle['type'],
                'class_names':class_names,
                'feature_names':selected_feats,
                'priors': (bundle.get('priors').tolist() if 'priors' in bundle else
                        (bundle['model'].class_prior_.tolist() if bundle['type']=='naive' and hasattr(bundle['model'],'class_prior_') else None)),
                'aug': args.aug
            }, indent=2))
            print('Done. Artifacts in', str(out_dir))
            
            def tan_fast_rank_features(bundle, x_continuous, y_star):
            # discretize
                x_b = quantize_with_edges(np.asarray(x_continuous, float)[None, :], bundle["bin_edges"])[0]
                parents = bundle["parents_tree"]
                d = len(parents)
                scores = np.zeros(d, dtype=float)

                for i in range(d):
                    p = parents[i]
                    if p == -1:
                        tab = bundle["cpts"][f"{i}|C"]  # (ki,C)
                        scores[i] = np.log(tab[int(x_b[i]), int(y_star)] + 1e-12)
                    else:
                        tab = bundle["cpts"][f"{i}|C,{p}"]  # (ki,C,kp)
                        scores[i] = np.log(tab[int(x_b[i]), int(y_star), int(x_b[p])] + 1e-12)

                # larger magnitude => more "influential" (you can also use abs(scores))
                return np.argsort(-np.abs(scores))

            # Build TAN-based subsets
            _, proba_full = predict_tan(bundle, X_te)
            y_star = np.argmax(proba_full, axis=1)

            k_list = [1,3,5,10]
            subsets_by_k = {k: [] for k in k_list}
            for i in range(len(X_te)):
                rank = tan_fast_rank_features(bundle, X_te[i], y_star[i])
                for k in k_list:
                    subsets_by_k[k].append(rank[:k])

            res = tan_local_fidelity_exact(bundle, X_te, subsets_by_k, proba_full=proba_full, n_instances=200, seed=0)
            print(res)
            
            def explanation_stability_lipschitz_bn(
                bundle,
                X_eval,          # e.g., X_te (continuous features)
                X_ref,           # e.g., X_tr (used for resampling + bounds + std)
                n_instances=50,
                n_neighbors=1,  # number of x' samples per x
                eps=0.02,        # perturbation scale (in "std units", see below)
                n_mc_importance=10,
                seed=0,
                denom_eps=1e-12,
            ):
                """
                Explanation Stability via local Lipschitz constant:
                Lhat(x) = max_{x' in B_eps(x)} ||E(x)-E(x')||_2 / ||x-x'||_2

                Here E(x) is the BN explanation vector = per-feature local importance
                returned by _estimate_importance_by_resampling(...).

                eps meaning:
                x' = x + Normal(0, (eps * std_j)^2) per feature j
                """

                rng = np.random.default_rng(seed)

                X_eval = np.asarray(X_eval, dtype=float)
                X_ref  = np.asarray(X_ref, dtype=float)

                n = min(len(X_eval), int(n_instances))
                idxs = rng.choice(len(X_eval), size=n, replace=False) if n < len(X_eval) else np.arange(len(X_eval))

                # feature-wise scale + bounds (keeps perturbations realistic)
                std = X_ref.std(axis=0, ddof=0)
                std = np.where(std < 1e-12, 1.0, std)  # avoid zero-std features
                fmin = X_ref.min(axis=0)
                fmax = X_ref.max(axis=0)

                Lhats = []
                per_sample = []

                for ii in idxs:
                    x0 = X_eval[ii].copy()

                    # Base prediction + base explanation E(x)
                    _, p0 = _predict_proba_bundle(bundle, x0[None, :])  # uses your helper :contentReference[oaicite:2]{index=2}
                    p0 = p0[0]
                    y0 = int(np.argmax(p0))

                    E0 = _estimate_importance_by_resampling(  # uses your helper :contentReference[oaicite:3]{index=3}
                        bundle=bundle, x0=x0, p_full=p0, y_pred=y0, X_ref=X_ref,
                        n_mc=n_mc_importance, rng=rng
                    )

                    # Sample neighbors and take max ratio
                    best = 0.0
                    for _ in range(int(n_neighbors)):
                        noise = rng.normal(loc=0.0, scale=eps * std, size=x0.shape)
                        x1 = x0 + noise
                        x1 = np.clip(x1, fmin, fmax)

                        # Denominator
                        dx = float(np.linalg.norm(x0 - x1))
                        if dx <= denom_eps:
                            continue

                        # Explanation at x'
                        _, p1 = _predict_proba_bundle(bundle, x1[None, :])
                        p1 = p1[0]
                        y1 = int(np.argmax(p1))

                        E1 = _estimate_importance_by_resampling(
                            bundle=bundle, x0=x1, p_full=p1, y_pred=y1, X_ref=X_ref,
                            n_mc=n_mc_importance, rng=rng
                        )

                        num = float(np.linalg.norm(E0 - E1))
                        ratio = num / (dx + denom_eps)
                        if ratio > best:
                            best = ratio

                    Lhats.append(best)
                    per_sample.append((int(ii), float(best)))

                Lhats = np.asarray(Lhats, dtype=float)

                print("\n=== Explanation Stability (local Lipschitz) for BN explanation ===")
                print(f"n_instances   : {len(Lhats)}")
                print(f"n_neighbors   : {n_neighbors}")
                print(f"eps (std frac): {eps}")
                print(f"n_mc_import.  : {n_mc_importance}")
                print(f"Mean Lhat     : {float(np.mean(Lhats)):.6f}   (lower => more stable)")
                print(f"Median Lhat   : {float(np.median(Lhats)):.6f}")
                print(f"Max Lhat      : {float(np.max(Lhats)):.6f}")
                print("Per-sample Lhat (first 10):")
                for j, (idx, val) in enumerate(per_sample[:10]):
                    print(f"  sample[{idx}] Lhat = {val:.6f}")
                print("===============================================================\n")

                return None

            t = time.time()

            stab = explanation_stability_lipschitz_bn(
    bundle=bundle,
    X_eval=X_te,
    X_ref=X_tr,
    n_instances=min(5, len(X_te)),
    n_neighbors=1,
    eps=0.02,
    n_mc_importance=2,
    seed=0
)
            print(time.time() - t)

    print(acc_all)
    print(sum(acc_all)/len(acc_all))

if __name__=='__main__':
    main()