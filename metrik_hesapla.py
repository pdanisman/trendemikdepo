#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrEndemik - Kamera doğrulama testi: metrik hesaplama betiği
============================================================

Girdi : test_sablonu.xlsx (Fotograf_Kaydi, Sonuc_Kaydi, Tutarlilik) + veri.xlsx (sizin tür listeniz)
Çıktı : sonuclar/ klasörü (ozet.md, metrikler.csv, esik_tarama.csv, hata_adaylari.csv, grafikler)

Kullanım:
    python metrik_hesapla.py --sablon test_sablonu.xlsx --veri.xlsx
    python metrik_hesapla.py --demo            # sentetik verilerle deneme (gerçek sonuç değildir)

Test edilen karar kuralları (hepsi aynı kayıtlı top-5 sonuçlarından çevrimdışı hesaplanır):
    V0 : Sitenin mevcut mantığı. PlantNet'in top-1 adı, listedeki Latince adlarla BİREBİR eşleşirse "endemik".
    V1 : Top-5 + skor eşiği + ad normalizasyonu (yalnızca ilk iki sözcük: cins + tür). Liste = Nuh'un Gemisi türleri.
    V2 : V1 ile aynı, ancak liste = türler + güncel (kabul edilen) adlar + alt taksonların tür adları (eş anlamlı ad uyuşmazlığını çözer).
    (Eski koordinatlı veri.xlsx verilirse V1 = koordinatlı satırlar, V2 = tüm tür düzeyindeki satırlardır.)
Eşik (V1, V2) yalnızca GELISTIRME setinde seçilir (MCC'yi en yükseğe çıkaran değer) ve TEST setinde değiştirilmeden ölçülür.
Kıyas: "hep endemik" ve "hiç endemik değil" kuralları.
"""
import argparse
import math
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

SEED = 42
NAN = float("nan")
DEFAULT_ESIK = 0.30

# ------------------------------------------------------------------ ad işleme
def _s(x):
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()


def exact(x):
    """Sitedeki JS karşılaştırması: toLowerCase().trim()"""
    return _s(x).lower()


def norm(x):
    s = _s(x).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def binom(x):
    t = [w for w in norm(x).split(" ") if w]
    return " ".join(t[:2])


def genus(x):
    t = [w for w in norm(x).split(" ") if w]
    return t[0] if t else ""


def to_float(x):
    s = _s(x).replace(",", ".")
    if s == "":
        return NAN
    try:
        return float(s)
    except ValueError:
        return NAN


# ------------------------------------------------------------------ istatistik
def wilson(k, n, z=1.96):
    if n == 0:
        return (NAN, NAN)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def pct(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def pct_ci(k, n):
    if n == 0:
        return "–"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}–{100 * hi:.1f}] (n={n})"


def f3(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def confusion(y, p):
    y = np.asarray(y, bool)
    p = np.asarray(p, bool)
    return int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum()), int((~y & ~p).sum())  # tp fp fn tn


def metrics_from_conf(tp, fp, fn, tn):
    def div(a, b):
        return a / b if b else NAN

    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    ppv, npv = div(tp, tp + fp), div(tn, tn + fn)
    acc = div(tp + tn, tp + fp + fn + tn)
    bal = (sens + spec) / 2 if not (math.isnan(sens) or math.isnan(spec)) else NAN
    f1 = div(2 * tp, 2 * tp + fp + fn)
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0  # tanımsızsa 0 (yaygın kabul)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, sens=sens, spec=spec, ppv=ppv, npv=npv, acc=acc, bal=bal, f1=f1, mcc=mcc)


def ppv_at_prevalence(sens, spec, prev):
    if any(math.isnan(v) for v in (sens, spec)):
        return NAN
    den = sens * prev + (1 - spec) * (1 - prev)
    return sens * prev / den if den else NAN


def mcc_of(y, p):
    return metrics_from_conf(*confusion(y, p))["mcc"]


def boot_ci(fn, n, B, rng):
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(fn(idx))
    vals = np.array(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (NAN, NAN)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def auc_roc(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return NAN
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pr_points(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    idx = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp, fp = tp[idx], fp[idx]
    prec = tp / (tp + fp)
    rec = tp / max(1, y.sum())
    return rec, prec


def avg_precision(y, s):
    if int(np.asarray(y, bool).sum()) == 0:
        return NAN
    rec, prec = pr_points(y, s)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def roc_points(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    n1, n0 = max(1, int(y.sum())), max(1, int((~y).sum()))
    ths = np.unique(s)[::-1]
    fpr, tpr = [0.0], [0.0]
    for t in ths:
        p = s >= t
        tpr.append((y & p).sum() / n1)
        fpr.append((~y & p).sum() / n0)
    return np.array(fpr), np.array(tpr)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n == 0:
        return NAN, NAN, NAN, NAN
    po = float(np.mean(a == b))
    cats = np.unique(np.r_[a, b])
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    if pe == 1:
        return NAN, po, NAN, NAN
    k = (po - pe) / (1 - pe)
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    return k, po, k - 1.96 * se, k + 1.96 * se


# ------------------------------------------------------------------ dosya okuma
def read_sheet(path, sheet, key):
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
    idx = None
    for i in range(min(10, len(raw))):
        if _s(raw.iat[i, 0]) == key:
            idx = i
            break
    if idx is None:
        raise SystemExit(f"'{sheet}' sayfasında '{key}' başlığı bulunamadı.")
    df = raw.iloc[idx + 1:].copy()
    df.columns = [_s(c) for c in raw.iloc[idx]]
    df = df[df[key].notna()]
    df[key] = df[key].map(_s)
    df = df[(df[key] != "") & (~df[key].str.upper().str.startswith("ORNEK"))]
    return df.reset_index(drop=True)


def load_lists(v):
    ncol, ccol = "Bilimsel Adı (Latince)", "Koordinat (Enlem, Boylam)"
    if ncol not in v.columns:
        raise SystemExit(f"veri dosyasında '{ncol}' sütunu yok.")
    yeni_liste = ccol not in v.columns
    if yeni_liste:                 # yeni veri seti (Nuh'un Gemisi): koordinat yok, listedeki her tür geçerli sayılır
        v = v.copy()
        v[ccol] = "0,0"

    def has_coord(x):
        s = _s(x)
        if s == "" or "bulunamad" in s.lower():
            return False
        p = s.split(",")
        if len(p) != 2:
            return False
        try:
            float(p[0]); float(p[1])
            return True
        except ValueError:
            return False

    v = v.copy()
    v["_c"] = v[ccol].map(has_coord)
    v["_sp"] = v[ncol].map(lambda x: bool(re.search(r"\bsp\.?$", _s(x))))
    L = {
        "exact_coord": {exact(n) for n in v.loc[v["_c"], ncol]},
        "binom_coord": {binom(n) for n in v.loc[v["_c"] & ~v["_sp"], ncol]},
        "binom_all": {binom(n) for n in v.loc[~v["_sp"], ncol]},
        "genus_sp": {genus(n) for n in v.loc[v["_sp"], ncol]},
    }
    if yeni_liste:                 # V2: güncel adlar ve alt taksonların tür adları da listeye girer
        ek = set()
        if "Kabul_Edilen_Ad" in v.columns:
            ek |= {binom(x) for x in v["Kabul_Edilen_Ad"] if _s(x)}
        if "Alt_Taksonlar" in v.columns:
            for x in v["Alt_Taksonlar"]:
                ek |= {binom(y) for y in _s(x).split(";") if y.strip()}
        L["binom_all"] = L["binom_all"] | ek
    info = dict(satir=len(v), koordinatli=int(v["_c"].sum()), cins_duzeyi=int(v["_sp"].sum()),
                tur_koordinatli=len(L["binom_coord"]), tur_tum=len(L["binom_all"]))
    return L, info


# ------------------------------------------------------------------ karar kuralları
def top_list(row):
    out = []
    for i in range(1, 6):
        n, s = _s(row.get(f"top{i}_ad")), to_float(row.get(f"top{i}_skor"))
        if n != "":
            out.append((n, 0.0 if math.isnan(s) else s))
    return out


def decide(version, top, L, esik):
    """(karar: bool, eşleşen ad veya None, skor)"""
    if not top:
        return False, None, 0.0
    if version == "V0":
        n, s = top[0]
        return (exact(n) in L["exact_coord"]), (n if exact(n) in L["exact_coord"] else None), (s if exact(n) in L["exact_coord"] else 0.0)
    lst = L["binom_coord"] if version == "V1" else L["binom_all"]
    best = (None, 0.0)
    for n, s in top:
        if binom(n) in lst and s > best[1]:
            best = (n, s)
    return (best[0] is not None and best[1] >= esik and best[1] > 0), (best[0] if best[1] >= esik else None), best[1]


def list_score(version, top, L):
    """ROC için sürekli skor: listede olan adlar arasındaki en yüksek skor (yoksa 0)."""
    if not top:
        return 0.0
    lst = L["binom_coord"] if version != "V2" else L["binom_all"]
    if version == "V0":
        n, s = top[0]
        return s if exact(n) in L["exact_coord"] else 0.0
    return max([s for n, s in top if binom(n) in lst] or [0.0])


def error_category(version, kind, rec, L, esik):
    """Otomatik hata adayı (öğrenciler onaylar). kind: 'FN' veya 'FP'."""
    top = rec["_top"]
    true_b = binom(rec.get("gercek_tur"))
    top_b = [binom(n) for n, _ in top]
    lst = L["binom_coord"] if version != "V2" else L["binom_all"]
    if kind == "FN":
        if not top:
            return "E7", "PlantNet sonuç döndürmedi"
        if not true_b or true_b not in top_b:
            note = "cins aynı (sinonim/yanlış tür?)" if top and genus(top[0][0]) == genus(true_b) and true_b else ""
            return "E1", note
        j = top_b.index(true_b)
        if true_b not in lst:
            return ("E3", "listede yalnızca 'cins sp.' var") if genus(true_b) in L["genus_sp"] else ("E2", "tür listede yok")
        if version == "V0":
            if j == 0:
                return "E4", "ad biçimi birebir eşleşmedi"
            return "E5", "doğru tür top-1 değil"
        return ("E5", f"skor {top[j][1]:.2f} < eşik {esik:.2f}") if top[j][1] < esik else ("E?", "kontrol edin")
    # FP
    _, matched, _ = decide(version, top, L, esik)
    mb = binom(matched) if matched else ""
    if true_b and mb == true_b:
        return "E6", "tür listede 'endemik' ama sizin etiketiniz 'değil': liste/etiket çelişkisi"
    note = "yakın akraba karışıklığı (aynı cins)" if true_b and mb and genus(matched) == genus(true_b) else "yanlış türe eşleşti"
    return "E1", note


# ------------------------------------------------------------------ analiz
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def run_analysis(photos, results, tut, L, linfo, args):
    rng = np.random.default_rng(SEED)
    out_dir = args.cikti
    os.makedirs(out_dir, exist_ok=True)
    rep, warns, rows_csv = [], [], []

    # ---- doğrulama ve birleştirme
    for c in ("foto_id", "set", "grup", "gercek_endemik"):
        if c not in photos.columns:
            raise SystemExit(f"Fotograf_Kaydi'nda '{c}' sütunu yok.")
    for c in ("gercek_tur", "kaynak_turu", "organ", "tek_tur_mu", "dogrulayici1_etiket", "dogrulayici2_etiket"):
        if c not in photos.columns:
            photos[c] = ""
    photos = photos.copy()
    for c in photos.columns:
        photos[c] = photos[c].map(_s)
    ok = {"EVET": True, "E": True, "1": True, "TRUE": True, "HAYIR": False, "H": False, "0": False, "FALSE": False}
    bad = photos[~photos["gercek_endemik"].str.upper().isin(ok)]
    if len(bad):
        raise SystemExit("gercek_endemik EVET/HAYIR olmalı. Hatalı foto_id: " + ", ".join(bad["foto_id"].head(10)))
    photos["y"] = photos["gercek_endemik"].str.upper().map(ok).astype(bool)
    photos["set"] = photos["set"].str.upper()
    dup = photos["foto_id"].duplicated().sum()
    if dup:
        warns.append(f"{dup} yinelenen foto_id var (ilki kullanıldı).")
        photos = photos.drop_duplicates("foto_id")

    results = results.copy()
    if results["foto_id"].duplicated().any():
        warns.append("Sonuc_Kaydi'nda aynı foto_id birden fazla satırda; ilk satır kullanıldı (tekrarları Tutarlilik sayfasına yazın).")
        results = results.drop_duplicates("foto_id")
    df = photos.merge(results, on="foto_id", how="left", suffixes=("", "_r"))
    df["_top"] = df.apply(lambda r: top_list(r), axis=1)
    has_res = df["foto_id"].isin(results["foto_id"])
    if (~has_res).any():
        warns.append(f"{int((~has_res).sum())} fotoğrafın Sonuc_Kaydi'nda sonucu yok; analiz dışı bırakıldı.")
    df = df[has_res].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("Analiz edilecek kayıt yok.")

    # ---- set seçimi
    sets = df["set"].value_counts().to_dict()
    dev = df[df["set"] == "GELISTIRME"].reset_index(drop=True)
    test = df[df["set"] == "TEST"].reset_index(drop=True)
    if args.set == "TUMU":
        ev, ev_name = df, "TUMU"
    elif args.set in ("TEST", "GELISTIRME"):
        ev, ev_name = (test if args.set == "TEST" else dev), args.set
    else:
        if len(test):
            ev, ev_name = test, "TEST"
        else:
            ev, ev_name = df, "TUMU (TEST seti yok - bağımsız test değil, geliştirme sonucudur)"
            warns.append("TEST setinde fotoğraf yok: sonuçlar bağımsız test olarak raporlanmamalı.")
    ev = ev.reset_index(drop=True)
    if len(ev) == 0:
        raise SystemExit("Seçilen sette kayıt yok.")

    # ---- eşik seçimi (yalnızca geliştirme setinde)
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    esik, esik_src, sweep = {}, {}, {}
    for v in ("V1", "V2"):
        pool = dev if len(dev) else df
        s = np.array([list_score(v, t, L) for t in pool["_top"]])
        y = pool["y"].to_numpy()
        sweep[v] = []
        for t in grid:
            p = (s >= t) & (s > 0)
            m = metrics_from_conf(*confusion(y, p))
            sweep[v].append((t, m["sens"], m["spec"], m["mcc"]))
        usable = len(pool) >= 20 and y.any() and (~y).any()
        if args.esik is not None:
            esik[v], esik_src[v] = args.esik, "elle verildi (--esik)"
        elif usable:
            mccs = [r[3] for r in sweep[v]]
            esik[v] = float(grid[int(np.argmax(mccs))])
            esik_src[v] = ("GELISTIRME setinde MCC'yi maksimize eden değer" if len(dev) else "TÜM veride (geliştirme seti yok!)")
            if not len(dev):
                warns.append("GELISTIRME seti yok: eşik tüm veride seçildi, sonuç iyimser olur.")
        else:
            esik[v], esik_src[v] = DEFAULT_ESIK, "varsayılan (yeterli geliştirme verisi yok)"
            warns.append(f"{v}: eşik ayarı için yeterli veri yok, varsayılan {DEFAULT_ESIK} kullanıldı.")
    pd.DataFrame([(v, *r) for v in sweep for r in sweep[v]], columns=["surum", "esik", "duyarlilik", "ozgulluk", "mcc"]).to_csv(
        os.path.join(out_dir, "esik_tarama.csv"), index=False)

    # ---- kararlar
    versions = ["V0", "V1", "V2"]
    for v in versions:
        res = [decide(v, t, L, esik.get(v, 0)) for t in ev["_top"]]
        ev[f"karar_{v}"] = [r[0] for r in res]
        ev[f"esles_{v}"] = [r[1] for r in res]
        ev[f"skor_{v}"] = [list_score(v, t, L) for t in ev["_top"]]
    ev["karar_HEP_EVET"] = True
    ev["karar_HEP_HAYIR"] = False
    y = ev["y"].to_numpy()

    # ======================= RAPOR
    rep.append("# TrEndemik V0 baseline testi – otomatik özet\n")
    rep.append(f"*Değerlendirme seti:* **{ev_name}** · fotoğraf: {len(ev)} (pozitif {int(y.sum())}, negatif {int((~y).sum())})  ")
    rep.append(f"*Tüm kayıtlar:* {sets}  ")
    rep.append(f"*Veri listesi:* {linfo['satir']} satır, cins düzeyi ('sp.') {linfo['cins_duzeyi']}; "
               f"farklı tür adı: V0/V1 listesi {linfo['tur_koordinatli']}, V2 listesi (güncel/eş anlamlı adlar dahil) {linfo['tur_tum']}  ")
    rep.append(f"*Eşik:* V1 = {esik['V1']:.2f} ({esik_src['V1']}); V2 = {esik['V2']:.2f} ({esik_src['V2']})  ")
    n_multi = int((ev["tek_tur_mu"].str.upper() == "H").sum())
    if n_multi:
        rep.append(f"*Uyarı:* {n_multi} fotoğraf 'tek_tur_mu = H' (karede birden fazla tür) olarak işaretli; analize dahildir.  ")
    rep.append("")

    # ---- etiket uyumu
    both = ev[(ev["dogrulayici1_etiket"] != "") & (ev["dogrulayici2_etiket"] != "")]
    rep.append("## 1. Etiket doğrulama uyumu (Cohen kappa)\n")
    if len(both) >= 5:
        k, po, lo, hi = cohen_kappa(both["dogrulayici1_etiket"].str.upper(), both["dogrulayici2_etiket"].str.upper())
        rep.append(f"n = {len(both)}, gözlenen uyum = {pct(po)}, kappa = {f3(k)} (yaklaşık %95 GA {f3(lo)}–{f3(hi)})\n")
        rows_csv.append(("tumu", ev_name, "-", "kappa", k, lo, hi))
    else:
        rep.append("İki doğrulayıcı etiketi yeterli sayıda girilmemiş (en az 5 fotoğraf gerekir).\n")

    # ---- tanıma düzeyi
    rep.append("## 2. Tanıma düzeyi (PlantNet doğru türü buluyor mu?)\n")
    idm = ev[ev["gercek_tur"] != ""].copy()
    def topk_hits(r, k, level):
        f = binom if level == "tur" else genus
        t = f(r["gercek_tur"])
        return t != "" and t in [f(n) for n, _ in r["_top"][:k]]
    for k in (1, 3, 5):
        idm[f"tur_top{k}"] = idm.apply(lambda r: topk_hits(r, k, "tur"), axis=1)
        idm[f"cins_top{k}"] = idm.apply(lambda r: topk_hits(r, k, "cins"), axis=1)
    idm["sonuc_yok_b"] = idm["_top"].map(len) == 0
    def id_rows(g):
        n = len(g)
        return [n] + [pct_ci(int(g[c].sum()), n) for c in ("tur_top1", "tur_top3", "tur_top5", "cins_top1", "cins_top5")] + [pct(g["sonuc_yok_b"].mean())]
    hdr = ["Grup", "n", "Tür top-1", "Tür top-3", "Tür top-5", "Cins top-1", "Cins top-5", "Sonuç yok"]
    rows = [["TÜMÜ"] + id_rows(idm)]
    for col, label in (("grup", "grup"), ("kaynak_turu", "kaynak"), ("organ", "organ")):
        for val, g in idm.groupby(col):
            if val != "":
                rows.append([f"{label}={val}"] + id_rows(g))
    rep.append(md_table(hdr, rows))
    rep.append(f"\n*Gerçek tür adı girilen fotoğraf:* {len(idm)}/{len(ev)}. Alt grup satırları keşfedicidir (küçük örneklem).\n")
    for k in (1, 3, 5):
        n = len(idm); kk = int(idm[f"tur_top{k}"].sum()); lo, hi = wilson(kk, n)
        rows_csv.append(("tumu", ev_name, "tanima", f"tur_top{k}", kk / n if n else NAN, lo, hi))

    # ---- tavanlar
    pos = ev[ev["y"]]
    rep.append("## 3. İki tavan (pozitif fotoğraflar için)\n")
    rows = []
    pos_id = pos[pos["gercek_tur"] != ""]
    if len(pos_id):
        rec_ceiling = int(pos_id.apply(lambda r: topk_hits(r, 5, "tur"), axis=1).sum())
        rows.append(["Tanıma tavanı: gerçek tür PlantNet top-5'te", pct_ci(rec_ceiling, len(pos_id))])
        for v, nm in (("V0/V1", "binom_coord"), ("V2", "binom_all")):
            c = int(pos_id["gercek_tur"].map(lambda x: binom(x) in L[nm]).sum())
            rows.append([f"Liste tavanı ({v}): gerçek tür listede", pct_ci(c, len(pos_id))])
    rep.append(md_table(["Tavan", "Oran"], rows) if rows else "Pozitif fotoğraflarda gercek_tur girilmemiş.")
    rep.append("\n*Liste tavanı, listeyi büyüttükçe artması beklenen bir tasarım özelliğidir; buluş olarak değil, kapsam ölçüsü olarak raporlayın.*\n")

    # ---- karar düzeyi
    rep.append("## 4. Karar düzeyi (\"endemik mi?\")\n")
    names = {"V0": "V0 (mevcut sistem)", "V1": "V1 (top-5+eşik+normalizasyon)", "V2": "V2 (V1 + güncel/eş anlamlı adlar)",
             "HEP_EVET": "Kıyas: hep 'endemik'", "HEP_HAYIR": "Kıyas: hiç 'endemik değil'"}
    mm, cm = {}, {}
    for v in ["V0", "V1", "V2", "HEP_EVET", "HEP_HAYIR"]:
        p = ev[f"karar_{v}"].to_numpy()
        tp, fp, fn, tn = confusion(y, p)
        mm[v] = metrics_from_conf(tp, fp, fn, tn)
        cm[v] = (tp, fp, fn, tn)
    rows = []
    for v, m in mm.items():
        tp, fp, fn, tn = cm[v]
        rows.append([names[v], tp, fp, fn, tn])
    rep.append("**Karışıklık matrisi** (pozitif = endemik)\n")
    rep.append(md_table(["Sürüm", "TP", "FP", "FN", "TN"], rows))
    rep.append("")
    rows = []
    for v, m in mm.items():
        tp, fp, fn, tn = cm[v]
        mcc_ci = boot_ci(lambda idx, p=ev[f"karar_{v}"].to_numpy(): mcc_of(y[idx], p[idx]), len(ev), args.bootstrap, rng) if v in ("V0", "V1", "V2") else (NAN, NAN)
        rows.append([names[v], pct_ci(tp, tp + fn), pct_ci(tn, tn + fp), pct_ci(tp, tp + fp), pct_ci(tn, tn + fn),
                     pct_ci(tp + tn, len(ev)), pct(m["bal"]), f3(m["f1"]),
                     f"{f3(m['mcc'])}" + (f" [{f3(mcc_ci[0])}; {f3(mcc_ci[1])}]" if v in ("V0", "V1", "V2") else "")])
        for key, k_, n_ in (("duyarlilik", tp, tp + fn), ("ozgulluk", tn, tn + fp), ("kesinlik_ppv", tp, tp + fp), ("npv", tn, tn + fn), ("dogruluk", tp + tn, len(ev))):
            lo, hi = wilson(k_, n_)
            rows_csv.append((v, ev_name, "karar", key, k_ / n_ if n_ else NAN, lo, hi))
        rows_csv.append((v, ev_name, "karar", "dengeli_dogruluk", m["bal"], NAN, NAN))
        rows_csv.append((v, ev_name, "karar", "f1", m["f1"], NAN, NAN))
        rows_csv.append((v, ev_name, "karar", "mcc", m["mcc"], mcc_ci[0], mcc_ci[1]))
    rep.append("**Metrikler** (Wilson %95 GA; MCC için bootstrap GA)\n")
    rep.append(md_table(["Sürüm", "Duyarlılık", "Özgüllük", "Kesinlik (PPV)", "NPV", "Doğruluk", "Dengeli doğr.", "F1", "MCC"], rows))
    rep.append("\n*MCC tanımsız olduğunda (ör. 'hep endemik' kıyası) 0 alınır.*\n")

    rep.append("**Gerçekçi yaygınlıkta kesinlik (PPV)** – dengeli testte PPV şişkindir; sahadaki bitkilerin yalnızca %5 veya %10'u endemik ise:\n")
    rows = [[names[v], pct(ppv_at_prevalence(mm[v]["sens"], mm[v]["spec"], 0.05)), pct(ppv_at_prevalence(mm[v]["sens"], mm[v]["spec"], 0.10))] for v in ("V0", "V1", "V2")]
    rep.append(md_table(["Sürüm", "PPV (yaygınlık %5)", "PPV (yaygınlık %10)"], rows))
    rep.append("")

    # ---- alt gruplar
    rep.append("**Gruplara göre doğru karar oranı** (pozitif gruplarda = duyarlılık, negatif gruplarda = özgüllük; keşfedici)\n")
    rows = []
    for g, sub in ev.groupby("grup"):
        row = [g, len(sub)]
        for v in ("V0", "V1", "V2"):
            corr = int((sub[f"karar_{v}"] == sub["y"]).sum())
            row.append(pct_ci(corr, len(sub)))
        rows.append(row)
    rep.append(md_table(["Grup", "n", "V0", "V1", "V2"], rows))
    rep.append("")

    # ---- doğru ama yanlış nedenle
    rep.append("**'Doğru ama yanlış nedenle'** – pozitif fotoğrafta karar doğru (TP) ama PlantNet top-1 gerçek tür değil:\n")
    rows = []
    for v in ("V0", "V1", "V2"):
        tpset = ev[(ev[f"karar_{v}"]) & ev["y"] & (ev["gercek_tur"] != "")]
        w = int(tpset.apply(lambda r: binom(r["_top"][0][0]) != binom(r["gercek_tur"]) if r["_top"] else True, axis=1).sum()) if len(tpset) else 0
        rows.append([names[v], len(tpset), w, pct(w / len(tpset)) if len(tpset) else "–"])
    rep.append(md_table(["Sürüm", "TP (tür adı girilen)", "Top-1 yanlış", "Oran"], rows))
    rep.append("")

    # ---- ROC / PR
    rep.append("## 5. Eşik bağımsız ölçüler (V1, V2)\n")
    rows = []
    curves = {}
    for v in ("V1", "V2"):
        s = ev[f"skor_{v}"].to_numpy()
        a = auc_roc(y, s)
        ap = avg_precision(y, s)
        a_ci = boot_ci(lambda idx, s=s: auc_roc(y[idx], s[idx]), len(ev), args.bootstrap, rng)
        rows.append([names[v], f3(a), f"[{f3(a_ci[0])}; {f3(a_ci[1])}]", f3(ap), pct(y.mean())])
        curves[v] = (roc_points(y, s), pr_points(y, s), a, ap)
        rows_csv.append((v, ev_name, "esik_bagimsiz", "auc_roc", a, a_ci[0], a_ci[1]))
        rows_csv.append((v, ev_name, "esik_bagimsiz", "ortalama_kesinlik_ap", ap, NAN, NAN))
    rep.append(md_table(["Sürüm", "ROC AUC", "AUC bootstrap GA", "PR-AUC (AP)", "Pozitif oranı (AP taban çizgisi)"], rows))
    rep.append("\n*Dengesiz durumda PR eğrisi daha bilgilendiricidir. Eşik seçimi tablosu: esik_tarama.csv.*\n")

    # ---- kalibrasyon
    rep.append("## 6. Güven skoru kalibrasyonu (PlantNet top-1 skoru)\n")
    cal = idm[idm["_top"].map(len) > 0].copy()
    cal["s"] = cal["_top"].map(lambda t: t[0][1])
    cal["ok"] = cal["tur_top1"].astype(float)
    cal_rows = []
    if len(cal) >= 10:
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
        cal["bin"] = np.digitize(cal["s"], bins[1:-1])
        ece = 0.0
        for b in range(5):
            g = cal[cal["bin"] == b]
            if len(g):
                acc_b, conf_b = g["ok"].mean(), g["s"].mean()
                ece += len(g) / len(cal) * abs(acc_b - conf_b)
                lo, hi = wilson(int(g["ok"].sum()), len(g))
                cal_rows.append([f"{bins[b]:.1f}–{min(bins[b+1],1.0):.1f}", len(g), pct(conf_b), pct_ci(int(g["ok"].sum()), len(g))])
        brier = float(((cal["s"] - cal["ok"]) ** 2).mean())
        rep.append(md_table(["Skor aralığı", "n", "Ortalama skor", "Gerçek top-1 tür doğruluğu"], cal_rows))
        rep.append(f"\nBrier skoru = {f3(brier)}, ECE = {f3(ece)}. Skor kalibre bir olasılık değilse, sitedeki 'Eşleşme Oranı %' ifadesi yanıltıcı olabilir.\n")
        rows_csv.append(("PlantNet", ev_name, "kalibrasyon", "brier", brier, NAN, NAN))
        rows_csv.append(("PlantNet", ev_name, "kalibrasyon", "ece", ece, NAN, NAN))
        pd.DataFrame(cal_rows, columns=["aralik", "n", "ort_skor", "dogruluk"]).to_csv(os.path.join(out_dir, "kalibrasyon.csv"), index=False)
    else:
        rep.append("Kalibrasyon için en az 10 fotoğraf (gercek_tur girilmiş) gerekir.\n")

    # ---- sürüm karşılaştırma
    rep.append("## 7. Sürüm karşılaştırması (aynı fotoğraflar, eşleştirilmiş)\n")
    rows = []
    for a, b in (("V0", "V1"), ("V1", "V2"), ("V0", "V2")):
        ca = (ev[f"karar_{a}"] == ev["y"]).to_numpy()
        cb = (ev[f"karar_{b}"] == ev["y"]).to_numpy()
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p = mcnemar_exact(n10, n01)
        pa, pb = ev[f"karar_{a}"].to_numpy(), ev[f"karar_{b}"].to_numpy()
        d = lambda idx: mcc_of(y[idx], pb[idx]) - mcc_of(y[idx], pa[idx])
        dlo, dhi = boot_ci(d, len(ev), args.bootstrap, rng)
        d0 = mcc_of(y, pb) - mcc_of(y, pa)
        rows.append([f"{a} → {b}", f"{a} doğru, {b} yanlış: {n10}", f"{a} yanlış, {b} doğru: {n01}", f"{p:.4f}", f"{d0:+.3f} [{dlo:+.3f}; {dhi:+.3f}]"])
        rows_csv.append((f"{a}->{b}", ev_name, "karsilastirma", "mcnemar_p", p, NAN, NAN))
        rows_csv.append((f"{a}->{b}", ev_name, "karsilastirma", "mcc_farki", d0, dlo, dhi))
    rep.append(md_table(["Karşılaştırma", "Uyumsuz çift 1", "Uyumsuz çift 2", "McNemar tam p", "ΔMCC [bootstrap %95 GA]"], rows))
    rep.append("\n*Çoklu karşılaştırma yapıldığını raporda belirtin. V1/V2 eşikleri geliştirme setinde seçildiği için TEST setinde adil karşılaştırma yapılır.*\n")

    # ---- tutarlılık
    rep.append("## 8. Tutarlılık (aynı fotoğraf tekrar gönderildiğinde)\n")
    if tut is not None and len(tut):
        t = tut.copy()
        t["skor"] = t["top1_skor"].map(to_float)
        t["ad"] = t["top1_ad"].map(_s)
        base = results[["foto_id", "top1_ad", "top1_skor"]].copy()
        base["skor"] = base["top1_skor"].map(to_float)
        base["ad"] = base["top1_ad"].map(_s)
        allr = pd.concat([base[["foto_id", "ad", "skor"]], t[["foto_id", "ad", "skor"]]])
        cnt = allr.groupby("foto_id").size()
        multi = cnt[cnt >= 2].index
        if len(multi):
            g = allr[allr["foto_id"].isin(multi)].groupby("foto_id")
            same = int((g["ad"].nunique() == 1).sum())
            spread = (g["skor"].max() - g["skor"].min())
            rep.append(f"{len(multi)} fotoğraf ≥2 kez gönderildi: top-1 ad hep aynı = {pct_ci(same, len(multi))}; skor aralığı (maks−min) ortalaması = {f3(float(spread.mean()))}, en büyüğü = {f3(float(spread.max()))}.\n")
            rows_csv.append(("PlantNet", ev_name, "tutarlilik", "ayni_top1_orani", same / len(multi), *wilson(same, len(multi))))
        else:
            rep.append("Tekrarı olan fotoğraf bulunamadı.\n")
    else:
        rep.append("Tutarlilik sayfası boş.\n")

    # ---- hata adayları
    rep.append("## 9. Otomatik hata adayları (öğrenciler onaylayıp Hata_Analizi sayfasına yazar)\n")
    errs = []
    for v in ("V0", "V1", "V2"):
        for _, r in ev.iterrows():
            kind = None
            if r["y"] and not r[f"karar_{v}"]:
                kind = "FN"
            elif (not r["y"]) and r[f"karar_{v}"]:
                kind = "FP"
            if kind:
                cat, note = error_category(v, kind, r, L, esik.get(v, 0))
                t1 = r["_top"][0] if r["_top"] else ("", NAN)
                errs.append(dict(foto_id=r["foto_id"], surum=v, set=r["set"], grup=r["grup"], hata_turu=kind, gercek_tur=r["gercek_tur"],
                                 top1_ad=t1[0], top1_skor=t1[1], eslesen_ad=r[f"esles_{v}"], otomatik_kategori=cat, not_=note))
    E = pd.DataFrame(errs)
    if len(E):
        E.rename(columns={"not_": "not"}).to_csv(os.path.join(out_dir, "hata_adaylari.csv"), index=False, encoding="utf-8-sig")
        piv = E.groupby(["surum", "otomatik_kategori"]).size().unstack(fill_value=0)
        rep.append(md_table(["Sürüm"] + list(piv.columns), [[v] + [int(x) for x in piv.loc[v]] if v in piv.index else [v] for v in ("V0", "V1", "V2")]))
        rep.append("\n*E1 yanlış tür · E2 tür doğru ama listede yok · E3 listede yalnızca cins düzeyi · E4 ad biçimi · E5 eşik/sıra · E6 etiket/liste çelişkisi · E7 sonuç yok/kalite. E4 ve E7'nin bir kısmı yalnızca elle belirlenebilir.*\n")
    else:
        rep.append("Hatalı karar yok.\n")

    # ---- ek dosyalar
    pd.DataFrame(rows_csv, columns=["surum", "set", "duzey", "metrik", "deger", "ci_alt", "ci_ust"]).to_csv(os.path.join(out_dir, "metrikler.csv"), index=False, encoding="utf-8-sig")
    keep = ["foto_id", "set", "grup", "gercek_tur", "gercek_endemik", "top1_ad", "top1_skor"] + [c for v in versions for c in (f"karar_{v}", f"esles_{v}")]
    ev[[c for c in keep if c in ev.columns]].to_csv(os.path.join(out_dir, "birlesik_veri.csv"), index=False, encoding="utf-8-sig")

    if warns:
        rep.append("## Uyarılar\n")
        rep += [f"- {w}" for w in warns]
        rep.append("")
    text = "\n".join(rep)
    with open(os.path.join(out_dir, "ozet.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

    try:
        make_plots(out_dir, curves, sweep, esik, mm, cal if len(cal) >= 10 else None, E, dev_n=len(dev) if len(dev) else len(df))
    except Exception as e:  # grafikler isteğe bağlı
        print(f"\n[grafikler çizilemedi: {e}]")
    print(f"\nSonuçlar '{out_dir}' klasörüne yazıldı.")
    return text


# ------------------------------------------------------------------ grafikler
def make_plots(out_dir, curves, sweep, esik, mm, cal, E, dev_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # doğrulanmış kategorik sıra: mavi, turuncu, aqua

    def style(ax):
        ax.set_facecolor(SURF)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)
        ax.title.set_color(INK)

    def fig_(w, h, n=1):
        f, axs = plt.subplots(1, n, figsize=(w, h), facecolor=SURF)
        return f, (axs if n > 1 else [axs])

    # 1) ROC + PR
    f, (a1, a2) = fig_(10, 4.3, 2)
    for v, col in (("V1", C1), ("V2", C2)):
        (fpr, tpr), (rec, prec), auc, ap = curves[v]
        a1.plot(fpr, tpr, color=col, lw=2, label=f"{v}  AUC={auc:.2f}")
        a2.plot(rec, prec, color=col, lw=2, label=f"{v}  AP={ap:.2f}")
    m0 = mm["V0"]
    a1.plot([1 - m0["spec"]], [m0["sens"]], "o", color=C3, ms=9, mec=SURF, mew=2, label="V0 (tek nokta)")
    a2.plot([m0["sens"]], [m0["ppv"]], "o", color=C3, ms=9, mec=SURF, mew=2, label="V0 (tek nokta)")
    a1.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--")
    a1.set(xlabel="Yanlış pozitif oranı (1 − özgüllük)", ylabel="Duyarlılık", title="ROC", xlim=(0, 1), ylim=(0, 1.02))
    a2.set(xlabel="Duyarlılık (recall)", ylabel="Kesinlik (precision)", title="Precision–Recall", xlim=(0, 1), ylim=(0, 1.02))
    for a in (a1, a2):
        style(a); a.legend(frameon=False, fontsize=9, loc="lower right" if a is a1 else "lower left")
    f.tight_layout(); f.savefig(os.path.join(out_dir, "roc_pr.png"), dpi=160, facecolor=SURF); plt.close(f)

    # 2) Eşik tarama
    f, axs = fig_(10, 4.3, 2)
    for ax, v in zip(axs, ("V1", "V2")):
        t = [r[0] for r in sweep[v]]
        for idx, col, lab in ((1, C1, "Duyarlılık"), (2, C2, "Özgüllük"), (3, C3, "MCC")):
            ax.plot(t, [r[idx] for r in sweep[v]], color=col, lw=2, label=lab)
        ax.axvline(esik[v], color=INK2, lw=1.2, ls="--")
        ax.text(esik[v] + 0.01, 0.04, f"seçilen eşik {esik[v]:.2f}", color=INK2, fontsize=9)
        ax.set(xlabel="Skor eşiği", title=f"{v} – geliştirme setinde eşik taraması (n={dev_n})", xlim=(0, 1), ylim=(-0.05, 1.02))
        style(ax); ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 0.1))
    f.tight_layout(); f.savefig(os.path.join(out_dir, "esik_tarama.png"), dpi=160, facecolor=SURF); plt.close(f)

    # 3) Kalibrasyon
    if cal is not None:
        f, (ax,) = fig_(5.2, 4.6)
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
        xs, ys, ns = [], [], []
        for b in range(5):
            g = cal[cal["bin"] == b]
            if len(g):
                xs.append(g["s"].mean()); ys.append(g["ok"].mean()); ns.append(len(g))
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--", label="ideal")
        ax.plot(xs, ys, color=C1, lw=2, label="PlantNet top-1")
        ax.scatter(xs, ys, s=[30 + 6 * n for n in ns], color=C1, edgecolor=SURF, linewidth=2, zorder=3)
        for x_, y_, n_ in zip(xs, ys, ns):
            ax.text(x_, y_ + 0.04, f"n={n_}", ha="center", fontsize=8, color=INK2)
        ax.set(xlabel="Ortalama skor", ylabel="Gerçek top-1 tür doğruluğu", title="Güven skoru kalibrasyonu", xlim=(0, 1), ylim=(0, 1.1))
        style(ax); ax.legend(frameon=False, fontsize=9, loc="upper left")
        f.tight_layout(); f.savefig(os.path.join(out_dir, "kalibrasyon.png"), dpi=160, facecolor=SURF); plt.close(f)

    # 4) Hata dağılımı
    if E is not None and len(E):
        cats = sorted(E["otomatik_kategori"].unique())
        f, (ax,) = fig_(8, 4.3)
        w = 0.26
        x = np.arange(len(cats))
        for i, (v, col) in enumerate((("V0", C1), ("V1", C2), ("V2", C3))):
            vals = [int(((E["surum"] == v) & (E["otomatik_kategori"] == c)).sum()) for c in cats]
            ax.bar(x + (i - 1) * (w + 0.02), vals, w, color=col, label=v, edgecolor=SURF, linewidth=2)
        ax.set_xticks(x); ax.set_xticklabels(cats)
        ax.set(xlabel="Hata kategorisi (otomatik aday)", ylabel="Hatalı karar sayısı", title="Hata dağılımı")
        style(ax); ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=9)
        f.tight_layout(); f.savefig(os.path.join(out_dir, "hata_dagilimi.png"), dpi=160, facecolor=SURF); plt.close(f)


# ------------------------------------------------------------------ demo
def demo_data(seed=7):
    rng = np.random.default_rng(seed)
    veri_rows = []
    for i in range(60):
        veri_rows.append((f"Endemia{i // 3} specia{i}", "39.0,35.0" if i < 40 else "Koordinat Bulunamadı"))
    for g in range(5):
        veri_rows.append((f"Genusx{g} sp.", "38.0,34.0"))
    veri = pd.DataFrame(veri_rows, columns=["Bilimsel Adı (Latince)", "Koordinat (Enlem, Boylam)"])
    P = []  # (grup, tür, endemik)
    for i in range(45): P.append(("P1", f"Endemia{i // 3} specia{i}", "EVET"))
    for i in range(45, 55): P.append(("P1", f"Endemia{i // 3} specia{i}", "EVET"))
    for i in range(45): P.append(("P2", f"Extraendemia{i} turcica", "EVET"))
    for i in range(40): P.append(("N1", f"Common{i} vulgaris", "HAYIR"))
    for i in range(30): P.append(("N2", f"Endemia{i} otherspecies", "HAYIR"))
    for i in range(30): P.append(("N3", f"Orna{i} hortensis", "HAYIR"))
    rng.shuffle(P)
    rows_f, rows_r = [], []
    pid = {"P1": 0.80, "P2": 0.55, "N1": 0.85, "N2": 0.70, "N3": 0.80}
    for k, (g, sp, en) in enumerate(P):
        fid = f"F{k + 1:03d}"
        st = "GELISTIRME" if rng.random() < 0.5 else "TEST"
        rows_f.append(dict(foto_id=fid, dosya_adi=fid + ".jpg", set=st, grup=g, gercek_tur=sp, gercek_endemik=en,
                           dogrulayici1_etiket=en, dogrulayici2_etiket=en if rng.random() > 0.05 else ("HAYIR" if en == "EVET" else "EVET"),
                           kaynak_turu="OGRENCI_CEKIMI" if rng.random() < .6 else "ACIK_GOZLEM",
                           organ=rng.choice(["cicek", "yaprak", "bitki"]), tek_tur_mu="E"))
        ok = rng.random() < pid[g]
        cands = []
        if ok:
            cands.append((sp, float(rng.beta(6, 2))))
        else:
            if g == "N2" and rng.random() < 0.5:
                j = int(sp.split()[0].replace("Endemia", ""))
                cands.append((f"Endemia{j} specia{3 * j}", float(rng.beta(3, 3))))
            else:
                cands.append((f"Random{int(rng.integers(0, 999))} wrongus", float(rng.beta(3, 4))))
            if rng.random() < 0.5:
                cands.append((sp, float(rng.beta(2, 5))))
        while len(cands) < 5:
            cands.append((f"Filler{int(rng.integers(0, 9999))} minor", float(rng.beta(1, 12))))
        cands = sorted(cands, key=lambda x: -x[1])[:5]
        r = dict(foto_id=fid, sorgu_tarihi="2026-10-16", sistem_surumu="V0", sonuc_yok="H")
        for i, (n, s) in enumerate(cands, 1):
            r[f"top{i}_ad"], r[f"top{i}_skor"] = n, round(s, 4)
        rows_r.append(r)
    tut_rows = []
    for fid in [f"F{k:03d}" for k in range(1, 21)]:
        base = [r for r in rows_r if r["foto_id"] == fid][0]
        for t in (1, 2):
            tut_rows.append(dict(foto_id=fid, tekrar_no=t, sorgu_tarihi="2026-10-17", top1_ad=base["top1_ad"], top1_skor=base["top1_skor"]))
    return pd.DataFrame(rows_f), pd.DataFrame(rows_r), pd.DataFrame(tut_rows), veri


def main():
    ap = argparse.ArgumentParser(description="TrEndemik kamera testi metrik hesaplama")
    ap.add_argument("--sablon", default="test_sablonu.xlsx")
    ap.add_argument("--veri", default="veri.xlsx")
    ap.add_argument("--cikti", default="sonuclar")
    ap.add_argument("--esik", type=float, default=None, help="V1/V2 için elle eşik (verilmezse GELISTIRME setinde seçilir)")
    ap.add_argument("--set", default="OTO", choices=["OTO", "TEST", "GELISTIRME", "TUMU"], help="değerlendirme seti (OTO: TEST varsa TEST)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--demo", action="store_true", help="sentetik verilerle deneme çalıştırması")
    a = ap.parse_args()

    if a.demo:
        photos, results, tut, veri = demo_data()
        if a.cikti == "sonuclar":
            a.cikti = os.path.join(tempfile.gettempdir(), "trendemik_demo_sonuclar")
        print("*** DEMO: sentetik veri, gerçek sonuç DEĞİLDİR ***\n")
    else:
        for p in (a.sablon, a.veri):
            if not os.path.exists(p):
                raise SystemExit(f"Dosya bulunamadı: {p}")
        photos = read_sheet(a.sablon, "Fotograf_Kaydi", "foto_id")
        results = read_sheet(a.sablon, "Sonuc_Kaydi", "foto_id")
        tut = read_sheet(a.sablon, "Tutarlilik", "foto_id")
        veri = pd.read_excel(a.veri)
    L, linfo = load_lists(veri)
    run_analysis(photos, results, tut, L, linfo, a)


if __name__ == "__main__":
    main()
