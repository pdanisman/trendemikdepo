#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrEndemik - PlantNet sorgu betiği (test fotoğraflarını sırayla gönderir, top-5 sonucu şablona yazar)
=====================================================================================================

Kullanım:
    export PLANTNET_API_KEY="yeni-anahtariniz"        # Windows PowerShell: $env:PLANTNET_API_KEY="..."
    python test_kos.py --sablon test_sablonu.xlsx --klasor fotograflar --surum V0

    # tutarlılık testi: sonucu olan 20 rastgele fotoğrafı 2 kez daha gönderir (ilk gönderimle toplam 3)
    python test_kos.py --sablon test_sablonu.xlsx --klasor fotograflar --tekrar 2 --tekrar-adet 20

    # göndermeden sırayı ve dosyaları kontrol etmek için
    python test_kos.py --sablon test_sablonu.xlsx --klasor fotograflar --kuru

Notlar
 * API anahtarı koda yazılmaz, yalnızca PLANTNET_API_KEY ortam değişkeninden okunur.
 * Fotoğraflar sabit bir tohumla RASTGELE sırada gönderilir (--tohum). Aynı tohumla sıra yeniden üretilebilir.
 * Sonucu zaten kayıtlı fotoğraflar atlanır; günlük kota dolarsa betik durur, ertesi gün aynı komutla devam eder.
 * Her sorgu tarihi, sistem sürümü ve API'nin döndürdüğü sürüm/kota bilgisi 'not' sütununa yazılır.
 * Fotoğraf dosyası adı = Fotograf_Kaydi'ndaki dosya_adi (yoksa foto_id + .jpg/.jpeg/.png denenir).
"""
import argparse
import mimetypes
import os
import random
import sys
import time
from datetime import datetime

import requests
from openpyxl import load_workbook

# ---- PlantNet API anahtarınız (kendi bilgisayarınızda çalıştırdığınız için doğrudan buraya yazıldı) ----
# Ekran görüntüsünden okundu; "I" (büyük i) ile "l" (küçük L) karışmış olabilir.
# "API anahtarı reddedildi" hatası alırsanız PlantNet sayfasından kopyalayıp aşağıya yapıştırın.
API_ANAHTARI = "2b10QZMSWK1IiA1Y3zElp8Du"

URL = "https://my-api.plantnet.org/v2/identify/all"   # sitedeki (index.html) çağrıyla aynı uç nokta
SONUC_BASLIK = ["foto_id", "sorgu_tarihi", "sistem_surumu", "top1_ad", "top1_skor", "top2_ad", "top2_skor", "top3_ad", "top3_skor",
                "top4_ad", "top4_skor", "top5_ad", "top5_skor", "sonuc_yok", "sure_sn", "not"]


class KotaDoldu(Exception):
    pass


def find_header(ws, key="foto_id"):
    for r in range(1, 8):
        if str(ws.cell(r, 1).value or "").strip() == key:
            return r, {str(c.value).strip(): i for i, c in enumerate(ws[r], 1) if c.value is not None}
    raise SystemExit(f"'{ws.title}' sayfasında '{key}' başlığı bulunamadı.")


def read_rows(ws):
    hr, cols = find_header(ws)
    rows = []
    for r in range(hr + 1, ws.max_row + 1):
        fid = ws.cell(r, 1).value
        if fid is None or str(fid).strip() == "" or str(fid).strip().upper().startswith("ORNEK"):
            continue
        rows.append({c: ws.cell(r, i).value for c, i in cols.items()})
    return rows


def append_row(ws, values: dict):
    hr, cols = find_header(ws)
    r = hr + 1
    while ws.cell(r, 1).value not in (None, ""):
        r += 1
    for c, v in values.items():
        if c in cols:
            ws.cell(r, cols[c], v)


def find_file(klasor, dosya_adi, foto_id):
    cands = [dosya_adi] if dosya_adi else []
    cands += [foto_id + e for e in (".jpg", ".jpeg", ".JPG", ".png", ".PNG")]
    for c in cands:
        if c:
            p = os.path.join(klasor, str(c))
            if os.path.isfile(p):
                return p
    return None


def sorgula(path, key, zaman_asimi=60):
    """(top5 listesi, sonuc_yok, sure, not). Kota/anahtar hatalarında istisna fırlatır."""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    t0 = time.time()
    with open(path, "rb") as fh:
        r = requests.post(URL, params={"api-key": key}, files=[("images", (os.path.basename(path), fh, mime))],
                          data={"organs": "auto"}, timeout=zaman_asimi)
    sure = round(time.time() - t0, 2)
    if r.status_code == 429:
        raise KotaDoldu("PlantNet günlük/saatlik kota sınırı (HTTP 429).")
    if r.status_code in (401, 403):
        raise SystemExit(f"API anahtarı reddedildi (HTTP {r.status_code}). Anahtarı kontrol edin.")
    if r.status_code == 404:  # PlantNet eşleşme bulamadığında 404 döndürebilir
        return [], True, sure, "HTTP 404 (sonuç yok)"
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    res = sorted(data.get("results", []), key=lambda x: -x.get("score", 0))[:5]
    top = [(x["species"]["scientificNameWithoutAuthor"], round(float(x["score"]), 5)) for x in res]
    note = f"API sürümü={data.get('version')}; kalan kota={data.get('remainingIdentificationRequests')}"
    return top, (len(top) == 0), sure, note


def top_to_cols(top):
    d = {}
    for i in range(1, 6):
        n, s = top[i - 1] if i <= len(top) else ("", "")
        d[f"top{i}_ad"], d[f"top{i}_skor"] = (n or None), (s if s != "" else None)
    return d


def main():
    ap = argparse.ArgumentParser(description="TrEndemik PlantNet test sorgu betiği")
    ap.add_argument("--sablon", default="test_sablonu.xlsx")
    ap.add_argument("--klasor", default="fotograflar")
    ap.add_argument("--surum", default="V0")
    ap.add_argument("--tohum", type=int, default=2027, help="rastgele sıra için tohum")
    ap.add_argument("--gunluk-limit", type=int, default=100, help="bu çalıştırmada en fazla kaç sorgu (PlantNet güncel kotanızı kontrol edin)")
    ap.add_argument("--bekle", type=float, default=1.0, help="sorgular arası bekleme (sn)")
    ap.add_argument("--tekrar", type=int, default=0, help="tutarlılık: seçilen fotoğrafları kaç KEZ DAHA göndersin")
    ap.add_argument("--tekrar-adet", type=int, default=20, help="tutarlılık testine alınacak fotoğraf sayısı")
    ap.add_argument("--kuru", action="store_true", help="göndermeden planı yazdır")
    a = ap.parse_args()

    key = (os.environ.get("PLANTNET_API_KEY", "").strip() or API_ANAHTARI).strip()
    if not key and not a.kuru:
        raise SystemExit("API anahtarı yok. Dosyanın başındaki API_ANAHTARI satırına yazın.")
    if not os.path.isfile(a.sablon):
        raise SystemExit(f"Şablon bulunamadı: {a.sablon}")
    if not os.path.isdir(a.klasor):
        raise SystemExit(f"Fotoğraf klasörü bulunamadı: {a.klasor}")

    wb = load_workbook(a.sablon)
    foto = read_rows(wb["Fotograf_Kaydi"])
    sonuc_ws, tut_ws = wb["Sonuc_Kaydi"], wb["Tutarlilik"]
    done = {str(r["foto_id"]).strip() for r in read_rows(sonuc_ws)}
    rng = random.Random(a.tohum)

    if a.tekrar > 0:
        pool = [r for r in foto if str(r["foto_id"]).strip() in done]
        rng.shuffle(pool)
        secilen = pool[:a.tekrar_adet]
        plan = [(r, k) for r in secilen for k in range(1, a.tekrar + 1)]
        rng.shuffle(plan)
        hedef_ws, mod = tut_ws, "TEKRAR"
    else:
        pend = [r for r in foto if str(r["foto_id"]).strip() not in done]
        rng.shuffle(pend)
        plan = [(r, 0) for r in pend]
        hedef_ws, mod = sonuc_ws, "ANA"

    yok = []
    kayit = []
    for r, k in plan:
        p = find_file(a.klasor, r.get("dosya_adi"), str(r["foto_id"]).strip())
        (kayit if p else yok).append((r, k, p))
    print(f"Mod: {mod} | kayıtlı fotoğraf: {len(foto)} | planlanan sorgu: {len(plan)} | dosyası bulunan: {len(kayit)} | dosyası YOK: {len(yok)}")
    for r, _, _ in yok[:10]:
        print("  dosya bulunamadı:", r["foto_id"], r.get("dosya_adi"))
    if a.kuru:
        print("İlk 10 sorgu sırası:", [str(r["foto_id"]) + (f"#{k}" if k else "") for r, k, _ in kayit[:10]])
        return

    n_ok = 0
    try:
        for r, k, p in kayit[:a.gunluk_limit]:
            fid = str(r["foto_id"]).strip()
            zaman = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                top, yok_mu, sure, note = sorgula(p, key)
            except KotaDoldu as e:
                print(f"\nDURDU: {e} Aynı komutla yarın devam edin; kayıtlı sonuçlar korunur.")
                break
            except (requests.RequestException, RuntimeError) as e:
                print(f"  ! {fid}: {e} (atlandı, sonra yeniden denenebilir)")
                continue
            if mod == "ANA":
                append_row(hedef_ws, dict(foto_id=fid, sorgu_tarihi=zaman, sistem_surumu=a.surum, **top_to_cols(top),
                                          sonuc_yok="E" if yok_mu else "H", sure_sn=sure, **{"not": note}))
            else:
                t1 = top[0] if top else (None, None)
                append_row(hedef_ws, dict(foto_id=fid, tekrar_no=k, sorgu_tarihi=zaman, top1_ad=t1[0], top1_skor=t1[1]))
            n_ok += 1
            print(f"  {fid}{'#' + str(k) if k else ''}: {top[0][0] + ' (' + format(top[0][1], '.2f') + ')' if top else 'sonuç yok'}")
            if n_ok % 5 == 0:
                wb.save(a.sablon)
            time.sleep(a.bekle)
    finally:
        wb.save(a.sablon)
    print(f"\n{n_ok} sorgu kaydedildi → {a.sablon}")
    kalan = len(kayit) - n_ok
    if kalan > 0:
        print(f"Kalan sorgu: {kalan}. Aynı komutu tekrar çalıştırın.")


if __name__ == "__main__":
    main()
