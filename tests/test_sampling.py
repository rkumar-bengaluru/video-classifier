import pandas as pd

from sampling import stratified_sample

DOMAIN_SIZES = {"Knowledge": 90, "Film & Television": 40, "Sports Competition": 50,
                "Artistic Performance": 40, "Life Record": 70, "Multilingual": 10}


def fake_videomme():
    rows, i = [], 0
    for duration in ("short", "medium", "long"):
        for domain, size in DOMAIN_SIZES.items():
            for _ in range(size):
                rows.append({"video_id": f"{i:03d}", "youtube_id": f"yt{i}", "duration": duration,
                             "domain": domain, "sub_category": f"{domain}-sub"})
                i += 1
    return pd.DataFrame(rows)


def test_sample_is_ten_percent_per_stratum_with_disjoint_spares():
    s = stratified_sample(fake_videomme())
    prim = s[s.role == "primary"]
    assert len(prim) == 90
    per = prim.groupby(["duration", "domain"]).size()
    for (duration, domain), n in per.items():
        assert n == round(DOMAIN_SIZES[domain] * 0.1)
    spares = s[s.role == "spare"]
    assert len(spares) == 36
    assert set(spares.video_id).isdisjoint(prim.video_id)


def test_sample_is_deterministic_for_seed():
    a = stratified_sample(fake_videomme(), seed=42)
    b = stratified_sample(fake_videomme(), seed=42)
    c = stratified_sample(fake_videomme(), seed=7)
    assert a.video_id.tolist() == b.video_id.tolist()
    assert a.video_id.tolist() != c.video_id.tolist()
