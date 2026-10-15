"""CLI entry point: python src/main.py <command>."""
import argparse
import sys

# Each entry: name -> (help, configure(parser), handler(args) -> int)
SUBCOMMANDS = {}


def _read(path):
    import pandas as pd
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def cmd_sample(args):
    import config as C
    from sampling import fetch_parquet, load_videomme, stratified_sample
    parquet = fetch_parquet(C.VIDEOMME_PARQUET_URL, C.DATA / "videomme.parquet")
    plan = stratified_sample(load_videomme(parquet))
    C.ART.mkdir(parents=True, exist_ok=True)
    plan.to_csv(C.ART / "sample_plan.csv", index=False)
    print(plan.groupby("role").size().to_string())


SUBCOMMANDS["sample"] = ("draw the stratified 10%% VideoMME sample", lambda p: None, cmd_sample)


def cmd_download(args):
    import config as C
    from download import download_sample
    final, missing = download_sample(_read(C.ART / "sample_plan.csv"))
    final.to_csv(C.ART / "sample.csv", index=False)
    print(f"{len(final)} videos ready, {int((final.replaced != '').sum())} replaced by spares, "
          f"unfilled slots: {missing or 'none'}")


SUBCOMMANDS["download"] = ("download sampled videos (yt-dlp, <=240p)", lambda p: None, cmd_download)


def _ready_ids():
    import config as C
    sample = _read(C.ART / "sample.csv")
    return sample, [v for v in sample.video_id if (C.FRAMES / f"{v}.npz").exists()]


def cmd_extract(args):
    import config as C
    from featurize import extract_all
    failed = extract_all(_read(C.ART / "sample.csv"))
    for vid, reason in failed:
        print(f"FAILED {vid}: {reason}")
    return 1 if failed else 0


def cmd_label_sheets(args):
    import cv2
    import numpy as np
    import config as C
    from labels import contact_sheet, sheet_pages
    sample, ids = _ready_ids()
    info = sample.set_index("video_id")
    C.SHEETS.mkdir(parents=True, exist_ok=True)
    sheets = []
    for vid in ids:
        row = info.loc[vid]
        title = f"{vid} | {row.sub_category} -> {C.SUBCATEGORY_TO_CLASS[row.sub_category]}"
        sheet = contact_sheet(np.load(C.FRAMES / f"{vid}.npz")["rgb"], title)
        cv2.imwrite(str(C.SHEETS / f"{vid}.jpg"), sheet)
        sheets.append(sheet)
    for k, page in enumerate(sheet_pages(sheets)):
        cv2.imwrite(str(C.SHEETS / f"page_{k:02d}.jpg"), page)
    print(f"wrote {len(sheets)} sheets to {C.SHEETS}")


def cmd_labels(args):
    import config as C
    from labels import build_labels, load_overrides
    df = build_labels(_read(C.ART / "sample.csv"), load_overrides(C.ART / "label_overrides.csv"))
    df.to_csv(C.ART / "labels.csv", index=False)
    print(df.label_final.value_counts().to_string())
    print(f"changed by review: {int(df.label_changed.sum())}/{len(df)}")


def cmd_featurize(args):
    from featurize import featurize_all
    _, ids = _ready_ids()
    featurize_all(ids)


SUBCOMMANDS["extract"] = ("decode frames + bursts for the sample", lambda p: None, cmd_extract)
SUBCOMMANDS["label-sheets"] = ("render contact sheets for label review", lambda p: None, cmd_label_sheets)
SUBCOMMANDS["labels"] = ("build labels.csv from mapping + overrides", lambda p: None, cmd_labels)
SUBCOMMANDS["featurize"] = ("compute motion features and encoder embeddings", lambda p: None, cmd_featurize)


def cmd_evaluate(args):
    from train_eval import run_evaluation
    run_evaluation(n_repeats=args.repeats)


SUBCOMMANDS["evaluate"] = ("run the CV sweep on committed features",
                           lambda p: p.add_argument("--repeats", type=int, default=5), cmd_evaluate)


def cmd_export(args):
    from export import export_production
    export_production()


SUBCOMMANDS["export"] = ("export the production head to ONNX", lambda p: None, cmd_export)


PREDICT_CHUNK = 32


def cmd_predict(args):
    import json
    from predict import Predictor, collect_videos
    videos = collect_videos(args.paths)
    if not videos:
        print("error: no video files found in the given paths", file=sys.stderr)
        return 1
    predictor, failed = Predictor(), False
    for i in range(0, len(videos), PREDICT_CHUNK):
        for r in predictor.predict_many(videos[i:i + PREDICT_CHUNK]):
            print(json.dumps(r), flush=True)  # stream: one line per video as its chunk finishes
            failed = failed or "error" in r
    return 1 if failed else 0


def cmd_benchmark(args):
    import json
    import config as C
    from benchmark import run
    from predict import Predictor, collect_videos
    out = run(Predictor(), collect_videos(args.paths))
    (C.ART / "benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def cmd_fetch_encoder(args):
    import json
    import config as C
    from encoders import load
    if args.all:
        wanted = list(json.loads((C.FEATURES / "encoders.json").read_text()).values())
    else:
        enc = json.loads((C.MODELS / "production.json").read_text())["encoder"]
        wanted = [enc] if enc else []
    for name, pretrained in wanted:
        load(name, pretrained)
        print(f"encoder ready: {name}/{pretrained}")
    if not wanted:
        print("production config uses no encoder; nothing to fetch")


SUBCOMMANDS["predict"] = ("classify videos (files or directories)",
                          lambda p: p.add_argument("paths", nargs="+"), cmd_predict)
SUBCOMMANDS["benchmark"] = ("time the production pipeline at batch 1 and 32",
                            lambda p: p.add_argument("paths", nargs="+"), cmd_benchmark)
SUBCOMMANDS["fetch-encoder"] = ("download encoder weights (production, or --all)",
                                lambda p: p.add_argument("--all", action="store_true"), cmd_fetch_encoder)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="main.py", description="Video type classifier for chunking routing")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, (help_text, configure, handler) in SUBCOMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        configure(p)
        p.set_defaults(func=handler)
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
