"""Controlled V3 experiments with per-fold provenance and resumable artifacts."""
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

from ubigraph_core.config import ROOT
from ubigraph_core.data import prepare
from ubigraph_core.spatial import build_spatial
from ubigraph_core.utils import seed_all, write_json, save_torch, file_hash, digest, log
from .data import build_episodes, only_nodes_with_labels, budgeted_support
from .evaluation import apply_layout
from .memory import STBank, bank_audit
from .condensation import initial_bank, condense
from .training import fit, adapt, make_bundle, STForecaster, evaluate_model


def identity(cfg):
    files = sorted(p for folder in ("ubigraph_core", "ubigraph_st")
                   for p in (ROOT/folder).glob("*.py"))
    result = {"config": cfg.asdict(),
            "inputs": {k: file_hash(cfg.path(k)) for k in ("raw_csv", "metadata_csv")},
            "code": {str(p.relative_to(ROOT)): file_hash(p) for p in files}}
    if cfg.poi_csv:
        result["inputs"]["poi_csv"] = file_hash(cfg.path("poi_csv"))
    for name in ("validation_layout", "test_layout"):
        if getattr(cfg, name):
            result["inputs"][name] = file_hash(cfg.path(name))
    return result


def metric_record(truth, prediction, capacity):
    e = np.asarray(prediction)-np.asarray(truth)
    variance = ((np.asarray(truth)-np.mean(truth))**2).sum()
    return {"n": len(e), "mae_spaces": float(np.abs(e).mean()*capacity),
            "rmse_spaces": float(np.sqrt((e**2).mean())*capacity),
            "nmae": float(np.abs(e).mean()), "nrmse": float(np.sqrt((e**2).mean())),
            "r2": float(1-(e**2).sum()/variance) if variance > 1e-12 else None}


def record_prediction(output, protocol, seed, method, mode, query, prediction, results, target=None, suffix=""):
    nodes = [target] if target else query.names
    for name in nodes:
        j = query.names.index(name)
        valid = query.labeled[:, j].numpy()
        truth = query.y[:, j].numpy()[valid]
        pred = prediction[:, j].numpy()[valid]
        capacity = float(query.capacities[j])
        key = {"protocol": protocol, "seed": seed, "method": method, "mode": mode+suffix, "garage": name}
        results.append({**key, **metric_record(truth, pred, capacity)})
        table = pd.DataFrame({"anchor_time": pd.to_datetime(query.anchor_ns.numpy()[valid]),
                              "prediction_time": pd.to_datetime(query.label_ns.numpy()[valid]),
                              "capacity": capacity, "truth": truth*capacity, "prediction": pred*capacity})
        folder = output/f"seed_{seed}"/"predictions"/protocol
        folder.mkdir(parents=True, exist_ok=True)
        table.to_csv(folder/f"{name}_{method}_{mode}{suffix}.csv", index=False, float_format="%.8f")


def modes_for(method, full, cfg):
    if not cfg.ablations:
        return ("attention",)
    if full:
        if method == "groc":
            return ("attention", "local", "gate", "uniform", "no_prior", "no_residual")
        return ("attention", "local") if method in ("full", "random") else ("attention",)
    return ("attention", "local", "gate") if method == "groc" else ("attention",)


def cached_bank(folder, real, prior, cfg, seed, method):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f"{method}.pt"
    if path.exists() and path.with_suffix(".json").exists():
        return STBank.fromdict(torch.load(path, weights_only=True)), json.loads(path.with_suffix(".json").read_text())
    started = time.perf_counter()
    if method == "full":
        bank, summary = STBank(real), {"kind": "full", "original_episodes": len(real)}
    elif method == "random":
        bank, ids = initial_bank(real, cfg, seed, True)
        summary = {"kind": method, "initial_indices": ids.tolist(), "original_episodes": len(real)}
    else:
        bank, summary, trace = condense(real, prior, cfg, seed, method == "groc", str(folder/method))
        pd.DataFrame(trace).to_csv(path.with_suffix(".trace.csv"), index=False)
    summary.update(seconds=time.perf_counter()-started, **bank_audit(bank, cfg))
    save_torch(path, bank.asdict())
    write_json(path.with_suffix(".json"), summary)
    return bank, summary


def run(cfg, output, resume=False):
    cfg.validate()
    output = output.resolve()
    info = identity(cfg)
    run_hash = digest(info)
    manifest_path = output/"manifest.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest_path.exists():
            raise ValueError("Use an empty output directory, or --resume for an identical run")
        if json.loads(manifest_path.read_text())["run_hash"] != run_hash:
            raise ValueError("Resume rejected: source/config/data changed")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seed_all(cfg.seeds[0], cfg.torch_threads)
    manifest = {"format_version": 3, "run_hash": run_hash, "identity": info, "status": "running",
                "python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(),
                "device": "cpu", "client_date": "2026-10-26", "note": "Offline parking model training"}
    write_json(manifest_path, manifest)
    write_json(output/"config.json", cfg.asdict())
    prepared = prepare(cfg, output/"prepared")
    parts = build_episodes(prepared, cfg)
    names = parts["train"].names
    spatial = build_spatial(prepared.metadata, cfg, output/"spatial")
    ids = [spatial.garages.index(g) for g in names]
    prior_all = torch.tensor(spatial.similarity[np.ix_(ids, ids)], dtype=torch.float32)
    floor = min(int(parts["test"].anchor_ns.min()), pd.Timestamp(cfg.val_end).value-cfg.horizon*cfg.freq_minutes*60*10**9)
    # No validation label later than the earliest serving origin may select a model.
    parts["val"] = parts["val"].subset(parts["val"].label_ns <= floor)
    for split, layout in (("val", cfg.validation_layout), ("test", cfg.test_layout)):
        if layout:
            parts[split] = apply_layout(parts[split], cfg.path("validation_layout" if split == "val" else "test_layout"))
    write_json(output/"episode_audit.json", {
        "garages": names, "history_availability_uses_future_labels": False,
        "splits": {k: {"episodes": len(v), "labeled_by_garage": dict(zip(names, v.labeled.sum(0).tolist())),
                       "first_label": str(pd.Timestamp(int(v.label_ns.min()))),
                       "last_label": str(pd.Timestamp(int(v.label_ns.max())))} for k, v in parts.items()},
        "first_serving_anchor_ns": floor, "spatial_prior": spatial.details,
        "fewshot_budget": cfg.fewshot, "input_protocol": "current histories of all available eligible garages"})
    results, costs, bank_checks, budgets, sensitivities, parity = [], [], [], [], [], []
    serving = {"format_version": 3, "run_hash": run_hash, "seeds": {}}
    main = "groc" if "groc" in cfg.methods else cfg.methods[-1]
    for seed in cfg.seeds:
        seed_all(seed, cfg.torch_threads)
        serving["seeds"][str(seed)] = {"full_history": {}, "fewshot": {}}
        for target in [None]+(names if cfg.folds else []):
            protocol = "full_history" if target is None else "fewshot"
            tag = "all" if target is None else "without_"+target
            folder = output/f"seed_{seed}"/tag
            source_names = names if target is None else [g for g in names if g != target]
            source_ids = [names.index(n) for n in source_names]
            prior = prior_all[np.ix_(source_ids, source_ids)]
            train = only_nodes_with_labels(parts["train"].nodes(source_names))
            validation = only_nodes_with_labels(parts["val"].nodes(source_names))
            if target:
                support = budgeted_support(parts["train"], target, cfg.fewshot)
            for method in cfg.methods:
                log(f"ST seed={seed}, {tag}/{method}")
                bank, bank_info = cached_bank(folder/"banks", train, prior, cfg, seed, method)
                bank_checks.append({"seed": seed, "fold": tag, "method": method, **bank_info})
                if target and target in bank.episodes.names:
                    raise AssertionError("Held-out target leaked into source condensation")
                for mode in modes_for(method, target is None, cfg):
                    model_path = folder/"models"/f"{method}_{mode}.pt"
                    meta_path = model_path.with_suffix(".json")
                    if model_path.exists() and meta_path.exists():
                        predictor = STForecaster.load(model_path)
                        meta = json.loads(meta_path.read_text())
                    else:
                        model, meta, curve = fit(bank, validation, prior, cfg, seed, mode)
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(curve).to_csv(model_path.with_suffix(".curve.csv"), index=False)
                        adapter, final_bank, final_prior = None, bank, prior
                        if target:
                            final_bank, adapter, adaptation, trace = adapt(model, bank, support, target, prior_all, cfg, mode)
                            final_prior = prior_all
                            meta["adaptation"] = adaptation
                            pd.DataFrame(trace).to_csv(model_path.with_suffix(".adaptation.csv"), index=False)
                        provenance = {"source_names": source_names, "target": target,
                                      "source_train_last_label_ns": int(train.label_ns.max()),
                                      "source_selection_last_label_ns": int(validation.label_ns.max()),
                                      "target_label_budget": cfg.fewshot if target else None,
                                      "input_protocol": "available synchronous current histories",
                                      "synthetic_kind": bank.kind}
                        bundle = make_bundle(model, final_bank, final_prior, cfg, mode, floor, adapter, target, provenance)
                        save_torch(model_path, bundle)
                        write_json(meta_path, meta)
                        predictor = STForecaster(bundle)
                    costs.append({"seed": seed, "fold": tag, "method": method, "mode": mode,
                                  "fit_seconds": meta["fit_seconds"], "selected_epoch": meta["selected_epoch"],
                                  "parameters": meta["parameters"], "checkpoint_bytes": model_path.stat().st_size,
                                  "adaptation_seconds": meta.get("adaptation", {}).get("seconds", 0)})
                    test = parts["test"]
                    if target:
                        test = test.subset(test.labeled[:, names.index(target)])
                    eval_start = time.perf_counter()
                    prediction, detail = predictor.predict(test, info=True)
                    costs[-1]["test_seconds"] = time.perf_counter()-eval_start
                    record_prediction(output, protocol, seed, method, mode, test, prediction, results, target)
                    if target:
                        budgets.append({"seed": seed, "target": target, "method": method, "mode": mode,
                                        **meta["adaptation"], "source_target_node_overlap": 0})
                    if method == main and mode == "attention":
                        # Same weights and memory, remove other CURRENT histories as a sensitivity check.
                        # This is not a separately trained equal-input baseline; label it accordingly.
                        focus = [target] if target else names
                        for name in focus:
                            j = names.index(name)
                            valid = test.labeled[:, j]
                            q = test.subset(valid)
                            d = dict(q.__dict__)
                            av = torch.zeros_like(q.available)
                            av[:, j] = q.available[:, j]
                            d["available"] = av
                            from .data import Episodes
                            missing = Episodes(**d)
                            without = predictor.predict(missing)
                            change = (without[:, j]-prediction[valid, j])*q.capacities[j]
                            sensitivities.append({"seed": seed, "protocol": protocol, "garage": name,
                                                  "intervention": "mask_other_current_histories_same_checkpoint",
                                                  "mean_abs_change_spaces": float(change.abs().mean()),
                                                  "max_abs_change_spaces": float(change.abs().max()),
                                                  "mean_self_attention": float(detail["spatial_weights"][valid, j, j].mean())})
                            record_prediction(output, protocol, seed, method, mode, missing, without, results,
                                              name, "_others_missing")
                        sample = test.subset(np.arange(min(3, len(test))))
                        reloaded = STForecaster.load(model_path)
                        expected = predictor.predict(sample)
                        actual = reloaded.predict(sample)
                        error = float(((expected-actual)*sample.capacities).abs().max())
                        if error > 1e-5:
                            raise AssertionError("Saved model differs from in-memory model")
                        parity.append({"seed": seed, "fold": tag, "max_reload_error_spaces": error})
                        if target:
                            serving["seeds"][str(seed)]["fewshot"][target] = str(model_path.relative_to(output))
                        else:
                            serving["seeds"][str(seed)]["full_history"]["all"] = str(model_path.relative_to(output))
                        if seed == cfg.seeds[0]:
                            examples = output/"examples"
                            examples.mkdir(exist_ok=True)
                            for name in ([target] if target else [names[0]]):
                                j = names.index(name)
                                row = int(torch.where(test.labeled[:, j])[0][0])
                                anchor = pd.Timestamp(int(test.anchor_ns[row]))
                                tables = []
                                for g in names:
                                    grid = prepared.grids[g].loc[:anchor].tail(cfg.window)
                                    tables.append(pd.DataFrame({"garagecode": g, "timestamp": grid.index,
                                        "available": grid.available_rate.to_numpy()*prepared.splits[g]["train"].capacity,
                                        "capacity": prepared.splits[g]["train"].capacity,
                                        "observed": grid.observed.to_numpy().astype(int), "age": grid.age.to_numpy()}))
                                pd.concat(tables).to_csv(examples/f"{protocol}_{name}_histories.csv", index=False)
                                write_json(examples/f"{protocol}_{name}_expected.json", {
                                    "garage": name, "protocol": protocol, "as_of": str(anchor),
                                    "prediction_time": str(pd.Timestamp(int(test.label_ns[row]))),
                                    "predicted_available": float(prediction[row, j]*test.capacities[j]), "seed": seed})
                    log(f"ST evaluated {seed}/{tag}/{method}/{mode}")
            # Persistence is identical across methods: record once per fold/seed.
            test = parts["test"]
            record_prediction(output, protocol, seed, "persistence", "local", test,
                              test.x[..., cfg.window-1], results, target)
    frame = pd.DataFrame(results)
    frame.to_csv(output/"metrics_by_garage.csv", index=False)
    records = []
    for keys, group in frame.groupby(["protocol", "seed", "method", "mode"]):
        w = group.n/group.n.sum()
        records.append(dict(zip(("protocol", "seed", "method", "mode"), keys),
            mae_spaces=float((w*group.mae_spaces).sum()), rmse_spaces=float(np.sqrt((w*group.rmse_spaces**2).sum())),
            macro_nmae=float(group.nmae.mean()), macro_r2=float(group.r2.mean()), r2_defined_garages=int(group.r2.notna().sum()),
            test_predictions=int(group.n.sum())))
    by_seed = pd.DataFrame(records)
    by_seed.to_csv(output/"metrics_by_seed.csv", index=False)
    means = by_seed.groupby(["protocol", "method", "mode"])[["mae_spaces", "rmse_spaces", "macro_nmae"]].agg(["mean", "std"])
    means.columns = ["_".join(c) for c in means.columns]
    means.reset_index().to_csv(output/"metrics_across_seeds.csv", index=False)
    for filename, data in (("training_costs", costs), ("bank_checks", bank_checks), ("fewshot_budget", budgets),
                            ("input_sensitivity", sensitivities), ("reload_parity", parity)):
        pd.DataFrame(data).to_csv(output/f"{filename}.csv", index=False)
    serving["default_protocol"] = "full_history"
    serving["protocol_configs"] = {p: "config.json" for p in ("full_history", "fewshot")}
    serving["evaluation"] = {p: {"labels_per_seed": int(group.test_predictions.iloc[0]),
        "history_points": cfg.window, "scope": "固定测试标签；单独按训练协议统计"}
        for p, group in by_seed.groupby("protocol")}
    write_json(output/"serving.json", serving)
    manifest.update(status="evaluated", elapsed_seconds=time.perf_counter()-started, eligible_garages=names)
    write_json(manifest_path, manifest)
    from .report import build_report
    build_report(output)
    manifest.update(status="complete")
    write_json(manifest_path, manifest)
    log(f"ST complete: {output}, elapsed={manifest['elapsed_seconds']:.1f}s")
    return manifest
