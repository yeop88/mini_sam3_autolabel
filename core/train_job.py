"""학습 프로세스 진입점. GUI 가 별도 프로세스로 실행한다 (GUI 가 멈추지 않고, 중지하면 프로세스째 끝냄).

    python -m core.train_job runs\\<이름>\\job_config.json

설정 json: sessions(세션 폴더 list), name, model, epochs, imgsz, batch, patience, val_ratio, block, seed, device, workers
결과: dataset\\<이름>\\, runs\\<이름>\\weights\\best.pt, runs\\<이름>\\train_summary.json
마지막 줄에 '@@SUMMARY <train_summary.json 경로>' 를 찍는다 (GUI 가 읽음).
"""
import json
import os
import sys
import time


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main(cfg_path):
    from core.dataset import build, resolve_model
    from core.paths import RUNS_DIR
    from core.util import write_json

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    name = cfg["name"]
    yaml_path, counts, names, task = build(cfg["sessions"], name, cfg.get("val_ratio", 0.15), cfg.get("block", 10),
                                           cfg.get("seed", 0), log=log)
    log(f"작업 {task}, 클래스 {len(names)}개 {names}")
    log(f"데이터: train {counts['train']}, val {counts['val']}")

    from ultralytics import YOLO

    from core.dataset import default_model, is_seg_model
    if task == "detect" and is_seg_model(cfg["model"]):      # 박스 라벨로는 분할 모델을 못 배움
        log(f"[참고] 박스 라벨이 있어 분할 모델 {cfg['model']} 대신 {default_model('detect')} 로 학습합니다.")
        cfg["model"] = default_model("detect")
    model_path = resolve_model(cfg["model"])
    log(f"모델: {model_path}")
    t0 = time.time()
    YOLO(model_path).train(data=yaml_path, epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
                           patience=cfg["patience"], workers=cfg.get("workers", 4), device=cfg.get("device", "0"),
                           seed=cfg.get("seed", 0), project=RUNS_DIR, name=name, exist_ok=True, plots=True,
                           verbose=False)
    run_dir = os.path.join(RUNS_DIR, name)
    best = os.path.join(run_dir, "weights", "best.pt")
    metrics = {}
    try:
        m = YOLO(best).val(data=yaml_path, imgsz=cfg["imgsz"], batch=cfg["batch"], device=cfg.get("device", "0"),
                           verbose=False, project=run_dir, name="val_best", exist_ok=True, plots=False)
        metrics = dict(box_map50=round(float(m.box.map50), 4), box_map50_95=round(float(m.box.map), 4))
        if hasattr(m, "seg"):
            metrics.update(mask_map50=round(float(m.seg.map50), 4), mask_map50_95=round(float(m.seg.map), 4))
    except Exception as e:
        log(f"[참고] best.pt 검증 실패: {e}")
    minutes = round((time.time() - t0) / 60, 1)
    engine = None
    if cfg.get("trt"):       # 실시간용 TensorRT 엔진 미리 만들기 (4 실시간 탭이 바로 씀)
        from core.realtime import export_engine
        log("TensorRT 변환 중 (약 4분)...")
        t1 = time.time()
        try:
            engine = export_engine(best, cfg["imgsz"])
            log(f"TensorRT 변환 끝 ({time.time() - t1:.0f}s): {engine}")
        except Exception as e:
            log(f"[참고] TensorRT 변환 실패 (실시간은 PyTorch 로 됨): {e}")
    summary = os.path.join(run_dir, "train_summary.json")
    write_json(summary, dict(name=name, task=task, sessions=[os.path.basename(d) for d in cfg["sessions"]], names=names,
                             model=model_path, epochs=cfg["epochs"], imgsz=cfg["imgsz"], counts=counts,
                             metrics=metrics, minutes=minutes, best=best, engine=engine))
    log(f"학습 끝 ({(time.time() - t0) / 60:.1f}분): {metrics}")
    print(f"@@SUMMARY {summary}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
