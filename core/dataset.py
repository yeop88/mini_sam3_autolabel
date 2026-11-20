"""세션들 -> YOLO 학습 데이터셋 (dataset/<이름>/images|labels/train|val, data.yaml).

- 클래스: 세션마다 클래스 목록이 달라도 **이름으로 합친다** (모든 물체 모드는 세션마다 이름이 다름).
  합친 목록 = 세션 순서대로 처음 나온 순서. 라벨 파일의 번호를 합친 목록 번호로 바꿔 새로 쓴다.
- 작업: 박스 형식 세션이 하나라도 있으면 detect (다각형 라벨도 박스로 쓰임), 모두 분할이면 segment.
- 검증 분할: 연속 프레임은 서로 비슷하므로 '연속 block 장 묶음' 단위로 떼어 낸다 (장 단위면 점수가 부풀려짐).
"""
import glob
import os
import random
import shutil

from core.paths import DATASET_DIR, END2_WEIGHTS, WEIGHTS_DIR
from core.session import session_info


def merge_classes(session_dirs):
    """-> (합친 클래스 이름 list, task('segment'|'detect'), 세션별 정보 list)"""
    infos = [session_info(d) for d in session_dirs]
    names = []
    for s in infos:
        for n in s["names"]:
            if n not in names:
                names.append(n)
    task = "detect" if any(s["format"] == "box" for s in infos) else "segment"
    return names, task, infos


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def build(session_dirs, name, val_ratio=0.15, block=10, seed=0, log=print):
    """-> (data.yaml 경로, counts, names, task)"""
    names, task, infos = merge_classes(session_dirs)
    rng = random.Random(seed)
    train, val = [], []
    remap = {}           # 세션 폴더 -> {세션 번호: 합친 번호}
    for s in infos:
        d = s["dir"]
        remap[d] = {i: names.index(n) for i, n in enumerate(s["names"])}
        imgs = sorted(glob.glob(os.path.join(d, "images", "*.jpg")))
        blocks = [imgs[i:i + block] for i in range(0, len(imgs), block)]
        rng.shuffle(blocks)
        n_val = max(1, round(len(blocks) * val_ratio)) if len(blocks) >= 3 else 0
        for j, b in enumerate(blocks):
            (val if j < n_val else train).extend(b)
        log(f"세션 {s['name']} ({s['format']}, 클래스 {len(s['names'])}개): {len(imgs)}장 ({len(blocks)}묶음, val {n_val}묶음)")
    if not val:                       # 세션이 너무 작으면 장 단위로라도 떼어 냄
        rng.shuffle(train)
        k = max(1, int(len(train) * val_ratio))
        val, train = train[:k], train[k:]
    if len(train) < 10:
        raise ValueError(f"학습 이미지가 너무 적습니다 ({len(train)}장). 더 찍어 주세요 (클래스당 100장 이상 권장).")

    ds = os.path.join(DATASET_DIR, name)
    counts = {}
    for split, files in (("train", train), ("val", val)):
        idir = os.path.join(ds, "images", split)
        ldir = os.path.join(ds, "labels", split)
        os.makedirs(idir, exist_ok=True)
        os.makedirs(ldir, exist_ok=True)
        inst = [0] * len(names)
        for ip in files:
            sdir = os.path.dirname(os.path.dirname(ip))
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(sdir, "labels", stem + ".txt")
            _link_or_copy(ip, os.path.join(idir, stem + ".jpg"))
            out = []
            if os.path.isfile(lp):
                with open(lp, encoding="utf-8") as f:
                    for line in f:
                        v = line.split()
                        if not v:
                            continue
                        c = remap[sdir].get(int(v[0]))
                        if c is None:
                            continue
                        inst[c] += 1
                        out.append(" ".join([str(c)] + v[1:]))
            with open(os.path.join(ldir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(out) + ("\n" if out else ""))
        counts[split] = dict(images=len(files), instances=dict(zip(names, inst)))
    yaml_path = os.path.join(ds, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {ds.replace(os.sep, '/')}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {_yaml_str(n)}\n" for i, n in enumerate(names)))
    return yaml_path, counts, names, task


def _yaml_str(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def is_seg_model(name):
    return "-seg" in os.path.basename(name)


def pretrained_choices(task="segment"):
    """학습 시작 가중치 후보: 이 폴더 weights\\, end_2 weights\\ 에 있는 YOLO 가중치 + 내려받을 수 있는 이름.
    task='segment' -> *-seg.pt, 'detect' -> 박스 모델 (yolo*.pt, -seg 아님)"""
    want_seg = task == "segment"
    found = []
    for d in (WEIGHTS_DIR, END2_WEIGHTS):
        for p in sorted(glob.glob(os.path.join(d, "yolo*.pt"))):
            n = os.path.basename(p)
            if n.startswith("yoloe") or is_seg_model(n) != want_seg or n in found:
                continue
            found.append(n)
    extra = (["yolo26n-seg.pt", "yolo26s-seg.pt", "yolo11n-seg.pt", "yolo11s-seg.pt"] if want_seg
             else ["yolo26n.pt", "yolo26s.pt", "yolo11n.pt", "yolo11s.pt"])
    return found + [n for n in extra if n not in found]


def default_model(task):
    return "yolo26s-seg.pt" if task == "segment" else "yolo26s.pt"


def resolve_model(name):
    if os.path.isfile(name):
        return name
    for d in (WEIGHTS_DIR, END2_WEIGHTS):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(WEIGHTS_DIR, name)      # 없으면 Ultralytics 가 weights\ 로 내려받음
