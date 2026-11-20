"""classes.json: 학습 클래스 이름과 SAM 3 텍스트 프롬프트.

{"classes": [{"name": "cup", "prompts": ["cup", "mug"]}, ...]}
- name    = 학습 클래스 이름
- prompts = SAM 3 에 줄 영어 명사구 (한 클래스에 여러 개 가능, 같은 물체를 두 번 잡으면 점수 높은 것만 남김)
"""
from core.paths import CLASSES_JSON
from core.util import read_json, write_json

NOTE = ("name = 학습 클래스 이름(영어 권장), prompts = SAM 3 에 줄 텍스트(영어 명사구). "
        "한 클래스에 프롬프트 여러 개 가능")


def load(path=CLASSES_JSON):
    cfg = read_json(path, {}) or {}
    out = []
    for c in cfg.get("classes", []):
        name = str(c.get("name", "")).strip()
        if not name:
            continue
        prompts = [str(p).strip() for p in (c.get("prompts") or [name]) if str(p).strip()] or [name]
        out.append(dict(name=name, prompts=prompts))
    return out


def save(classes, path=CLASSES_JSON):
    cfg = read_json(path, {}) or {}
    extra = {k: v for k, v in cfg.items() if k.startswith("_")}
    extra.setdefault("_설명", NOTE)
    write_json(path, dict(**extra, classes=[dict(name=c["name"], prompts=list(c["prompts"])) for c in classes]))


def check(classes):
    """문제가 있으면 설명 문자열, 없으면 빈 문자열"""
    if not classes:
        return "클래스가 없습니다. 클래스를 하나 이상 추가하세요."
    names = [c["name"] for c in classes]
    if len(set(names)) != len(names):
        return "클래스 이름이 겹칩니다."
    for c in classes:
        if not c["prompts"]:
            return f"'{c['name']}' 에 프롬프트가 없습니다."
    return ""


def flatten(classes):
    """-> (names, prompt_list, prompt_to_cls)  prompt_to_cls[i] = prompt_list[i] 의 클래스 번호"""
    names, prompt_list, prompt_to_cls = [], [], []
    for i, c in enumerate(classes):
        names.append(c["name"])
        for p in c["prompts"]:
            prompt_list.append(p)
            prompt_to_cls.append(i)
    return names, prompt_list, prompt_to_cls
