# -*- coding: utf-8 -*-
"""Web 端可编辑的参数面板（滑块/输入框那套）—— **校验在后端，不许只靠前端**。

最要紧的两条：

* 一个不合格就**一个都不写**（不留半套配置 —— 那种状态最难查）；
* 落盘之后**真的生效**（`config.current()` 里是新值，`intent` 的运行值也跟着变）。

还有一条容易被忽略但很坑的：「这一项在当前后端下到底生效吗」——
页面必须能标明"调了没反应"的项（模型后端下，只对启发式生效的项要标出来）。
"""
from __future__ import annotations

import sys
from pathlib import Path

from a9route import config as cfgmod
from a9route import paths
from a9route.core import intent as IT
from a9route.tests.support import check, summary, tmp_dir
from a9route.web import config_ui as CU

TMP = Path(".")


def t1_table():
    print("\n=== T1 参数表本身 ===")
    data = CU.payload(backends={"keys": "onnx", "choice": "onnx"})
    keys = [f["key"] for f in data["fields"]]
    check("参数表非空、每项都有中文名/范围/默认值",
          len(keys) >= 15 and all(f["zh"] and f["kind"] for f in data["fields"]),
          "{0} 项".format(len(keys)))
    for want in ("scan.choice_idle_hold", "scan.choice_min_hold",
                 "scan.choice_cross_tol", "vision.choice_conf"):
        check("  选了「{0}」".format(want), want in keys, "")
    check("  每项都带 min/max（滑块要用）",
          all("min" in f and "max" in f for f in data["fields"]), "")
    # ⚠️ "调了没反应"是最坑的误导：模型后端下，只对启发式生效的项要标出来
    inactive = [f["key"] for f in data["fields"] if not f["applies"]]
    check("**当前用模型时，只对启发式生效的项被标成「不生效」**",
          "vision.choice_param2" in inactive and "vision.brake_white_thr" in inactive,
          str(inactive[:4]))
    check("  而且给了人能看懂的原因",
          all(f["note"] for f in data["fields"] if not f["applies"]), "")


def t2_validate():
    print("\n=== T2 校验：范围 / 类型 / 未知键 ===")
    bad_cases = [
        ("scan.choice_idle_hold", -1, "不能小于"),
        ("scan.choice_idle_hold", 9999, "不能大于"),
        ("scan.choice_idle_hold", "abc", "要一个数"),
        ("vision.choice_conf", 5.0, "不能大于"),
        ("vision.brake_key_box", "1,2,3", "4 个数"),
        ("vision.brake_key_box", [10, 10, 0, 50], "宽高必须"),
    ]
    for key, val, want in bad_cases:
        _, err = CU.coerce(CU.BY_KEY[key], val)
        check("  {0}={1!r} 被拒：{2}".format(key, val, want), want in err, err)
    ok_v, err = CU.coerce(CU.BY_KEY["scan.choice_idle_hold"], "12")
    check("  正常值能过（字符串形式也认）", ok_v == 12 and not err, str(ok_v))

    res = CU.save({"scan.choice_idle_hold": 12, "不存在的键": 1})
    check("**有一个不合格 -> 一个都不写**（不留半套配置）",
          res["ok"] is False and res["written"] == []
          and any("不认识" in e for e in res["errors"]), str(res["errors"])[:80])
    check("  而且被拒的那一项**没被写进去**",
          (cfgmod.current().get("scan") or {}).get("choice_idle_hold") != 12,
          str((cfgmod.current().get("scan") or {}).get("choice_idle_hold")))


def t3_save_and_effect():
    print("\n=== T3 落盘 + 生效 + 恢复默认 ===")
    res = CU.save({"scan.choice_idle_hold": 6, "intent.drift_min": 0.30})
    check("合法值：落盘成功", res["ok"] and res["written"], str(res)[:80])
    check("  **真的生效了**（config.current 里是新值）",
          (cfgmod.current().get("scan") or {}).get("choice_idle_hold") == 6,
          str((cfgmod.current().get("scan") or {}).get("choice_idle_hold")))
    check("  而且写进了 config.json（不是只在内存里）",
          "choice_idle_hold" in (paths.CONFIG_FILE.read_text(encoding="utf-8")
                                 if paths.CONFIG_FILE.is_file() else ""), "")
    check("  `config.apply` 之后 intent 的运行值也跟着变",
          "CHOICE_IDLE_HOLD" in dir(IT) and IT.CHOICE_IDLE_HOLD == 6,
          str(getattr(IT, "CHOICE_IDLE_HOLD", None)))
    res2 = CU.reset(["scan.choice_idle_hold"])
    check("恢复默认：回到内置值", res2["ok"]
          and (cfgmod.current().get("scan") or {}).get("choice_idle_hold")
          == CU._default_of("scan.choice_idle_hold"),
          str((cfgmod.current().get("scan") or {}).get("choice_idle_hold")))
    # ⚠️ 恢复默认要**把键从文件里删掉**，而不是把默认值写进去 ——
    # 否则 config.json 会越写越长，`config list --changed` 也全是噪音
    txt = paths.CONFIG_FILE.read_text(encoding="utf-8")
    check("  恢复默认后 config.json 里**不再有这个键**（只记真正的覆盖）",
          "choice_idle_hold" not in txt, txt[-120:].replace("\n", " "))
    res3 = CU.reset(None)
    check("全部恢复默认也不报错", res3["ok"], str(res3.get("errors")))


def main() -> int:
    global TMP
    TMP = tmp_dir()
    t1_table()
    t2_validate()
    t3_save_and_effect()
    return summary("test_config_ui")


if __name__ == "__main__":
    sys.exit(main())
