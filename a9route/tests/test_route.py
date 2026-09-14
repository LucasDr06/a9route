# -*- coding: utf-8 -*-
"""路线脚本格式的离线测试（**不碰视频、不碰设备**）。

锁的是那份路线脚本的**逐条语义** —— 这是最容易"看起来对、实际全错"的地方：

  T1 五种操作的解析：选路 / 氮气 / 漂移 / 关自动驾驶 / 360
  T2 用户给的**原样例子**必须都能解析（含 `50,D:100|N:100:1:100`）
  T3 错误与提示：中文标点、缺逗号、选路越界、氮气 0 次、间隔为 0
  T4 排序 / 合并 / 警告：同百分比、漂移窗口里的氮气
  T5 扁平逗号流：一整行的写法也认，并且能原地吐回来

    python -m a9route.tests.test_route
"""
from __future__ import annotations

import sys

from a9route.core import route as R
from a9route.tests.support import check, summary


# ------------------------------------------------------------------ T1
def t1_ops():
    print("\n=== T1 五种操作的解析 ===")
    op = R.parse_op("32")
    check("选路 32 -> 3 个图标选第 2 个", op.kind == R.OP_ROUTE
          and op.icons == 3 and op.index == 2, op.describe())
    op = R.parse_op("21")
    check("选路 21 -> 2 个里选第 1 个", op.icons == 2 and op.index == 1, op.describe())
    op = R.parse_op("44")
    check("选路 44 -> 4 个里选第 4 个", op.icons == 4 and op.index == 4, op.describe())
    op = R.parse_op("N:0:2:750")
    check("氮气 N:0:2:750 -> 延时0/2次/间隔750", op.kind == R.OP_NITRO
          and (op.delay_ms, op.times, op.interval_ms) == (0, 2, 750), op.describe())
    op = R.parse_op("N:500:10:100")
    check("氮气 N:500:10:100 -> 延时500/10次/间隔100",
          (op.delay_ms, op.times, op.interval_ms) == (500, 10, 100), op.describe())
    op = R.parse_op("D:3000")
    check("漂移 D:3000 -> 3000ms", op.kind == R.OP_DRIFT and op.ms == 3000, op.describe())
    op = R.parse_op("S:2000")
    check("关自动驾驶 S:2000 -> 2000ms", op.kind == R.OP_AUTOPILOT and op.ms == 2000,
          op.describe())
    op = R.parse_op("360")
    check("360", op.kind == R.OP_SPIN, op.describe())


# ------------------------------------------------------------------ T2
def t2_user_examples():
    print("\n=== T2 用户给的例子原样可解析 ===")
    text = "\n".join([
        "21,22",
        "35,N:0:2:750",
        "40,D:3000",
        "50,D:100|N:100:1:100",
        "50,S:2000",
        "60,360",
    ])
    route = R.parse_route(text)
    check("6 行都解析出来了", len(route.entries) == 6, str(len(route.entries)))
    check("第一个是选路 21 -> 2 个里选第 1 个",
          route.entries[0].ops[0].kind == R.OP_ROUTE
          and route.entries[0].ops[0].icons == 2, route.entries[0].describe())
    check("`50,D:100|N:100:1:100` 有两条操作（同时）",
          len([e for e in route.entries if e.percent == 50 and len(e.ops) == 2]) == 1,
          str([e.describe() for e in route.entries]))
    combo = [e for e in route.entries if len(e.ops) == 2][0]
    check(" 组合内容是 D:100 + N:100:1:100",
          combo.ops[0].kind == R.OP_DRIFT and combo.ops[1].kind == R.OP_NITRO
          and combo.ops[1].delay_ms == 100, combo.describe())
    check("同百分比 50 出现两次 -> 有警告", any("50" in w for w in route.warnings),
          str(route.warnings))
    check("行号正确（第 1 行是第一个条目）", route.entries[0].line_no == 1,
          str(route.entries[0].line_no))


# ------------------------------------------------------------------ T3
def t3_errors():
    print("\n=== T3 错误与提示 ===")

    def err(text):
        try:
            R.parse_route(text)
            return ""
        except R.RouteError as exc:
            return str(exc)

    msg = err("35，N:0:2:750")
    check("中文逗号 -> 报错并给英文写法", "中文标点" in msg and "35,N:0:2:750" in msg, msg[:80])
    msg = err("35，N:0:2:750".replace("，", "："))
    check("中文冒号 -> 也提示", "中文标点" in msg, msg[:80])
    check("缺逗号 -> 明确提示格式", "少写了逗号" in err("35 N:0:2:750"),
          err("35 N:0:2:750")[:60])
    check("百分比越界 -> 报错", "不在 0~100" in err("120,D:1000"))
    check("选路越界（23 里选第 3 个）-> 报错", "不合法" in err("30,23"), err("30,23")[:60])
    check("氮气次数 0 -> 报错", "不能是 0" in err("30,N:0:0:100"))
    check("氮气点 2 次但间隔 0 -> 报错", "间隔不能是 0" in err("30,N:0:2:0"))
    check("只点 1 次时间隔写 0 是允许的（间隔没意义）",
          R.parse_op("N:0:1:0").times == 1, R.parse_op("N:0:1:0").describe())
    check("点 20 次不再报错（用户真实路线里就有 N:0:20:100）",
          R.parse_op("N:0:20:100").times == 20, R.parse_op("N:0:20:100").describe())
    check("漂移 0ms -> 报错", "大于 0" in err("30,D:0"))
    check("看不懂的操作 -> 报错里列出全部写法",
          "N:延时:次数:间隔" in err("30,X:1"), err("30,X:1")[:60])
    check("空文件 -> 报错", "一条有效操作都没有" in err("\n\n# 只有注释\n"), "ok")
    check("注释与空行被忽略", len(R.parse_route("# 注释\n\n30,D:1000\n").entries) == 1)
    # 中文标点检查**只针对数据行**：注释里带全角标点必须能过
    # （自动生成的路线草稿注释里就有「，：」，早期把检查放在剥注释之前，
    #   结果自己生成的路线被自己判非法 ✗ —— webvideo 测试抓出来的这个 bug）
    try:
        r2 = R.parse_route("# 说明：这是注释，带全角标点没关系；\n30,D:1000\n")
        check("注释里的中文标点不影响解析", len(r2.entries) == 1, str(r2.entries))
    except R.RouteError as exc:
        check("注释里的中文标点不影响解析", False, str(exc))
    check(" 但数据行里的中文标点仍然要报错",
          "中文标点" in err("30，D:1000"), err("30，D:1000")[:50])


# ------------------------------------------------------------------ T4
def t4_sort_warn():
    print("\n=== T4 排序 / 合并 / 警告 ===")
    route = R.parse_route("60,360\n35,N:0:2:750\n40,D:3000\n")
    check("按百分比升序排序", route.percents() == [35, 40, 60], str(route.percents()))
    route2 = R.parse_route("40,D:2000\n40,N:0:1:100\n")
    check("氮气与漂移几乎同时 -> 警告", any("取消漂移" in w for w in route2.warnings),
          str(route2.warnings))
    route2b = R.parse_route("40,D:2000\n42,N:0:1:100\n")
    check("漂移后隔 2 个点再点氮气 -> **不**警告（用户路线里的常规技术动作）",
          route2b.warnings == [], str(route2b.warnings))
    route3 = R.parse_route("40,D:100|N:100:1:100\n")
    check("同一条里配 | 不算冲突（用户明确要的用法）", route3.warnings == [],
          str(route3.warnings))
    route4 = R.parse_route("40,D:2000\n50,N:0:1:100\n")
    check("漂移窗口之外 -> 不警告", route4.warnings == [], str(route4.warnings))


# ------------------------------------------------------------------ T5
def t5_flat_stream():
    """**扁平逗号流**（用户 2026-09-13 指定的格式）：一整行也认，并且能吐回一行。"""
    print("\n=== T5 扁平逗号流 ===")
    flat = "1,31,2,N:0:2:750,4,360,40,21|D:2050"
    route = R.parse_route(flat)
    check("一整行扁平流能解析", len(route.entries) == 4, str(len(route.entries)))
    check(" 操作按逗号两两配对",
          [e.percent for e in route.entries] == [1, 2, 4, 40], str(route.percents()))
    check(" 同时操作用 | 也认（一条里两个操作）",
          any(len(e.ops) == 2 for e in route.entries),
          str([e.describe() for e in route.entries]))
    same = R.parse_route("49,22,49,N:0:4:100")
    check("同百分比两条都保留（按条目去重，不按百分比）",
          len(same.entries) == 2, str([e.describe() for e in same.entries]))
    # 用户真实路线（47 条）必须原样可用
    from a9route.paths import ROUTES_DIR
    f = ROUTES_DIR / "user_beach_landing.txt"
    if f.is_file():
        r = R.parse_route(f.read_text(encoding="utf-8"))
        check("routes/user_beach_landing.txt（用户真实路线）能解析",
              len(r.entries) == 47, str(len(r.entries)))
        check(" 且没有硬错误（警告可以有）", True, str(r.warnings[:2]))
    f2 = ROUTES_DIR / "demo.txt"
    if f2.is_file():
        r = R.parse_route(f2.read_text(encoding="utf-8"))
        check("routes/demo.txt（格式说明样例）能解析", len(r.entries) > 0,
              str(len(r.entries)))


def main() -> int:
    t1_ops()
    t2_user_examples()
    t3_errors()
    t4_sort_warn()
    t5_flat_stream()
    return summary("test_route")


if __name__ == "__main__":
    sys.exit(main())
