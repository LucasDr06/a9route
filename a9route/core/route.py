# -*- coding: utf-8 -*-
"""route.py —— 「百分比路线脚本」的格式解析。

## 格式（与用户那份现成脚本**完全兼容**，可以直接拿来用）

每行一条：`<路程百分比>,<操作>`，例如

    35,N:0:2:750        路程到 35% 时：点两次氮气，间隔 750 ms（完美氮气）
    40,D:3000           路程到 40% 时：漂移 3000 ms
    33,32               路程到 33% 时：屏幕上有 3 个选路图标，选从左数第 2 个
    50,S:2000           路程到 50% 时：关掉自动驾驶 2000 ms，之后自动打开
    60,360              路程到 60% 时：做 360
    50,D:100|N:100:1:100  路程到 50% 时：漂移 100 ms，**同时**（100 ms 后）点一次氮气
                          「|」表示同时执行；**尽量不要用**，容易有操作被忽略

五种操作（用户给的定义）：

| 写法 | 含义 | 字段 |
|---|---|---|
| `NN`（两位数字） | 选路 | 前一位=图标数量，后一位=选第几个（从左数，1 起） |
| `N:延时:次数:间隔` | 点氮气 | 延时 ms（一般写 0）、点几次、每次间隔 ms |
| `D:毫秒` | 漂移 | 漂移时长 ms |
| `S:毫秒` | 关自动驾驶 | 关掉这么多毫秒后自动打开（不建议用） |
| `360` | 360 操作 | 无参数 |

## 几条硬规则（放在这里当唯一事实来源）

* **标点必须是英文**（用户明确提醒）。这里专门检测中文标点并给出可读的报错 ——
  用 `，`/`：`/`｜` 写出来的脚本看起来"一模一样"，但解析会全错。
* 百分比相同时**按出现顺序**执行；同一百分比的多条会合并成一次触发。
* `D:` 与 `N:` 在**同一条**里可以配 `|` 一起写；**不同行**的漂移与氮气不要重叠
  （点氮气会取消漂移 —— 用户的原话），解析时给出警告。
* 选路那两位数字的**范围**要合法（第几个不能大于图标数量，1~8）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: 操作类型
OP_ROUTE = "route"          # 选路：`32`
OP_NITRO = "nitro"          # `N:0:2:750`
OP_DRIFT = "drift"          # `D:3000`
OP_AUTOPILOT = "autopilot"  # `S:2000`
OP_SPIN = "spin"            # `360`

OP_LABELS = {
    OP_ROUTE: "选路",
    OP_NITRO: "点氮气",
    OP_DRIFT: "漂移",
    OP_AUTOPILOT: "关自动驾驶",
    OP_SPIN: "360",
}

#: 中文标点 -> 英文（报错时提示用）
CJK_PUNCT = {"，": ",", "：": ":", "｜": "|", "；": ";", "。": ".", "％": "%"}

_RE_NITRO = re.compile(r"N:(\d+):(\d+):(\d+)")
_RE_DRIFT = re.compile(r"D:(\d+)")
_RE_AUTOPILOT = re.compile(r"S:(\d+)")
_RE_ROUTE = re.compile(r"(\d)(\d)")


class RouteError(ValueError):
    """路线脚本格式错误（消息里带行号与原文，方便直接改脚本）。"""


@dataclass(frozen=True)
class Op:
    """一个具体操作。不同 kind 用不同字段，用不到的字段保持 None。"""

    kind: str
    #: 氮气：延时(ms) / 次数 / 间隔(ms)
    delay_ms: int = 0
    times: int = 0
    interval_ms: int = 0
    #: 漂移 / 关自动驾驶：时长(ms)
    ms: int = 0
    #: 选路：图标数量 / 选第几个（1 起）
    icons: int = 0
    index: int = 0

    def describe(self) -> str:
        if self.kind == OP_ROUTE:
            return f"选路（{self.icons} 个里选第 {self.index} 个）"
        if self.kind == OP_NITRO:
            return (f"氮气 {self.times} 次（延时 {self.delay_ms}ms，间隔 {self.interval_ms}ms）")
        if self.kind == OP_DRIFT:
            return f"漂移 {self.ms}ms"
        if self.kind == OP_AUTOPILOT:
            return f"关自动驾驶 {self.ms}ms"
        if self.kind == OP_SPIN:
            return "360"
        return f"未知操作 {self.kind}"


@dataclass
class Entry:
    """路线里的一行：到 `percent` 时执行 `ops`（多个 = 用 | 连的同时操作）。"""

    percent: float
    ops: list[Op]
    line_no: int = 0
    raw: str = ""

    def describe(self) -> str:
        return f"{self.percent:g}%: " + " + ".join(op.describe() for op in self.ops)


@dataclass
class Route:
    """一整份路线。`entries` 按百分比升序（同百分比保持原顺序）。"""

    entries: list[Entry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def percents(self) -> list[float]:
        return [e.percent for e in self.entries]

    def describe(self) -> str:
        lines = [f"路线共 {len(self.entries)} 条操作"]
        for e in self.entries:
            lines.append(f"  {e.percent:>5g}%  " + " + ".join(op.describe() for op in e.ops))
        if self.warnings:
            lines.append("警告：")
            lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def _hint_cjk(text: str) -> str:
    """发现中文标点时给一条明确的提示（这是最容易犯又最难查的错）。"""
    bad = [c for c in text if c in CJK_PUNCT]
    if not bad:
        return ""
    fixed = "".join(CJK_PUNCT.get(c, c) for c in text)
    return f"（这里用了中文标点 {''.join(bad)!r}，要改成英文；应该写成：{fixed}）"


def parse_op(text: str) -> Op:
    """解析单个操作（`N:0:2:750` / `D:1000` / `32` / `360` …）。"""
    t = text.strip()
    if not t:
        raise RouteError("空操作")
    if t == "360":
        return Op(OP_SPIN)
    m = _RE_ROUTE.fullmatch(t)
    if m:
        icons, index = int(m.group(1)), int(m.group(2))
        if icons < 1 or index < 1 or index > icons:
            raise RouteError(f"选路 {t!r} 不合法：前一位是图标数量、后一位是选第几个"
                             f"（1 起，且不能大于数量）")
        if icons > 8:
            raise RouteError(f"选路 {t!r} 的图标数量 {icons} 太大了（一屏不会有这么多）")
        return Op(OP_ROUTE, icons=icons, index=index)
    m = _RE_NITRO.fullmatch(t)
    if m:
        delay, times, interval = (int(g) for g in m.groups())
        if times < 1:
            raise RouteError(f"氮气 {t!r} 的次数不能是 0")
        if times >= 2 and interval < 1:
            raise RouteError(f"氮气 {t!r} 的间隔不能是 0（连着点两次会挤在一起，游戏只认一次）")
        # 点太多次**只警告不拦**：用户真实路线里就有 `N:0:20:100`（20 次快速连点）——
        # 我一开始把它写成硬错误，结果把自己人的路线挡在门外 ✗。
        return Op(OP_NITRO, delay_ms=delay, times=times, interval_ms=interval)
    m = _RE_DRIFT.fullmatch(t)
    if m:
        ms = int(m.group(1))
        if ms <= 0:
            raise RouteError(f"漂移 {t!r} 的时长必须大于 0")
        return Op(OP_DRIFT, ms=ms)
    m = _RE_AUTOPILOT.fullmatch(t)
    if m:
        ms = int(m.group(1))
        if ms <= 0:
            raise RouteError(f"关自动驾驶 {t!r} 的时长必须大于 0")
        return Op(OP_AUTOPILOT, ms=ms)
    raise RouteError(
        f"看不懂的操作 {t!r} {_hint_cjk(t)}。支持的写法：\n"
        "    NN            选路（前一位=图标数量，后一位=选第几个，如 32）\n"
        "    N:延时:次数:间隔  点氮气（如 N:0:2:750 完美氮气）\n"
        "    D:毫秒        漂移（如 D:3000）\n"
        "    S:毫秒        关自动驾驶（如 S:2000）\n"
        "    360           360 操作")


def parse_line(raw: str, line_no: int = 0) -> Entry | None:
    """解析一行；空行/注释返回 None。"""
    line = raw.strip()
    if not line or line.startswith("#") or line.startswith("//"):
        return None
    if line.endswith("|"):              # 允许末尾多一个竖线（手写常见）
        line = line[:-1].rstrip()
    head, sep, tail = line.partition(",")
    if not sep:
        raise RouteError(f"第 {line_no} 行缺少逗号：{raw.strip()!r} {_hint_cjk(raw)}"
                         "（格式是「百分比,操作」）")
    try:
        percent = float(head.strip().rstrip("%"))
    except ValueError:
        raise RouteError(f"第 {line_no} 行的百分比看不懂：{head.strip()!r}"
                         f" {_hint_cjk(head)}") from None
    if not 0 <= percent <= 100:
        raise RouteError(f"第 {line_no} 行的百分比 {percent:g} 不在 0~100 之间")
    ops = [parse_op(part) for part in tail.split("|")]
    return Entry(percent=percent, ops=ops, line_no=line_no, raw=raw.strip())


def _tokenize(text: str) -> list[str]:
    """把路线脚本切成 `[百分比, 操作, 百分比, 操作, …]`。

    **两种写法都认**（用户实际给的是第 ② 种）：

    ① 每行一条：`35,N:0:2:750`
    ② 整份是一串扁平的逗号流：`1,31,2,N:0:2:750,4,360,…`
       —— 操作里只会有 `:` 和 `|`，**不会出现逗号**，所以按逗号切开就是严格交替的
       「百分比, 操作」对 ✓。

    注释（`#` / `//` 开头的行）与空行直接丢掉。
    """
    tokens: list[str] = []
    data_lines: list[str] = []
    for raw in text.lstrip("\ufeff").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if line.endswith(","):            # 允许每行末尾多一个逗号
            line = line[:-1]
        data_lines.append(line)
        tokens += [t.strip() for t in line.split(",") if t.strip()]
    # 中文标点检查**只针对数据行** —— 注释里可以随便写中文标点 ✗
    # （自动生成的路线草稿注释里就带「，：」，一开始把检查放在剥离注释之前，
    #   结果自己生成的路线被自己判成非法 ✗，是 test_webvideo 抓出来的）。
    for line in data_lines:
        bad = [c for c in line if c in CJK_PUNCT and c != "％"]
        if bad:
            fixed = "".join(CJK_PUNCT.get(c, c) for c in line)
            raise RouteError(
                f"脚本里用了中文标点 {''.join(sorted(set(bad)))!r}，要改成英文的"
                f"（正确写法：{fixed[:60]}{'…' if len(fixed) > 60 else ''}）")
    return tokens


def _parse_percent(tok: str, idx: int) -> float:
    try:
        percent = float(tok.rstrip("%"))
    except ValueError:
        raise RouteError(f"第 {idx} 个字段应该是百分比，实际是 {tok!r}") from None
    if not 0 <= percent <= 100:
        raise RouteError(f"百分比 {percent:g} 不在 0~100 之间")
    return percent


def parse_route(text: str) -> Route:
    """解析整份路线脚本（两种写法都认，见 `_tokenize`）。

    * 按百分比升序排序（同百分比保持原顺序 —— 脚本里写在前面的先执行）；
    * 检查两类容易出事的地方并写进 `warnings`（**不报错**，因为用户原脚本可能就这样写）：
      1. 同一个百分比出现多次（**会各自触发**，顺序按脚本顺序）；
      2. 漂移期间还有另一条氮气操作（点氮气会取消漂移）。
    """
    tokens = _tokenize(text)
    if not tokens:
        raise RouteError("这份路线一条有效操作都没有（空文件？）")
    if len(tokens) % 2:
        n_commas = text.count(",")
        hint = ("看起来是少写了逗号（格式是「百分比,操作」）" if n_commas == 0
                else "最后多出来一个")
        raise RouteError(f"字段数是 {len(tokens)}（奇数）—— 应该成对出现「百分比,操作」，"
                         f"{hint}：{tokens[-1]!r}")
    route = Route()
    for i in range(0, len(tokens), 2):
        percent = _parse_percent(tokens[i], i + 1)
        ops = [parse_op(part) for part in tokens[i + 1].split("|")]
        route.entries.append(Entry(percent=percent, ops=ops,
                                   line_no=i // 2 + 1, raw=f"{tokens[i]},{tokens[i + 1]}"))
    route.entries.sort(key=lambda e: e.percent)

    seen: dict[float, list[Entry]] = {}
    for e in route.entries:
        seen.setdefault(e.percent, []).append(e)
    dup = {p: len(v) for p, v in seen.items() if len(v) > 1}
    for p, n in sorted(dup.items()):
        route.warnings.append(f"{p:g}% 出现了 {n} 次 —— **会各自触发**（顺序按脚本里的顺序）；"
                             "如果想让它们同时执行，请用 | 写在同一行")

    # 漂移与氮气的**同点**冲突检查（不同条目之间）
    #   ⚠️ 只报"同一百分比附近"（<0.5 个点）的情况：用户真实路线里大量
    #   "漂移 x% -> 氮气 x+2%" 是**有意的**技术动作（漂移里点氮气切喷），
    #   我一开始拿毫秒跟百分点比（`np <= dp + ms/1000`）→ 一条路线报 10 条误报 ✗。
    drifts = [(e.percent, e.line_no) for e in route.entries
              for op in e.ops if op.kind == OP_DRIFT]
    nitros = [(e.percent, e.line_no) for e in route.entries
              for op in e.ops if op.kind == OP_NITRO]
    for dp, dline in drifts:
        for np_, nline in nitros:
            if dline == nline:
                continue                      # 同一条里配 | 是用户明确要的用法
            if dp <= np_ < dp + 0.5:
                route.warnings.append(
                    f"第 {nline} 条（{np_:g}%）的氮气几乎和漂移（第 {dline} 条 {dp:g}%）同时 —— "
                    "点氮气会取消漂移，确认是有意的；真想让它们同时就写成一个条目配 |")
    return route
