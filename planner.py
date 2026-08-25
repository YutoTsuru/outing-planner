"""お出かけプランナー：お出かけプランを組み立てる部分

予測されたカテゴリ（outdoor / indoor / relax）と、
場所（緯度経度）・時間（開始〜終了）から、
1日のタイムスケジュールを作ります。

流れ:
    1. 開始〜終了の時間を、滞在時間＋移動時間で区切る（build_schedule）
    2. それぞれの枠に合うスポットを取ってくる（collect_spots）
       スポットは Google Maps → OpenStreetMap の順に探し、
       どちらも使えないときは Googleマップの検索リンクにする
    3. 表（Markdown）の形にして返す（build_plan）
"""

import re

import maps_api
import osm_api

# ---------------------------------------------------------------
# 設定・固定データ
# ---------------------------------------------------------------

# 予定の種類ごとの、だいたいの滞在時間（分）
STAY_MINUTES = {
    "outdoor": 90,
    "indoor": 120,
    "relax": 60,
    "lunch": 60,
    "cafe": 45,
}

# 次のスポットまでの移動時間の目安（分）
TRAVEL_MINUTES = 20

# お昼ごはんを入れる時間帯（11:30〜13:30）
LUNCH_START_MINUTES = 11 * 60 + 30
LUNCH_END_MINUTES = 13 * 60 + 30

# 最後の予定に使える最短の時間（分）
# これ以上あまっていたら、短めの滞在でもう1か所まわる
MIN_LAST_STAY_MINUTES = 45

# 1日にまわる予定の最大数（多すぎると現実的でないので上限を決めておく）
MAX_ACTIVITIES = 6

# 1つの予定の種類あたり、いくつのキーワードで検索するか
# （検索1回に数秒かかるので、増やしすぎると待ち時間が長くなる）
MAX_KEYWORDS_PER_KIND = 2

# 予定の種類ごとの表示名
KIND_NAMES = {
    "outdoor": "🌳 屋外観光",
    "indoor": "🏛️ 屋内観光",
    "relax": "☕ リラックス",
    "lunch": "🍽️ ランチ",
    "cafe": "🍰 カフェ休憩",
    "wish": "⭐ 行きたい場所",
}

# 「行きたい場所」をいくつまで受け付けるか
MAX_WISHES = 4

# 予定の種類ごとの、Google Maps 検索に使うキーワード
KIND_KEYWORDS = {
    "outdoor": ["公園", "神社", "展望台", "庭園"],
    "indoor": ["水族館", "美術館", "博物館", "映画館"],
    "relax": ["カフェ", "日帰り温泉", "図書館", "スパ"],
    "lunch": ["レストラン", "ラーメン", "そば"],
    "cafe": ["カフェ"],
}


# ---------------------------------------------------------------
# 時刻の計算（"10:00" ⇔ 分）
# ---------------------------------------------------------------

def time_to_minutes(text):
    """"10:30" のような文字列を、0時からの分数に変換する。"""
    hour, minute = text.split(":")
    return int(hour) * 60 + int(minute)


def minutes_to_time(minutes):
    """分数を "10:30" のような文字列に変換する。"""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def time_choices(start_hour=5, end_hour=23):
    """時刻の選択肢（30分きざみ）のリストを作る。"""
    return [
        f"{hour:02d}:{minute:02d}"
        for hour in range(start_hour, end_hour + 1)
        for minute in (0, 30)
    ]


def parse_wishes(text, limit=MAX_WISHES):
    """「行きたい場所」の入力を、キーワードのリストに分ける。

    「はま寿司、水族館」のように、読点・カンマ・スラッシュ・改行で区切る。
    （店名にスペースが入ることがあるので、スペースでは区切らない）
    """
    if not text:
        return []

    parts = re.split(r"[、,，/／\n]+", text.strip())
    return [part.strip() for part in parts if part.strip()][:limit]


# ---------------------------------------------------------------
# 1. 時間割を組み立てる
# ---------------------------------------------------------------

def build_schedule(label, start_minutes, end_minutes):
    """開始〜終了の時間を区切って、予定の並び（時間割）を作る。

    戻り値: (開始分, 終了分, 予定の種類) のタプルのリスト
    """
    schedule = []
    cursor = start_minutes
    main_count = 0        # メインのスポットを何か所まわったか
    lunch_added = False   # お昼ごはんを入れたか
    cafe_added = False    # カフェ休憩を入れたか

    while cursor < end_minutes and len(schedule) < MAX_ACTIVITIES:
        # お昼の時間帯にさしかかったら、まずランチを入れる
        if not lunch_added and LUNCH_START_MINUTES <= cursor <= LUNCH_END_MINUTES:
            kind = "lunch"
        # メインを2か所まわったら、いったんカフェで休憩する
        elif main_count >= 2 and not cafe_added and label != "relax":
            kind = "cafe"
        else:
            kind = label

        stay = STAY_MINUTES[kind]
        if cursor + stay > end_minutes:
            # 時間がぴったり足りないときは、残り時間で短めに立ち寄る
            if end_minutes - cursor >= MIN_LAST_STAY_MINUTES:
                schedule.append((cursor, end_minutes, kind))
            break

        schedule.append((cursor, cursor + stay, kind))

        if kind == "lunch":
            lunch_added = True
        elif kind == "cafe":
            cafe_added = True
        else:
            main_count += 1

        cursor += stay + TRAVEL_MINUTES

    return schedule


# ---------------------------------------------------------------
# 2. スポットを集める
# ---------------------------------------------------------------

def search_by_keyword(keyword, latitude, longitude, radius_m, limit=5):
    """1つのキーワードで、実在するスポットを探す。

    Google Maps が使えればそちらを、だめなら OpenStreetMap を使う。
    戻り値: (スポットのリスト, お知らせのリスト)
    """
    errors = []

    # 1. Google Maps（APIキーがあり、まだ拒否されていないとき）
    if maps_api.has_api_key() and maps_api.denied_reason() is None:
        try:
            spots = maps_api.search_places(keyword, latitude, longitude, radius_m, limit=limit)
            if spots:
                return spots, errors
        except maps_api.MapsError as error:
            errors.append(str(error))

    # 2. OpenStreetMap（キーも料金もいらない）
    try:
        return osm_api.search_spots(keyword, latitude, longitude, radius_m, limit=limit), errors
    except osm_api.OsmError as error:
        errors.append(str(error))

    return [], errors


def collect_spots(kind, latitude, longitude, area_name, radius_m, needed):
    """1つの予定の種類について、候補スポットを必要な数だけ集める。"""
    spots = []
    errors = []

    if latitude is not None:
        # キーワードを順に試す（1つで足りたら、そこでやめる）
        # 1回の検索に数秒かかるので、試すのは最大2つまでにする
        for keyword in KIND_KEYWORDS[kind][:MAX_KEYWORDS_PER_KIND]:
            if len(spots) >= needed:
                break
            found, keyword_errors = search_by_keyword(
                keyword, latitude, longitude, radius_m, limit=needed + 2
            )
            spots.extend(found)
            errors.extend(keyword_errors)

    # 何も見つからなかったときは、Googleマップの検索リンクで補う
    for keyword in KIND_KEYWORDS[kind]:
        if len(spots) >= needed:
            break
        spots.append(maps_api.fallback_spot(keyword, area_name, latitude, longitude))

    return spots, errors


def find_wish_spot(keyword, latitude, longitude, area_name, radius_m):
    """「行きたい場所」の名前から、実際のお店・施設を1件さがす。"""
    errors = []

    if latitude is not None:
        # 行きたい場所は少し広め（最低5km）に探す
        spots, errors = search_by_keyword(
            keyword, latitude, longitude, max(radius_m, 5000), limit=3
        )
        if spots:
            return spots[0], errors

    # 見つからなかったときは、Googleマップの検索リンクにしておく
    return maps_api.fallback_spot(keyword, area_name, latitude, longitude), errors


def assign_spots(schedule, latitude, longitude, area_name, radius_m, wishes=()):
    """時間割の各枠に、スポットを1つずつ割り当てる。

    「行きたい場所」があれば、前の枠から順に優先して入れる。
    """
    wish_queue = list(wishes)
    errors = []

    # まず、どの枠を「行きたい場所」で埋めるかを決める
    slots = []
    for start, end, kind in schedule:
        if wish_queue:
            slots.append((start, end, "wish", wish_queue.pop(0)))
        else:
            slots.append((start, end, kind, None))

    # 残った枠の分だけ、カテゴリごとの候補スポットをまとめて取得する
    needed_counts = {}
    for _, _, kind, wish in slots:
        if wish is None:
            needed_counts[kind] = needed_counts.get(kind, 0) + 1

    candidates = {}
    for kind, count in needed_counts.items():
        spots, kind_errors = collect_spots(
            kind, latitude, longitude, area_name, radius_m, needed=count + 1
        )
        candidates[kind] = spots
        errors.extend(kind_errors)

    # 前から順に、まだ使っていないスポットを割り当てる
    used_names = set()
    plan_rows = []
    for start, end, kind, wish in slots:
        if wish is not None:
            chosen, wish_errors = find_wish_spot(
                wish, latitude, longitude, area_name, radius_m
            )
            errors.extend(wish_errors)
        else:
            chosen = None
            for spot in candidates[kind]:
                if spot["name"] not in used_names:
                    chosen = spot
                    break
            if chosen is None:
                chosen = maps_api.fallback_spot(
                    KIND_KEYWORDS[kind][0], area_name, latitude, longitude
                )

        used_names.add(chosen["name"])
        plan_rows.append((start, end, kind, chosen))

    # 時間内に入りきらなかった「行きたい場所」を返す
    return plan_rows, errors, wish_queue


# ---------------------------------------------------------------
# 3. プランを文章（Markdown）にする
# ---------------------------------------------------------------

def format_spot(spot):
    """スポット1件を、リンク付きの1行の文字列にする。"""
    text = f"[{spot['name']}]({spot['url']})"

    if spot.get("rating"):
        text += f" ★{spot['rating']}（{spot['rating_count']}件）"

    distance = spot.get("distance_m")
    if distance:
        text += f" 約{distance / 1000:.1f}km" if distance >= 1000 else f" 約{distance}m"

    return text


def build_plan(label, latitude, longitude, area_name, start_time, end_time,
               radius_m=3000, note=None, wishes=()):
    """お出かけプラン全体を作って、Markdownの文字列で返す。

    note には、プランの下に出しておきたいお知らせ（APIが使えなかった等）を渡す。
    wishes には「行きたい場所」のキーワードのリストを渡す。
    """
    start_minutes = time_to_minutes(start_time)
    end_minutes = time_to_minutes(end_time)

    if end_minutes <= start_minutes:
        return "⚠️ 終了時刻は、開始時刻よりあとにしてください。"

    schedule = build_schedule(label, start_minutes, end_minutes)
    if not schedule:
        return (
            f"⚠️ 時間が短すぎてプランを作れませんでした。"
            f"（{KIND_NAMES[label]}には最低でも{STAY_MINUTES[label]}分ほど必要です）"
        )

    plan_rows, errors, left_wishes = assign_spots(
        schedule, latitude, longitude, area_name, radius_m, wishes
    )

    lines = [
        f"### 🗺️ {area_name} のお出かけプラン",
        f"**{start_time} 〜 {end_time}** ／ おすすめカテゴリ：{KIND_NAMES[label]}",
        "",
        "| 時間 | 予定 | 行き先 |",
        "| --- | --- | --- |",
    ]

    for start, end, kind, spot in plan_rows:
        time_range = f"{minutes_to_time(start)}〜{minutes_to_time(end)}"
        lines.append(f"| {time_range} | {KIND_NAMES[kind]} | {format_spot(spot)} |")

    last_end = plan_rows[-1][1]
    lines.append("")
    lines.append(
        f"※ 移動時間は各スポット間 約{TRAVEL_MINUTES}分で計算しています"
        f"（解散の目安：{minutes_to_time(last_end)}）。"
    )

    sources = {spot.get("source") for _, _, _, spot in plan_rows}
    if "osm" in sources:
        lines.append("※ スポット情報：© OpenStreetMap contributors")
    if "link" in sources:
        lines.append(
            "※ 見つからなかった行き先は、Googleマップの検索リンクにしています。"
        )
    if left_wishes:
        lines.append(
            "※ 時間内に入りきらなかった行きたい場所："
            + "、".join(left_wishes)
            + "（終了時刻をうしろにすると入ります）"
        )
    if note:
        lines.append(f"※ {note}")

    # 同じ内容のお知らせは1回だけ出す
    for message in dict.fromkeys(errors):
        lines.append(f"※ {message}")

    return "\n".join(lines)
