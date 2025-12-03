import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import networkx as nx


# ----- ユーティリティ関数 ----- #
def generate_gps_path(gps_dir, start_time, end_time, output_dir=""):
    """
    GPSファイルのパスを生成する関数。

    Parameters:
        gps_dir (str): GPSファイルが保存されているディレクトリのパス。
        start_time (str): 開始時刻（例: "14:40:00"）。
        end_time (str): 終了時刻（例: "14:50:00"）。

    Returns:
        str: 結果として得られるGPSファイルのパス。
        ex) "/gps_20250305_1440_1450.csv"
    """
    # ディレクトリ名から日付部分を抽出（数字のみを対象とする）
    date_str = "".join(filter(str.isdigit, gps_dir))
    if len(date_str) != 8:
        raise ValueError("gps_dir から正しい日付 (YYYYMMDD) が抽出できませんでした。")

    # 時分だけを抽出（例: "1440"）
    start_hm = start_time.replace(":", "")[:4]
    end_hm = end_time.replace(":", "")[:4]

    # ファイル名の組み立て
    filename = f"gps_{date_str}_{start_hm}_{end_hm}.csv"
    return os.path.join(output_dir, filename)


def parse_video_filename_to_gps_params(file_path, base_dir):
    """
    動画ファイル名から gps_dir, start_time, end_time を抽出する関数。

    Parameters:
        file_path (str): ファイルパス（例: "../2025-03-05 14-40-00~14-50-00.avi"）
        base_dir (str): BASE_DIR（例: "../"）

    Returns:
        tuple: (gps_dir: str, start_time: str, end_time: str)
    """
    filename = os.path.basename(file_path)
    name, _ = os.path.splitext(filename)

    try:
        date_part = name[: len("YYYY-MM-DD")]
        time_part = name[len("YYYY-MM-DD") + 1 :]
        start_raw, end_raw = time_part.split("~")

        # 日付を YYYYMMDD 形式に変換
        date_obj = datetime.strptime(date_part, "%Y-%m-%d")
        date_str = date_obj.strftime("%Y%m%d")

        # 時刻を HH:MM:SS に整形
        start_time = start_raw.replace("-", ":")
        end_time = end_raw.replace("-", ":")

        # GPSディレクトリパス
        gps_dir = os.path.join(base_dir, date_str + "/")

        return gps_dir, start_time, end_time
    except Exception as e:
        raise ValueError(f"ファイル名の形式が不正です: {file_path}") from e


def extract_start_end_datetime(p_movie: str):
    """
    動画ファイル名から開始・終了時刻を datetime オブジェクトで抽出する関数。

    Parameters:
        p_movie (str): 動画ファイルパス（例: "../2025-03-05 14-40-00~14-50-00.avi"）

    Returns:
        tuple: (start_time: datetime, end_time: datetime)
    """
    filename = os.path.basename(p_movie)
    name, _ = os.path.splitext(filename)

    try:
        date_str, time_range = name.split("_")
        start_str, end_str = time_range.split("~")

        # 日付を datetime にパース（時刻はあとで結合）
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

        # 各時刻文字列を datetime オブジェクトに変換
        start_time = datetime.strptime(
            f"{date_str} {start_str.replace('-', ':')}", "%Y-%m-%d %H:%M:%S"
        )
        end_time = datetime.strptime(
            f"{date_str} {end_str.replace('-', ':')}", "%Y-%m-%d %H:%M:%S"
        )

        return start_time, end_time
    except Exception as e:
        raise ValueError(f"ファイル名の形式が不正です: {p_movie}") from e


def get_start_end_datetime(p_camera):
    p_camera = os.path.basename(p_camera)
    _, camera_date, camera_start_time, camera_end_time = p_camera.replace(
        ".csv", ""
    ).split("_")
    start_datetime = pd.Timestamp(
        f"{camera_date} {camera_start_time[:2]}:{camera_start_time[2:]}:00"
    )
    end_datetime = pd.Timestamp(
        f"{camera_date} {camera_end_time[:2]}:{camera_end_time[2:]}:00"
    )
    return start_datetime, end_datetime


def frameid_to_timestamp(frame_id, fps, start_ts):
    return start_ts + pd.Timedelta(seconds=frame_id / fps)


def load_matching_gt(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normal = {int(k): v for k, v in data.get("normal", {}).items()}
    difficult = {int(k): v for k, v in data.get("difficult", {}).items()}

    return normal, difficult


def plot_matching_result(ground_truth_dict, prediction_dict, save_path=None):
    # --- 1. ノードの準備 ---
    all_keys = set(ground_truth_dict.keys()) | set(prediction_dict.keys())
    all_values = set(ground_truth_dict.values()) | set(prediction_dict.values())

    B = nx.Graph()

    # ノードをグラフに追加 (bipartite属性でグループ分け)
    for key_node in all_keys:
        B.add_node(key_node, bipartite=0)
    for value_node in all_values:
        B.add_node(value_node, bipartite=1)

    # --- 2. エッジの準備 ---
    gt_edges = set(ground_truth_dict.items())
    pred_edges = set(prediction_dict.items())

    agreement_edges = gt_edges & pred_edges
    gt_only_edges = gt_edges - pred_edges
    pred_only_edges = pred_edges - gt_edges

    all_graph_edges = list(agreement_edges | gt_only_edges | pred_only_edges)
    B.add_edges_from(all_graph_edges)

    # --- 3. グラフのレイアウト ---
    pos = nx.bipartite_layout(B, all_keys, align="vertical", scale=2)
    top_nodes = all_keys
    bottom_nodes = all_values

    # --- 4. グラフの描画 ---
    plt.figure(figsize=(14, 10))

    # ノードの描画
    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=list(top_nodes),
        node_color="skyblue",
        node_size=3000,
        label="キー",
    )
    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=list(bottom_nodes),
        node_color="lightgreen",
        node_size=3000,
        label="値",
    )

    # ラベルの描画
    nx.draw_networkx_labels(B, pos, font_size=10, font_family="sans-serif")

    # エッジの描画
    # edgelistが空の場合、draw_networkx_edgesは何も描画しないのでエラーにはならない
    nx.draw_networkx_edges(
        B,
        pos,
        edgelist=list(agreement_edges),
        edge_color="green",
        width=2.5,
        label="Agreement",
    )
    nx.draw_networkx_edges(
        B,
        pos,
        edgelist=list(gt_only_edges),
        edge_color="blue",
        width=2,
        style="--",
        label="Ground Truth Only",
    )
    nx.draw_networkx_edges(
        B,
        pos,
        edgelist=list(pred_only_edges),
        edge_color="red",
        width=2,
        style=":",
        label="Prediction Only",
    )

    # --- 5. 凡例の作成 ---
    green_line = mlines.Line2D(
        [], [], color="green", linestyle="-", linewidth=2.5, label="Agreement"
    )
    blue_line = mlines.Line2D(
        [], [], color="blue", linestyle="--", linewidth=2, label="Ground Truth Only"
    )
    red_line = mlines.Line2D(
        [], [], color="red", linestyle=":", linewidth=2, label="Prediction Only"
    )

    plt.legend(handles=[green_line, blue_line, red_line], loc="best", fontsize=12)
    plt.title("Graph of Ground Truth vs Prediction", fontsize=16)
    plt.axis("off")  # 軸を非表示
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="png", dpi=300)
        plt.close()
    else:
        plt.show()


def _imscatter(x, y, images, colors, linestyles, ax=None, zoom=0.2, linewidth=3):
    """
    指定された座標(x, y)に、色とスタイルが指定された枠を持つ画像をプロットする。
    引数としてファイルパスではなく、画像データ（NumPy配列）を直接受け取るように修正した。
    """
    if ax is None:
        ax = plt.gca()

    artists = []
    # 第3引数はもはやパスではないため、変数名を 'image_data' に変更
    for x0, y0, image_data, color, lstyle in zip(x, y, images, colors, linestyles):
        try:
            # 不要なファイル読み込み処理を削除し、渡された画像データを直接使用する
            im = OffsetImage(image_data, zoom=zoom)

            ab = AnnotationBbox(
                im,
                (x0, y0),
                xycoords="data",
                frameon=True,
                bboxprops=dict(
                    edgecolor=color,
                    lw=linewidth,
                    boxstyle="square,pad=0.0",
                    linestyle=lstyle,  # 枠線のスタイルを指定
                ),
            )
            artists.append(ax.add_artist(ab))
        except Exception as e:
            # エラーメッセージを現状に合わせて修正
            print(f"Warning: Could not plot image data. Reason: {e}")

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists
