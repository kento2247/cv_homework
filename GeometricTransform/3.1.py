"""
AKAZE + BF-Hamming で 2 枚の画像をマッチングし，
平行移動・類似・アフィン変換でステッチした結果を比較するスクリプト
"""

import cv2
import numpy as np
from pathlib import Path


# ---------- 特徴点抽出とマッチング ----------


def extract_features(img_bgr):
    """AKAZE で特徴点検出・記述子抽出（グレースケールで）"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create()
    return akaze.detectAndCompute(gray, None)  # (keypoints, descriptors)


def match_features(des_a, des_b, ratio=0.75):
    """BF-Hamming + Lowe ratio の良質マッチを返す"""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return good


def get_pts(kp_a, kp_b, matches):
    src = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src, dst


# ---------- 幾何変換推定 ----------


def estimate_translation(src, dst):
    """LS 最小二乗で平行移動のみ推定（類似変換行列を平行移動に丸める）"""
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None
    # スケール=1, 回転=0 に固定
    tx, ty = M[0, 2], M[1, 2]
    return np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)


def estimate_similarity(src, dst):
    M, _ = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000
    )
    return M.astype(np.float32) if M is not None else None


def estimate_affine(src, dst):
    M, _ = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000
    )
    return M.astype(np.float32) if M is not None else None


# ---------- キャンバス計算とブレンド ----------


def canvas_and_shift(img_a, img_b, M):
    """img_a を M で変換したときに両画像が収まるキャンバス幅高さとシフト行列 T を返す"""
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)
    corners_a_tf = cv2.transform(corners_a, M)
    corners_b = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)

    all_pts = np.vstack((corners_a_tf, corners_b)).reshape(-1, 2)
    min_xy = np.floor(all_pts.min(axis=0)).astype(int)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(int)

    W = int(max_xy[0] - min_xy[0])
    H = int(max_xy[1] - min_xy[1])

    T = np.array([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]], dtype=np.float32)
    return (W, H), T


def warp_and_blend(img_a, img_b, M, out_path):
    """M 変換で img_a をワープして img_b と α=0.5 合成，保存"""
    (W, H), T = canvas_and_shift(img_a, img_b, M)
    M_shift = (T @ np.vstack([M, [0, 0, 1]]))[:2]

    warped_a = cv2.warpAffine(img_a, M_shift, (W, H))
    shifted_b = cv2.warpAffine(img_b, T[:2], (W, H))

    blended = cv2.addWeighted(warped_a, 0.5, shifted_b, 0.5, 0)
    cv2.imwrite(str(out_path), blended)


# ---------- メイン ----------


def main(imgA_path="images/a.jpg", imgB_path="images/b.jpg", outdir="images/3.1"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    img_a = cv2.imread(imgA_path)
    img_b = cv2.imread(imgB_path)

    kp_a, des_a = extract_features(img_a)
    kp_b, des_b = extract_features(img_b)

    matches = match_features(des_a, des_b)
    if len(matches) < 3:
        raise RuntimeError(f"Good matches < 3 ({len(matches)}); 変換推定不可")

    # マッチ可視化
    vis = cv2.drawMatches(
        img_a,
        kp_a,
        img_b,
        kp_b,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out / "matches.jpg"), vis)

    src, dst = get_pts(kp_a, kp_b, matches)

    # 変換行列推定
    M_trans = estimate_translation(src, dst)
    M_simi = estimate_similarity(src, dst)
    M_aff = estimate_affine(src, dst)

    # ステッチ＆保存（推定に失敗した場合はスキップ）
    if M_trans is not None:
        warp_and_blend(img_a, img_b, M_trans, out / "translation.jpg")
    if M_simi is not None:
        warp_and_blend(img_a, img_b, M_simi, out / "similarity.jpg")
    if M_aff is not None:
        warp_and_blend(img_a, img_b, M_aff, out / "affine.jpg")


if __name__ == "__main__":
    main()
