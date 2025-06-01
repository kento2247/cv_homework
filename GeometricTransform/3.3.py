"""
images/a.jpg, images/b.jpg, images/c.jpg を対象にAKAZEを用いた際のステッチングを実施し，
アフィン変換の結果を比較する
"""

import cv2
import numpy as np
from pathlib import Path


def extract_features(img_bgr):
    """AKAZE で特徴点検出・記述子抽出（グレースケールで）"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create()
    return akaze.detectAndCompute(gray, None)


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


def estimate_affine(src, dst):
    """アフィン変換行列を推定"""
    M, _ = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000
    )
    return M.astype(np.float32) if M is not None else None


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


def warp_and_blend(img_a, img_b, M):
    """M 変換で img_a をワープして img_b と α=0.5 合成"""
    (W, H), T = canvas_and_shift(img_a, img_b, M)
    M_shift = (T @ np.vstack([M, [0, 0, 1]]))[:2]

    warped_a = cv2.warpAffine(img_a, M_shift, (W, H))
    shifted_b = cv2.warpAffine(img_b, T[:2], (W, H))

    return cv2.addWeighted(warped_a, 0.5, shifted_b, 0.5, 0)


def stitch_two_images(img_a, img_b):
    """2枚の画像をステッチングする"""
    kp_a, des_a = extract_features(img_a)
    kp_b, des_b = extract_features(img_b)

    matches = match_features(des_a, des_b)
    if len(matches) < 3:
        raise RuntimeError(f"Good matches < 3 ({len(matches)}); 変換推定不可")

    src, dst = get_pts(kp_a, kp_b, matches)
    M = estimate_affine(src, dst)
    if M is None:
        raise RuntimeError("アフィン変換行列の推定に失敗")

    return warp_and_blend(img_a, img_b, M)


def stitch_three_images(img_a, img_b, img_c):
    """3枚の画像を一気にステッチングする（単純な平均合成）"""
    kp_a, des_a = extract_features(img_a)
    kp_b, des_b = extract_features(img_b)
    kp_c, des_c = extract_features(img_c)

    # a-b間のマッチング
    matches_ab = match_features(des_a, des_b)
    src_ab, dst_ab = get_pts(kp_a, kp_b, matches_ab)
    M_ab = estimate_affine(src_ab, dst_ab)

    # b-c間のマッチング
    matches_bc = match_features(des_b, des_c)
    src_bc, dst_bc = get_pts(kp_b, kp_c, matches_bc)
    M_bc = estimate_affine(src_bc, dst_bc)

    if M_ab is None or M_bc is None:
        raise RuntimeError("アフィン変換行列の推定に失敗")

    # a, b, c それぞれの変換後の四隅を計算
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    h_c, w_c = img_c.shape[:2]

    corners_a = np.float32([[0,0], [w_a,0], [w_a,h_a], [0,h_a]]).reshape(-1,1,2)
    corners_b = np.float32([[0,0], [w_b,0], [w_b,h_b], [0,h_b]]).reshape(-1,1,2)
    corners_c = np.float32([[0,0], [w_c,0], [w_c,h_c], [0,h_c]]).reshape(-1,1,2)

    corners_a_tf = cv2.transform(corners_a, M_ab)
    corners_c_tf = cv2.transform(corners_c, M_bc)

    all_pts = np.vstack((corners_a_tf, corners_b, corners_c_tf)).reshape(-1,2)
    min_xy = np.floor(all_pts.min(axis=0)).astype(int)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(int)

    W = int(max_xy[0] - min_xy[0])
    H = int(max_xy[1] - min_xy[1])

    T = np.array([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]], dtype=np.float32)

    # 各画像をシフト
    M_a_shift = (T @ np.vstack([M_ab, [0,0,1]]))[:2]
    M_b_shift = T[:2]
    M_c_shift = (T @ np.vstack([M_bc, [0,0,1]]))[:2]

    # ワープ
    warped_a = cv2.warpAffine(img_a, M_a_shift, (W, H))
    warped_b = cv2.warpAffine(img_b, M_b_shift, (W, H))
    warped_c = cv2.warpAffine(img_c, M_c_shift, (W, H))

    # 単純平均合成
    result = (warped_a.astype(np.float32) + warped_b.astype(np.float32) + warped_c.astype(np.float32)) / 3
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def main():
    out = Path("images/3.3")
    out.mkdir(parents=True, exist_ok=True)

    # 画像の読み込み
    img_a = cv2.imread("images/a.jpg")
    img_b = cv2.imread("images/b.jpg")
    img_c = cv2.imread("images/c.jpg")

    # 2枚ずつのステッチング
    ab_result = stitch_two_images(img_a, img_b)
    abc_result = stitch_two_images(ab_result, img_c)
    cv2.imwrite(str(out / "a-b-c.jpg"), abc_result)

    # 3枚同時のステッチング
    abc_result = stitch_three_images(img_a, img_b, img_c)
    cv2.imwrite(str(out / "abc.jpg"), abc_result)


if __name__ == "__main__":
    main()
