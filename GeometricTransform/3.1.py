"""
images/a.jpg
images/b.jpg
を対象にAKAZEを用いた際のステッチングを実施し，平行移動，類似，アフィン変換の結果を比較する
"""

import cv2
import numpy as np


def extract_features(img):
    """画像からAKAZE特徴量を抽出する"""
    akaze = cv2.AKAZE_create()
    return akaze.detectAndCompute(img, None)


def match_features(des_a, des_b):
    """特徴量のマッチングを行う"""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, des_b, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches


def visualize_matches(img_a, kp_a, img_b, kp_b, good_matches, output_path):
    """マッチング結果を可視化して保存する"""
    img_matches = cv2.drawMatches(
        img_a,
        kp_a,
        img_b,
        kp_b,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(output_path, img_matches)


def get_matching_points(kp_a, kp_b, good_matches):
    """マッチング点の座標を取得する"""
    src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts


def calculate_transformation_matrices(src_pts, dst_pts):
    """各種変換行列を計算する"""

    # 平行移動（純粋なシフトベクトル）
    translation_vector = np.mean(dst_pts - src_pts, axis=0).reshape(2)
    M_translation = np.array(
        [[1, 0, translation_vector[0]], [0, 1, translation_vector[1]]], dtype=np.float32
    )

    # 類似変換（スケール・回転・平行移動）
    M_similarity = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )[0]

    # アフィン変換（一般アフィン変換）
    M_affine = cv2.estimateAffine2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )[0]

    return M_translation, M_similarity, M_affine


def transform_and_blend_images(img_a, img_b, M_translation, M_similarity, M_affine):
    """画像の変換と合成を行う"""
    h, w = img_b.shape[:2]
    img_translation = cv2.warpAffine(img_a, M_translation, (w, h))
    img_similarity = cv2.warpAffine(img_a, M_similarity, (w, h))
    img_affine = cv2.warpAffine(img_a, M_affine, (w, h))

    return img_translation, img_similarity, img_affine


def save_results(img_translation, img_similarity, img_affine, img_b):
    """結果を保存する"""
    cv2.imwrite(
        "images/3.1/translation.jpg",
        cv2.addWeighted(img_translation, 0.5, img_b, 0.5, 0),
    )
    cv2.imwrite(
        "images/3.1/similarity.jpg", cv2.addWeighted(img_similarity, 0.5, img_b, 0.5, 0)
    )
    cv2.imwrite(
        "images/3.1/affine.jpg", cv2.addWeighted(img_affine, 0.5, img_b, 0.5, 0)
    )


def main():
    # 画像の読み込み
    img_a = cv2.imread("images/a.jpg")
    img_b = cv2.imread("images/b.jpg")

    # 特徴量の抽出
    kp_a, des_a = extract_features(img_a)
    kp_b, des_b = extract_features(img_b)

    # 特徴量のマッチング
    good_matches = match_features(des_a, des_b)

    # マッチング結果の可視化
    visualize_matches(img_a, kp_a, img_b, kp_b, good_matches, "images/3.1/matches.jpg")

    # マッチング点の座標を取得
    src_pts, dst_pts = get_matching_points(kp_a, kp_b, good_matches)

    # 変換行列の計算
    M_translation, M_similarity, M_affine = calculate_transformation_matrices(
        src_pts, dst_pts
    )

    # 画像の変換と合成
    img_translation, img_similarity, img_affine = transform_and_blend_images(
        img_a, img_b, M_translation, M_similarity, M_affine
    )

    # 結果の保存
    save_results(img_translation, img_similarity, img_affine, img_b)


if __name__ == "__main__":
    main()
