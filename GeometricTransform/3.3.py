"""
images/a.jpg
images/b.jpg
images/c.jpg
を対象にAKAZEを用いた際のステッチングを実施し，アフィン変換の結果を比較する
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


def calculate_affine_matrix(src_pts, dst_pts):
    """アフィン変換行列を計算する"""
    return cv2.estimateAffine2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )[0]


def transform_and_blend_images(img_a, img_b, M_affine):
    """画像の変換と合成を行う"""
    h, w = img_b.shape[:2]
    img_affine = cv2.warpAffine(img_a, M_affine, (w, h))
    return cv2.addWeighted(img_affine, 0.5, img_b, 0.5, 0)


def stitch_two_images(img_a, img_b):
    """2枚の画像をステッチングする"""
    # 特徴量の抽出
    kp_a, des_a = extract_features(img_a)
    kp_b, des_b = extract_features(img_b)

    # 特徴量のマッチング
    good_matches = match_features(des_a, des_b)

    # マッチング点の座標を取得
    src_pts, dst_pts = get_matching_points(kp_a, kp_b, good_matches)

    # アフィン変換行列の計算
    M_affine = calculate_affine_matrix(src_pts, dst_pts)

    # 画像の変換と合成
    result = transform_and_blend_images(img_a, img_b, M_affine)
    return result


def stitch_three_images(img_a, img_b, img_c):
    """3枚の画像をステッチングする"""
    # 画像aとbのステッチング
    kp_a, des_a = extract_features(img_a)
    kp_b, des_b = extract_features(img_b)
    good_matches_ab = match_features(des_a, des_b)
    src_pts_ab, dst_pts_ab = get_matching_points(kp_a, kp_b, good_matches_ab)
    M_ab = calculate_affine_matrix(src_pts_ab, dst_pts_ab)

    # 画像bとcのステッチング
    kp_c, des_c = extract_features(img_c)
    good_matches_bc = match_features(des_b, des_c)
    src_pts_bc, dst_pts_bc = get_matching_points(kp_b, kp_c, good_matches_bc)
    M_bc = calculate_affine_matrix(src_pts_bc, dst_pts_bc)

    # 画像aをbの座標系に変換
    h_b, w_b = img_b.shape[:2]
    img_a_transformed = cv2.warpAffine(img_a, M_ab, (w_b, h_b))

    # 画像cをbの座標系に変換
    img_c_transformed = cv2.warpAffine(img_c, M_bc, (w_b, h_b))

    # 3枚の画像を合成
    result = cv2.addWeighted(img_a_transformed, 0.33, img_b, 0.33, 0)
    result = cv2.addWeighted(result, 1.0, img_c_transformed, 0.33, 0)
    return result



def main():
    # 画像の読み込み
    img_a = cv2.imread("images/a.jpg")
    img_b = cv2.imread("images/b.jpg")
    img_c = cv2.imread("images/c.jpg")

    # 2枚ずつのステッチング
    ab_result = stitch_two_images(img_a, img_b)
    abc_result = stitch_two_images(ab_result, img_c)
    cv2.imwrite("images/3.3/a-b-c.jpg", abc_result)

    # 3枚同時のステッチング
    abc_results = stitch_three_images(img_a, img_b, img_c)
    cv2.imwrite("images/3.3/abc.jpg", abc_results)


if __name__ == "__main__":
    main()
