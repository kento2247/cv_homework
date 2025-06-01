import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込み（ファイル名を自分のものに置き換えてください）
img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)

# 特徴点検出器（ORBを使用）
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# マッチング
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 上位50点のみ使用（必要に応じて調整）
matches = matches[:50]

# 対応点抽出
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Fundamental Matrix 推定
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# Fundamental Matrixを出力
print("Fundamental Matrix:")
print(F)

# 正しいマッチだけを選択
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]


# エピポーラ線を描画する関数
def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r_line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r_line[2] / r_line[1]])
        x1, y1 = map(int, [c, -(r_line[2] + r_line[0] * c) / r_line[1]])
        img1_color = cv2.line(
            img1_color, (x0, y0), (x1, y1), color, 2
        )  # 線の太さを2に変更
        img1_color = cv2.circle(img1_color, tuple(pt1), 30, color, -15)
        img2_color = cv2.circle(img2_color, tuple(pt2), 30, color, -15)
    return img1_color, img2_color


# エピポーラ線の計算（img1側）
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1.astype(int), pts2.astype(int))

# エピポーラ線の計算（img2側）
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2.astype(int), pts1.astype(int))

# 結果表示
plt.figure(figsize=(15, 5))
plt.subplot(121), plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
plt.title("Epipolar Lines on Image 1"), plt.axis("off")
plt.subplot(122), plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.title("Epipolar Lines on Image 2"), plt.axis("off")
# plt.show()
# save
plt.savefig("epipolar_lines.png")
