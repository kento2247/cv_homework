import cv2
import math
import numpy as np
import argparse


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx : :]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single image dehazing using dark channel prior"
    )
    parser.add_argument(
        "--input",
        nargs="?",
        default="data/hazy/01_indoor_hazy.jpg",
        help="Input image path (default: ./image/15.png)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./image/J.png",
        help="Output image path (default: ./image/J.png)",
    )
    parser.add_argument(
        "--show", "-s", action="store_true", help="Show intermediate results"
    )

    args = parser.parse_args()

    src = cv2.imread(args.input)
    if src is None:
        print(f"Error: Could not read image from {args.input}")
        exit(1)

    I = src.astype("float64") / 255

    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A, 0.1)

    if args.show:
        cv2.imshow("dark", dark)
        cv2.imshow("t", t)
        cv2.imshow("I", src)
        cv2.imshow("J", J)
        cv2.waitKey()

    # Create a side-by-side comparison
    h, w = src.shape[:2]
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    comparison[:, :w] = src
    comparison[:, w:] = (J * 255).astype(np.uint8)

    cv2.imwrite(args.output, comparison)
    print(f"Side-by-side comparison saved to {args.output}")
