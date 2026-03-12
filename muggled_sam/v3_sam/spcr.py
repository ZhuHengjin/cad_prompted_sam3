"""Spatial color-to-gray conversion (SPCR).

Refactored from a script into callable utilities.
"""

from __future__ import annotations

import argparse
import math
from typing import Optional

import cv2
import numpy as np
from scipy.sparse.linalg import cg


def color_to_gray(
    image_bgr: np.ndarray,
    mu: int = 1,
    npi: int = 1,
    dpi: int = 4,
    alpha: int = 20,
    debug: bool = False,
) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    if image_bgr.ndim == 2:
        return image_bgr.astype(np.uint8)
    if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
        raise ValueError("image_bgr must be HxWx3 BGR")

    theta = npi * math.pi / dpi
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    height, width, _ = img_lab.shape

    pixels = img_lab.astype(int)
    ps = pixels.reshape((height * width, 3))
    l, a, b = map(list, zip(*ps))

    l = list(map(lambda x: x * 100 / 255, l))
    a = list(map(lambda x: x - 128, a))
    b = list(map(lambda x: x - 128, b))

    l_avg = sum(l) / len(l)
    pixels = np.array(list(zip(l, a, b))).reshape((height, width, 3))

    def delta(i, j) -> float:
        da, db = i[1] - j[1], i[2] - j[2]
        dl = i[0] - j[0]
        dist_c = math.sqrt(da**2 + db**2)
        crunch_dist_c = alpha * math.tanh(dist_c / alpha)
        if abs(dl) > crunch_dist_c:
            return dl
        if da * cos_theta + db * sin_theta >= 0:
            return crunch_dist_c
        return -crunch_dist_c

    # Calculate how many pixels belong to each pixel's neighborhood,
    # which will be used in constructing matrix A
    nneighb = [[0 for _ in range(0, width)] for _ in range(0, height)]
    for i in range(0, height):
        for j in range(0, width):
            neighb_top = i - mu if i - mu >= 0 else 0
            neighb_bot = i + mu if i + mu <= height - 1 else height - 1
            neighb_left = j - mu if j - mu >= 0 else 0
            neighb_right = j + mu if j + mu <= width - 1 else width - 1
            for ni in range(neighb_left, neighb_right + 1):
                for nj in range(neighb_top, neighb_bot + 1):
                    if i * width + j != ni * width + nj:
                        nneighb[i][j] += 1

    # Calculate target difference, where deltas[i][j] stores delta_ij
    deltas = [[0 for _ in range(0, height * width)] for _ in range(0, height * width)]
    for i in range(0, height):
        for j in range(0, width):
            neighb_top = i - mu if i - mu >= 0 else 0
            neighb_bot = i + mu if i + mu <= height - 1 else height - 1
            neighb_left = j - mu if j - mu >= 0 else 0
            neighb_right = j + mu if j + mu <= width - 1 else width - 1
            for ni in range(neighb_top, neighb_bot + 1):
                for nj in range(neighb_left, neighb_right + 1):
                    deltas[i * width + j][ni * width + nj] = delta(pixels[i][j], pixels[ni][nj])

    # Construct matrix A in the linear system to be solved, where
    # A_ij = 2N if i = j, where N is the number of pixels in i's neighborhood
    # A_ij = -2 if j in N(i)
    # A_ij = 0  otherwise
    diag = []
    for row in nneighb:
        for col in row:
            diag.append(2 * col)
    A = np.diag(diag)
    for i in range(0, height):
        for j in range(0, width):
            neighb_top = i - mu if i - mu >= 0 else 0
            neighb_bot = i + mu if i + mu <= height - 1 else height - 1
            neighb_left = j - mu if j - mu >= 0 else 0
            neighb_right = j + mu if j + mu <= width - 1 else width - 1
            for ni in range(neighb_top, neighb_bot + 1):
                for nj in range(neighb_left, neighb_right + 1):
                    if i * width + j != ni * width + nj:
                        A[i * width + j][ni * width + nj] = -2
    if debug:
        print("********* Finished construct matrix A in linear system *********")
        print(A)

    # Construct vector b in the linear system to be solve, where
    # b_i = sum_{j in N(i)} (delta_ij - delta_ji)
    B = np.zeros((height * width,))
    for i in range(0, height):
        for j in range(0, width):
            neighb_top = i - mu if i - mu >= 0 else 0
            neighb_bot = i + mu if i + mu <= height - 1 else height - 1
            neighb_left = j - mu if j - mu >= 0 else 0
            neighb_right = j + mu if j + mu <= width - 1 else width - 1
            for ni in range(neighb_top, neighb_bot + 1):
                for nj in range(neighb_left, neighb_right + 1):
                    B[i * width + j] += deltas[i * width + j][ni * width + nj] - deltas[ni * width + nj][i * width + j]

    g_flat = np.asarray([[pixels[row][col][0] for col in range(0, width)] for row in range(0, height)]).flatten()
    res, info = cg(A, B, x0=g_flat)

    res = res + (l_avg - res.mean())
    res = list(map(lambda x: x * 255 / 100, res))

    out = np.reshape(res, (height, width))
    if debug:
        print("********* Finished solving linear system *********")
        print(out)
        print("status", info)

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    return out_u8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Color-to-gray conversion")
    parser.add_argument("--input", type=str, help="input image path", required=True)
    parser.add_argument("--output", type=str, help="output image path", required=True)
    parser.add_argument("--mu", type=int, help="neighborhood pixel size", required=True)
    parser.add_argument("--npi", type=int, default=1, help="numerator of theta: npi * pi")
    parser.add_argument("--dpi", type=int, default=4, help="denominator of theta")
    parser.add_argument("--alpha", "-a", type=int, default=20, help="user parameter alpha")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("start processing image", args.input)
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)
    out = color_to_gray(img, mu=args.mu, npi=args.npi, dpi=args.dpi, alpha=args.alpha, debug=args.debug)
    cv2.imwrite(args.output, out)
    print("finished processing image", args.input, "output image to", args.output)


if __name__ == "__main__":
    main()
