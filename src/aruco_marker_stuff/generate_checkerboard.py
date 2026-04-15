#!/usr/bin/env python3
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description="Generate a checkerboard pattern image")
parser.add_argument("--rows", type=int, default=9, help="Number of rows (squares)")
parser.add_argument("--cols", type=int, default=6, help="Number of columns (squares)")
parser.add_argument("--square-px", type=int, default=100, help="Size of each square in pixels")
parser.add_argument("-o", "--output", type=str, default="checkerboard.png", help="Output filename")
args = parser.parse_args()

rows, cols, sq = args.rows, args.cols, args.square_px
img = np.zeros((rows * sq, cols * sq), dtype=np.uint8)

for r in range(rows):
    for c in range(cols):
        if (r + c) % 2 == 0:
            img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 255

margin = sq
bordered = np.full((img.shape[0] + 2 * margin, img.shape[1] + 2 * margin), 255, dtype=np.uint8)
bordered[margin:margin + img.shape[0], margin:margin + img.shape[1]] = img

cv2.imwrite(args.output, bordered)
print(f"Saved {args.output}  ({bordered.shape[1]}x{bordered.shape[0]} px, {cols}x{rows} squares, {cols-1}x{rows-1} inner corners)")
