"""Generates icon.png (1024x1024), a simple eye icon, for the macOS app bundle.

Used by 'Build Mac App.command', which turns it into icon.icns.
Run manually with:  python make_icon.py
"""
import cv2
import numpy as np

S = 1024


def main(out: str = "icon.png"):
    img = np.zeros((S, S, 4), dtype=np.uint8)

    # Rounded-square background with a vertical gradient (BGR).
    top, bottom = np.array([70, 40, 28]), np.array([36, 22, 14])
    grad = np.linspace(top, bottom, S).astype(np.uint8)  # (S, 3)
    bg = np.repeat(grad[:, None, :], S, axis=1)
    mask = np.zeros((S, S), dtype=np.uint8)
    r, m = 200, 40
    cv2.rectangle(mask, (m + r, m), (S - m - r, S - m), 255, -1)
    cv2.rectangle(mask, (m, m + r), (S - m, S - m - r), 255, -1)
    for cx, cy in [(m + r, m + r), (S - m - r, m + r), (m + r, S - m - r), (S - m - r, S - m - r)]:
        cv2.circle(mask, (cx, cy), r, 255, -1)
    img[:, :, :3] = bg
    img[:, :, 3] = mask

    c = (S // 2, S // 2)
    # Eye white (almond shape from two ellipse arcs).
    cv2.ellipse(img, c, (360, 215), 0, 0, 360, (245, 245, 245, 255), -1, cv2.LINE_AA)
    # Iris, pupil, highlight.
    cv2.circle(img, c, 150, (255, 140, 80, 255), -1, cv2.LINE_AA)      # blue-ish iris (BGR)
    cv2.circle(img, c, 150, (200, 100, 50, 255), 10, cv2.LINE_AA)
    cv2.circle(img, c, 72, (20, 14, 10, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (c[0] - 45, c[1] - 50), 28, (255, 255, 255, 255), -1, cv2.LINE_AA)
    # Eyelid outline.
    cv2.ellipse(img, c, (360, 215), 0, 0, 360, (210, 190, 170, 255), 12, cv2.LINE_AA)

    # Keep everything outside the rounded square transparent.
    img[mask == 0] = 0
    cv2.imwrite(out, img)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
