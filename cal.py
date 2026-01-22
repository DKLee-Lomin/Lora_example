import math


# Return Euclidean distance between two 3D points.
def distance_3d(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


# Return midpoint coordinates between two 3D points.
def midpoint_3d(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)


# Parse a string into a 3-number 3D point tuple.
def parse_point(text):
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("point must have 3 numbers")
    return tuple(float(part) for part in parts)


# Read input points and print distance and midpoint.
def main():
    p1 = parse_point(input("Point 1 (x y z): "))
    p2 = parse_point(input("Point 2 (x y z): "))
    dist = distance_3d(p1, p2)
    mid = midpoint_3d(p1, p2)
    print(f"distance: {dist:.6f}")
    print(f"midpoint: ({mid[0]:.6f}, {mid[1]:.6f}, {mid[2]:.6f})")


if __name__ == "__main__":
    main()
