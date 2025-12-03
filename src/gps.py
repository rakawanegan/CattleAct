import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
import swifter  # noqa: F401

# ----- GPS関連の関数 ----- #


def calc_xy(phi_deg, lambda_deg, phi0_deg, lambda0_deg):
    """
    Convert geographic coordinates (latitude, longitude) to plane coordinates (x, y)
    using the plane rectangular coordinate system.

    Parameters:
        phi_deg (float or np.ndarray): Latitude(s) in degrees.
        lambda_deg (float or np.ndarray): Longitude(s) in degrees.
        phi0_deg (float): Origin latitude in degrees.
        lambda0_deg (float): Origin longitude in degrees.

    Returns:
        tuple: (x, y) coordinates in meters.
    """
    phi_rad = np.deg2rad(phi_deg)
    lambda_rad = np.deg2rad(lambda_deg)
    phi0_rad = np.deg2rad(phi0_deg)
    lambda0_rad = np.deg2rad(lambda0_deg)

    def A_array(n):
        """Compute the A coefficients for the series expansion."""
        A0 = 1 + n**2 / 4.0 + n**4 / 64.0
        A1 = -(3.0 / 2) * (n - n**3 / 8.0 - n**5 / 64.0)
        A2 = (15.0 / 16) * (n**2 - n**4 / 4.0)
        A3 = -(35.0 / 48) * (n**3 - (5.0 / 16) * n**5)
        A4 = (315.0 / 512) * n**4
        A5 = -(693.0 / 1280) * n**5
        return np.array([A0, A1, A2, A3, A4, A5])

    def alpha_array(n):
        """Compute the alpha coefficients for the series expansion."""
        a0 = np.nan  # dummy; index 0 unused
        a1 = (
            (1.0 / 2) * n
            - (2.0 / 3) * n**2
            + (5.0 / 16) * n**3
            + (41.0 / 180) * n**4
            - (127.0 / 288) * n**5
        )
        a2 = (
            (13.0 / 48) * n**2
            - (3.0 / 5) * n**3
            + (557.0 / 1440) * n**4
            + (281.0 / 630) * n**5
        )
        a3 = (61.0 / 240) * n**3 - (103.0 / 140) * n**4 + (15061.0 / 26880) * n**5
        a4 = (49561.0 / 161280) * n**4 - (179.0 / 168) * n**5
        a5 = (34729.0 / 80640) * n**5
        return np.array([a0, a1, a2, a3, a4, a5])

    m0 = 0.9999
    a = 6378137.0
    F = 298.257222101
    n = 1.0 / (2 * F - 1)
    A_arr = A_array(n)
    alpha_arr = alpha_array(n)

    A_ = (m0 * a / (1.0 + n)) * A_arr[0]
    S_ = (m0 * a / (1.0 + n)) * (
        A_arr[0] * phi0_rad + np.dot(A_arr[1:], np.sin(2 * phi0_rad * np.arange(1, 6)))
    )
    lambda_c = np.cos(lambda_rad - lambda0_rad)
    lambda_s = np.sin(lambda_rad - lambda0_rad)

    t = np.sinh(
        np.arctanh(np.sin(phi_rad))
        - (2 * np.sqrt(n) / (1 + n))
        * np.arctanh((2 * np.sqrt(n) / (1 + n)) * np.sin(phi_rad))
    )
    t_ = np.sqrt(1 + t**2)

    xi2 = np.arctan(t / lambda_c)
    eta2 = np.arctanh(lambda_s / t_)

    x_series = sum(
        alpha_arr[i] * np.sin(2 * i * xi2) * np.cosh(2 * i * eta2) for i in range(1, 6)
    )
    y_series = sum(
        alpha_arr[i] * np.cos(2 * i * xi2) * np.sinh(2 * i * eta2) for i in range(1, 6)
    )

    x = A_ * (xi2 + x_series) - S_
    y = A_ * (eta2 + y_series)
    return x, y


def convert_M2_gps_to_xy(cattle_N, cattle_E):
    """
    Convert GPS coordinates in M2 format to plane coordinates.

    The conversion applies:
        latitude  = 34 + (cattle_N * 100 - 3400) / 60
        longitude = 134 + (cattle_E * 100 - 13400) / 60
    Note: calc_xy returns (x, y) as (north, east); the function swaps them to (east, north).

    Parameters:
        cattle_N (float): Latitude value.
        cattle_E (float): Longitude value.

    Returns:
        tuple: (x, y) coordinates in meters.
    """
    cattle_N = 34 + (cattle_N * 100 - 3400) / 60
    cattle_E = 134 + (cattle_E * 100 - 13400) / 60
    cattle_y, cattle_x = calc_xy(cattle_N, cattle_E, 34.88252625, 134.8618488)
    return cattle_x, cattle_y


def drop_outlier_gps(gps_df):
    """
    Remove GPS data points outside the defined bounding box:
    x not in (-100, 200) or y not in (-100, 100) are discarded.

    Parameters:
        gps_df (pd.DataFrame): GPS data with 'x' and 'y' columns.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = gps_df.copy()
    return df[(df["x"] > -100) & (df["x"] < 200) & (df["y"] > -100) & (df["y"] < 100)]


def load_gps_data(
    output_path, gps_dir=None, start_time="14:40:00", end_time="14:50:00"
):
    """
    Load and process GPS data. If the output file exists, load it.
    Otherwise, read raw CSV files from a preset directory, filter by time,
    compute plane coordinates, drop outliers, and save the result.

    Parameters:
        output_path (str): Path to the processed GPS CSV.

    Returns:
        pd.DataFrame: Processed GPS data with columns ['timestamp', 'cattle_id', 'x', 'y'].
    """
    gps_columns = [
        "timestamp",
        "cattle_id",
        "sentence_id",
        "utc_time",
        "status",
        "latitude",
        "lat_dir",
        "longitude",
        "lon_dir",
        "speed_knots",
        "course",
        "date",
        "magnetic_variation",
        "magnetic_var_dir",
        "checksum",
    ]
    if os.path.exists(output_path):
        gps_df = pd.read_csv(output_path)
        gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"])
    else:
        if gps_dir is None:
            raise ValueError("GPS directory not provided.")
        gps_files = sorted(glob(os.path.join(gps_dir, "*Gps*.csv")))
        gps_df = pd.concat([pd.read_csv(p, names=gps_columns) for p in gps_files])
        gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"])
        mask = (gps_df["timestamp"].dt.time >= pd.to_datetime(start_time).time()) & (
            gps_df["timestamp"].dt.time <= pd.to_datetime(end_time).time()
        )
        gps_df = gps_df[mask].sort_values("timestamp").reset_index(drop=True)
        gps_df[["x", "y"]] = gps_df.swifter.apply(
            lambda x: convert_M2_gps_to_xy(x["latitude"], x["longitude"]),
            axis=1,
            result_type="expand",
        )
        gps_df = gps_df[["timestamp", "cattle_id", "x", "y"]]
        gps_df = drop_outlier_gps(gps_df)
        gps_df.to_csv(output_path, index=False)
    return gps_df


# ----- 射影変換関連の関数 ----- #


def load_drone_to_world_homography_matrix(filepath, scale=1):
    """
    Load the drone-to-world homography matrix from a file if it exists.
    Otherwise, compute the homography matrix from control points,
    save it to the file, and return it.

    Parameters:
        filepath (str): Path to save or load the homography matrix.
        scale (float, optional): Scale factor (pixels per meter). Default is 1.

    Returns:
        numpy.ndarray: The homography matrix mapping drone image to world plane.
    """
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        # Define control points in the drone image (pixel coordinates)
        pixel_points = np.array(
            [
                [618.8, 292.9],
                [513.5, 1643.1],
                [1677.9, 1550.2],
                [3009.5, 243.3],
                [2346.8, 1544.0],
            ],
            dtype=np.float32,
        )

        # Define control points in world coordinates (latitude, longitude)
        world_latlon = np.array(
            [
                [34.88325581, 134.8619961],
                [34.88252625, 134.8618488],
                [34.88253822, 134.8626284],
                [34.88319569, 134.8635738],
                [34.88251547, 134.8630714],
            ]
        )

        # Reference point for conversion (origin of the plane coordinate system)
        phi0, lambda0 = 34.88252625, 134.8618488

        # Convert world lat/lon to plane coordinates using calc_xy
        world_xy = []
        for lat, lon in world_latlon:
            x, y = calc_xy(lat, lon, phi0, lambda0)
            world_xy.append([x, y])
        world_xy = np.array(world_xy, dtype=np.float32)

        # Swap coordinates: assume calc_xy returns [North, East], so swap to [Easting, Northing]
        world_xy_swapped = np.array(
            [[pt[1], pt[0]] for pt in world_xy], dtype=np.float32
        )

        # Compute the homography matrix using RANSAC
        H, mask = cv2.findHomography(pixel_points, world_xy_swapped, cv2.RANSAC, 5.0)

        # Determine the extents of the world coordinates for transformation
        min_x, max_x = np.min(world_xy_swapped[:, 0]), np.max(world_xy_swapped[:, 0])
        min_y, max_y = np.min(world_xy_swapped[:, 1]), np.max(world_xy_swapped[:, 1])

        T = np.array(
            [[scale, 0, -min_x * scale], [0, scale, -min_y * scale], [0, 0, 1]],
            dtype=np.float32,
        )

        H_total = T @ H

        np.save(filepath, H_total)
        return H_total


def load_suzume_to_drone_homography_matrix(filepath):
    """
    Load the homography matrix from a file if it exists.
    Otherwise, compute the homography matrix using pre-defined
    correspondence points, save it to the file, and return it.

    Parameters:
        filepath (str): Path to save or load the homography matrix.

    Returns:
        numpy.ndarray: The homography matrix from camera to drone perspective.
    """
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        # Define correspondence points for Suzume (new image) and Drone
        pts_suzume = np.array(
            [
                [2792.79, 125.69],
                [3288.27, 299.11],
                [1114.34, 144.27],
                [2334.47, 1494.47],
            ],
            dtype=np.float32,
        )

        pts_drone = np.array(
            [[619.1, 298.0], [1282.3, 267.7], [552.6, 1054.5], [2730.0, 779.0]],
            dtype=np.float32,
        )

        # Calculate the homography matrix
        H, status = cv2.findHomography(pts_suzume, pts_drone)
        np.save(filepath, H)
        return H
