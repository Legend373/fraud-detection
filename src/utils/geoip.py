import pandas as pd
import numpy as np

class GeoIPMapper:
    def __init__(self, fraud_df, ip_df):
        self.fraud_df = fraud_df
        self.ip_df = ip_df

    def ip_to_int(self, ip):
        try:
            ip_str = str(ip)
            parts = ip_str.split(".")
            if len(parts) != 4:
                return np.nan
            return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
        except:
            return np.nan

    def convert_ips(self):
        # Fill missing or invalid IPs as NaN first
        self.fraud_df["ip_address"] = self.fraud_df["ip_address"].fillna("0.0.0.0")
        self.fraud_df["ip_int"] = self.fraud_df["ip_address"].apply(self.ip_to_int)
        return self

    def merge_country(self):
        # Sort ip ranges for faster lookup
        self.ip_df = self.ip_df.sort_values("lower_bound_ip_address")

        def lookup_country(x):
            matches = self.ip_df[
                (self.ip_df["lower_bound_ip_address"] <= x) &
                (self.ip_df["upper_bound_ip_address"] >= x)
            ]
            if not matches.empty:
                return matches["country"].values[0]
            return "Unknown"

        self.fraud_df["country"] = self.fraud_df["ip_int"].apply(lookup_country)
        return self.fraud_df
