def filter_mean_non_causal(data, window_size = 1):
    filtered = data.copy()
    cols_to_filter = [col for col in data.columns if col != 't']
    filtered[cols_to_filter] = data[cols_to_filter].rolling(window=window_size, center=True, min_periods=1).mean()
    return filtered

filter_data = filter_mean_non_causal