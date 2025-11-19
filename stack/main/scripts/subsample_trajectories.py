import pandas as pd
import argparse
import os


def subsample_dataframes(us_df, ys_df, n, keep_id=True):
    """
    Subsample dataframes by selecting every nth row.
    
    Args:
        us_df: Control inputs dataframe
        ys_df: Observations dataframe
        n: Sample every nth row (e.g., n=10 means keep rows 0, 10, 20, ...)
        keep_id: Whether to preserve the original ID column
    
    Returns:
        Tuple of (subsampled_us_df, subsampled_ys_df)
    """
    # Select every nth row
    us_subsampled = us_df.iloc[::n].copy()
    ys_subsampled = ys_df.iloc[::n].copy()
    
    if not keep_id:
        # Reset ID to be sequential starting from the first ID
        if 'ID' in us_subsampled.columns:
            start_id = us_subsampled['ID'].iloc[0]
            us_subsampled['ID'] = range(start_id, start_id + len(us_subsampled))
        if 'ID' in ys_subsampled.columns:
            start_id = ys_subsampled['ID'].iloc[0]
            ys_subsampled['ID'] = range(start_id, start_id + len(ys_subsampled))
    
    return us_subsampled, ys_subsampled


def main():
    parser = argparse.ArgumentParser(
        description='Subsample trajectory data by selecting every nth row'
    )
    parser.add_argument(
        '--inputs',
        type=str,
        default='../data/trajectories/dynamic/control_inputs_controlled_411.csv',
        help='Path to control inputs CSV file'
    )
    parser.add_argument(
        '--observations',
        type=str,
        default='../data/trajectories/dynamic/observations_controlled_411.csv',
        help='Path to observations CSV file'
    )
    parser.add_argument(
        '-n',
        type=int,
        required=True,
        help='Sample every nth row (e.g., 10 for every 10th row)'
    )
    parser.add_argument(
        '--output-inputs',
        type=str,
        default=None,
        help='Output path for subsampled control inputs (default: auto-generated)'
    )
    parser.add_argument(
        '--output-observations',
        type=str,
        default=None,
        help='Output path for subsampled observations (default: auto-generated)'
    )
    parser.add_argument(
        '--keep-original-id',
        action='store_true',
        help='Keep original ID values instead of making them sequential'
    )
    
    args = parser.parse_args()
    
    # Read the CSV files
    print(f"Reading {args.inputs}...")
    us_df = pd.read_csv(args.inputs)
    print(f"Reading {args.observations}...")
    ys_df = pd.read_csv(args.observations)
    
    print(f"\nOriginal shapes:")
    print(f"  Control inputs: {us_df.shape}")
    print(f"  Observations: {ys_df.shape}")
    
    # Subsample
    print(f"\nSubsampling every {args.n}th row...")
    us_subsampled, ys_subsampled = subsample_dataframes(
        us_df, ys_df, args.n, keep_id=args.keep_original_id
    )
    
    print(f"\nSubsampled shapes:")
    print(f"  Control inputs: {us_subsampled.shape}")
    print(f"  Observations: {ys_subsampled.shape}")
    
    # Generate output paths if not provided
    if args.output_inputs is None:
        base_dir = os.path.dirname(args.inputs)
        base_name = os.path.basename(args.inputs)
        name_parts = os.path.splitext(base_name)
        args.output_inputs = os.path.join(
            base_dir,
            f"{name_parts[0]}_n{args.n}{name_parts[1]}"
        )
    
    if args.output_observations is None:
        base_dir = os.path.dirname(args.observations)
        base_name = os.path.basename(args.observations)
        name_parts = os.path.splitext(base_name)
        args.output_observations = os.path.join(
            base_dir,
            f"{name_parts[0]}_n{args.n}{name_parts[1]}"
        )
    
    # Save the subsampled data
    print(f"\nSaving subsampled control inputs to {args.output_inputs}...")
    us_subsampled.to_csv(args.output_inputs, index=False)
    
    print(f"Saving subsampled observations to {args.output_observations}...")
    ys_subsampled.to_csv(args.output_observations, index=False)
    
    print("\n✓ Done!")
    print(f"\nFirst few rows of subsampled control inputs:")
    print(us_subsampled.head())
    print(f"\nFirst few rows of subsampled observations:")
    print(ys_subsampled.head())


if __name__ == '__main__':
    main()