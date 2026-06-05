import pandas as pd
import math
from pathlib import Path


def split_csv(
    input_path: str | Path,
    split_ratio: float = 0.25,
    output_dir: str | Path | None = None,
    random_state: int = 42,
) -> tuple[Path, Path]:
    """
    Split a CSV file into two subsets at a given ratio.

    The smaller subset (split_ratio) gets a ``_init`` suffix and the
    larger one (1 - split_ratio) gets a ``_main`` suffix.

    Parameters
    ----------
    input_path : str | Path
        Path to the source CSV file.
    split_ratio : float, optional
        Fraction of rows assigned to the *init* (smaller) subset.
        Must be in (0, 0.5) so that init < main. Default is 0.25.
    output_dir : str | Path | None, optional
        Directory where the output files are written.  Defaults to the
        same directory as ``input_path``.
    random_state : int, optional
        Random seed for reproducible shuffling. Default is 42.

    Returns
    -------
    init_path, main_path : tuple[Path, Path]
        Paths of the two output files.

    Raises
    ------
    ValueError
        If ``split_ratio`` is not in the open interval (0, 0.5).
    """
    if not 0 < split_ratio < 0.5:
        raise ValueError(
            f"split_ratio must be in (0, 0.5) so that 'init' < 'main', "
            f"got {split_ratio}"
        )

    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n_init = math.ceil(len(df) * split_ratio)   # smaller subset
    df_init = df.iloc[:n_init]
    df_main = df.iloc[n_init:]

    stem = input_path.stem
    init_path = output_dir / f"{stem}_init.csv"
    main_path = output_dir / f"{stem}_main.csv"

    df_init.to_csv(init_path, index=False)
    df_main.to_csv(main_path, index=False)

    print(
        f"{input_path.name}: {len(df)} rows → "
        f"init={len(df_init)} ({init_path.name}), "
        f"main={len(df_main)} ({main_path.name})"
    )
    return init_path, main_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Split one or more CSV files into an *_init* subset (smaller) "
            "and a *_main* subset (larger) at a given ratio."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="CSV",
        help="One or more input CSV file paths to split.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.25,
        metavar="FLOAT",
        help="Fraction of rows for the init (smaller) subset. Must be in (0, 0.5). Default: 0.25.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for reproducible shuffling. Default: 42.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory for output files. Defaults to each input file's own directory.",
    )

    args = parser.parse_args()

    for f in args.inputs:
        split_csv(
            f,
            split_ratio=args.ratio,
            output_dir=args.output_dir,
            random_state=args.random_state,
        )