from __future__ import annotations

from .console import console

class FigurePlotter:
    @staticmethod
    def plot_csv_files(files_dict: dict, x_column: str, y_column: str, window_size: int = 10):
        """
        Plot raw and rolling average lines from multiple CSV files.

        Args:
            files (dict): Dict mapping legend labels to CSV file paths.
            x_column (str): Column to use for x-axis.
            y_column (str): Column to use for y-axis.
            window_size (int): Window size for rolling average.
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        plt.figure(figsize=(10, 6))

        for label, file_path in files_dict.items():
            try:
                # Load CSV, skipping first row
                df = pd.read_csv(file_path, skiprows=1)

                if x_column not in df.columns or y_column not in df.columns:
                    console.error(f"Skipping {file_path}: missing '{x_column}' or '{y_column}'")
                    continue

                x = df[x_column]
                y = df[y_column]
                y_smooth = y.rolling(window=window_size, min_periods=1).mean()

                # Plot raw (dashed) and get color
                raw_line, = plt.plot(x, y, linestyle='--', alpha=0)
                color = raw_line.get_color()

                # Plot rolling average (solid) using same color, with legend
                plt.plot(x, y_smooth, linestyle='-', color=color, label=label)

            except Exception as e:
                console.error(f"Error reading {file_path}: {e}")
                continue

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f"{y_column} vs {x_column} (Rolling window={window_size})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return
