"""
=============================================================================
Engineering Data Systems Pipeline — BIO-01: Heart Rate Recovery Analysis
Course: Computer Programming 1 | Academic Year: 2026
Student: Farah Nicole Abdallah | ID: 25-0599
Dataset: Running and Heart Rate Data (Kaggle: mcandocia/running-heart-rate-recovery)
File Used: s1_stop_to_start.csv (renamed to dataset_original.csv)
Unique Filter: rest_time > 0 (records where runner actually stopped to recover)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.express as px
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)


class HeartRateRecoveryPipeline:
    """
    OOP Pipeline for BIO-01: Heart Rate Recovery Analysis.
    5 Modules: ingest_data, clean_data, analyze_data,
               visualize_static, visualize_animated
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_df   = None
        self.clean_df = None
        self.stats    = {}
        print("=" * 65)
        print("  BIO-01 Heart Rate Recovery Pipeline")
        print("  Student: Farah Nicole Abdallah | ID: 25-0599")
        print("=" * 65)

    # =========================================================================
    # MODULE 1: DATA INGESTION
    # =========================================================================
    def ingest_data(self):
        print("\n[1/5] INGESTING DATA...")
        try:
            df = pd.read_csv(self.filepath)
            print(f"  Loaded {len(df):,} total records.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File not found: {self.filepath}\n"
                "Make sure dataset_original.csv is inside the data/ folder."
            )
        except Exception as e:
            raise RuntimeError(f"Could not load file: {e}")

        # UNIQUE FILTER (Abdallah 25-0599):
        # Keep only rows where rest_time > 0
        # These are the true Heart Rate Recovery events
        try:
            df["rest_time"] = pd.to_numeric(df["rest_time"], errors="coerce")
            self.raw_df = df[df["rest_time"] > 0].reset_index(drop=True)
            print(f"  Unique Filter: rest_time > 0")
            print(f"  Records after filter: {len(self.raw_df):,}")
        except Exception as e:
            raise RuntimeError(f"Filter failed: {e}")

        self.raw_df.to_csv("data/dataset_original.csv", index=False)
        print("  Saved to data/dataset_original.csv")
        return self.raw_df

    # =========================================================================
    # MODULE 2: DATA CLEANING
    # =========================================================================
    def clean_data(self):
        print("\n[2/5] CLEANING DATA...")
        df = self.raw_df.copy()

        try:
            before = len(df)
            df.drop_duplicates(inplace=True)
            print(f"  Duplicates removed: {before - len(df)}")

            key_cols = [
                "rest_time", "heart_rate_stop", "heart_rate_start",
                "speed_stop", "speed_start", "altitude_stop",
                "altitude_start", "total_running_time"
            ]
            key_cols = [c for c in key_cols if c in df.columns]
            df = df[key_cols].copy()

            for col in key_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            before_null = len(df)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"  Null rows removed: {before_null - len(df)}")
            print(f"  Clean records remaining: {len(df):,}")

        except Exception as e:
            raise RuntimeError(f"Cleaning failed: {e}")

        # HRR = heart rate when stopped MINUS heart rate when restarting
        # Higher HRR = heart recovered more during the rest period
        df["HRR"] = df["heart_rate_stop"] - df["heart_rate_start"]

        self.clean_df = df
        self.clean_df.to_csv("data/dataset_cleaned.csv", index=False)
        print("  Saved to data/dataset_cleaned.csv")
        return self.clean_df

    # =========================================================================
    # MODULE 3: STATISTICAL ANALYSIS (NumPy)
    # =========================================================================
    def analyze_data(self):
        print("\n[3/5] PERFORMING STATISTICAL ANALYSIS...")
        df = self.clean_df
        stats = {}

        try:
            target_cols = ["HRR", "heart_rate_stop", "heart_rate_start",
                           "rest_time", "speed_stop"]
            target_cols = [c for c in target_cols if c in df.columns]

            # Descriptive Statistics
            desc = {}
            for col in target_cols:
                arr = np.array(df[col])
                desc[col] = {
                    "mean"    : np.mean(arr),
                    "median"  : np.median(arr),
                    "std"     : np.std(arr, ddof=1),
                    "variance": np.var(arr, ddof=1),
                    "min"     : np.min(arr),
                    "max"     : np.max(arr),
                }
            stats["descriptive"] = desc

            # Skewness
            def skewness(arr):
                n = len(arr)
                m = np.mean(arr)
                s = np.std(arr, ddof=1)
                return (n / ((n-1)*(n-2))) * np.sum(((arr - m) / s) ** 3)

            stats["skewness"] = {
                col: skewness(np.array(df[col])) for col in target_cols
            }

            # Outlier Detection (IQR)
            outliers = {}
            for col in target_cols:
                arr = np.array(df[col])
                Q1  = np.percentile(arr, 25)
                Q3  = np.percentile(arr, 75)
                IQR = Q3 - Q1
                low = Q1 - 1.5 * IQR
                up  = Q3 + 1.5 * IQR
                outliers[col] = {
                    "Q1": Q1, "Q3": Q3, "IQR": IQR,
                    "lower_fence": low, "upper_fence": up,
                    "n_outliers": int(np.sum((arr < low) | (arr > up)))
                }
            stats["outliers"] = outliers

            # Pearson Correlation
            corr_matrix = np.corrcoef(df[target_cols].values.T)
            stats["correlation_matrix"] = corr_matrix
            stats["correlation_cols"]   = target_cols

            # Comparative Group Analysis (Low HRR vs High HRR)
            hrr_arr    = np.array(df["HRR"])
            median_hrr = np.median(hrr_arr)
            group_low  = df[df["HRR"] <= median_hrr]
            group_high = df[df["HRR"] >  median_hrr]

            comp = {}
            for col in ["heart_rate_stop", "rest_time", "speed_stop"]:
                if col not in df.columns:
                    continue
                comp[col] = {
                    "low_HRR_mean" : np.mean(np.array(group_low[col])),
                    "high_HRR_mean": np.mean(np.array(group_high[col])),
                    "low_HRR_std"  : np.std(np.array(group_low[col]),  ddof=1),
                    "high_HRR_std" : np.std(np.array(group_high[col]), ddof=1),
                }
            stats["comparative"] = comp
            stats["median_hrr"]  = median_hrr

        except Exception as e:
            raise RuntimeError(f"Analysis failed: {e}")

        self.stats = stats
        self._print_summary()
        return stats

    def _print_summary(self):
        print("\n  ── DESCRIPTIVE STATISTICS ───────────────────────────────────")
        for col, vals in self.stats["descriptive"].items():
            print(f"\n  [{col}]")
            for k, v in vals.items():
                print(f"      {k:>10}: {v:>10.4f}")
        print("\n  ── SKEWNESS ─────────────────────────────────────────────────")
        for col, s in self.stats["skewness"].items():
            print(f"      {col:>20}: {s:>10.4f}")
        print("\n  ── OUTLIERS (IQR method) ────────────────────────────────────")
        for col, o in self.stats["outliers"].items():
            print(f"      {col:>20}: {o['n_outliers']} outliers | IQR = {o['IQR']:.4f}")
        print("  ─────────────────────────────────────────────────────────────\n")

    # =========================================================================
    # MODULE 4: STATIC VISUALIZATIONS (3 charts)
    # =========================================================================
    def visualize_static(self):
        print("[4/5] GENERATING STATIC VISUALIZATIONS...")
        df  = self.clean_df
        BG  = "#0d1117"
        PAN = "#161b22"
        C1  = "#58a6ff"
        C2  = "#f78166"
        C3  = "#3fb950"
        TXT = "#e6edf3"
        BDR = "#30363d"

        def style_ax(ax, fig):
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(PAN)
            ax.tick_params(colors=TXT)
            for spine in ax.spines.values():
                spine.set_edgecolor(BDR)

        # Plot 1: Histogram of HRR
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        style_ax(ax1, fig1)
        arr = np.array(df["HRR"])
        ax1.hist(arr, bins=40, color=C1, edgecolor=BG, alpha=0.85)
        ax1.axvline(np.mean(arr),   color=C2, linestyle="--", linewidth=2,
                    label=f"Mean = {np.mean(arr):.2f} bpm")
        ax1.axvline(np.median(arr), color=C3, linestyle=":",  linewidth=2,
                    label=f"Median = {np.median(arr):.2f} bpm")
        ax1.set_title("Distribution of Heart Rate Recovery (HRR)\nAbdallah | 25-0599",
                      color=TXT, fontsize=13, fontweight="bold")
        ax1.set_xlabel("HRR (bpm drop during rest)", color=TXT)
        ax1.set_ylabel("Frequency", color=TXT)
        ax1.legend(facecolor=PAN, labelcolor=TXT)
        plt.tight_layout()
        fig1.savefig("outputs/plot1_histogram.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig1)
        print("  Saved: outputs/plot1_histogram.png")

        # Plot 2: Boxplot — Low HRR vs High HRR
        median_hrr = self.stats["median_hrr"]
        low_grp    = df[df["HRR"] <= median_hrr]["HRR"].values
        high_grp   = df[df["HRR"] >  median_hrr]["HRR"].values

        fig2, ax2 = plt.subplots(figsize=(9, 6))
        style_ax(ax2, fig2)
        bp = ax2.boxplot(
            [low_grp, high_grp],
            labels=["Low HRR Group", "High HRR Group"],
            patch_artist=True,
            medianprops=dict(color=C2, linewidth=2.5),
            whiskerprops=dict(color=TXT),
            capprops=dict(color=TXT),
            flierprops=dict(marker="o", color=C3, alpha=0.5, markersize=4)
        )
        bp["boxes"][0].set_facecolor("#1e3a5f")
        bp["boxes"][1].set_facecolor("#3b1f2b")
        for box in bp["boxes"]:
            box.set_edgecolor(C1)
        ax2.set_title("Boxplot: Low HRR vs High HRR Groups\nAbdallah | 25-0599",
                      color=TXT, fontsize=13, fontweight="bold")
        ax2.set_ylabel("Heart Rate Recovery (bpm)", color=TXT)
        plt.tight_layout()
        fig2.savefig("outputs/plot2_boxplot.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig2)
        print("  Saved: outputs/plot2_boxplot.png")

        # Plot 3: Correlation Heatmap
        corr_cols   = self.stats["correlation_cols"]
        corr_matrix = self.stats["correlation_matrix"]

        fig3, ax3 = plt.subplots(figsize=(9, 7))
        style_ax(ax3, fig3)
        im = ax3.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        cbar = plt.colorbar(im, ax=ax3)
        cbar.ax.yaxis.set_tick_params(color=TXT)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TXT)
        ax3.set_xticks(range(len(corr_cols)))
        ax3.set_yticks(range(len(corr_cols)))
        ax3.set_xticklabels(corr_cols, rotation=35, ha="right",
                             color=TXT, fontsize=8)
        ax3.set_yticklabels(corr_cols, color=TXT, fontsize=8)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr_matrix[i, j]
                ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                         color="white" if abs(val) > 0.4 else "#888", fontsize=8)
        ax3.set_title("Pearson Correlation Heatmap\nAbdallah | 25-0599",
                      color=TXT, fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig3.savefig("outputs/plot3_heatmap.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig3)
        print("  Saved: outputs/plot3_heatmap.png")

    # =========================================================================
    # MODULE 5: ANIMATED VISUALIZATIONS (2 animations)
    # =========================================================================
    def visualize_animated(self):
        print("\n[5/5] GENERATING ANIMATED VISUALIZATIONS...")
        df  = self.clean_df
        BG  = "#0d1117"
        PAN = "#161b22"
        C1  = "#58a6ff"
        TXT = "#e6edf3"
        BDR = "#30363d"

        # Animation 1: Rolling Mean of HRR (Matplotlib GIF)
        arr    = np.array(df["HRR"])[:400]
        window = 20
        roll   = np.convolve(arr, np.ones(window) / window, mode="valid")
        x_raw  = np.arange(len(arr))
        x_roll = np.arange(window - 1, len(arr))

        fig_a, ax_a = plt.subplots(figsize=(11, 5))
        fig_a.patch.set_facecolor(BG)
        ax_a.set_facecolor(PAN)
        ax_a.tick_params(colors=TXT)
        for sp in ax_a.spines.values():
            sp.set_edgecolor(BDR)
        ax_a.plot(x_raw, arr, color="#30363d", linewidth=0.8,
                  alpha=0.5, label="Raw HRR")
        line_m, = ax_a.plot([], [], color=C1, linewidth=2.2,
                             label="Rolling Mean (20-window)")
        ax_a.set_xlim(0, len(arr))
        ax_a.set_ylim(arr.min() - 5, arr.max() + 5)
        ax_a.set_title("Animated Rolling Mean — Heart Rate Recovery\nAbdallah | 25-0599",
                        color=TXT, fontsize=13, fontweight="bold")
        ax_a.set_xlabel("Recovery Event Index", color=TXT)
        ax_a.set_ylabel("HRR (bpm)", color=TXT)
        ax_a.legend(facecolor=PAN, labelcolor=TXT)

        def init():
            line_m.set_data([], [])
            return (line_m,)

        def update(frame):
            line_m.set_data(x_roll[:frame+1], roll[:frame+1])
            return (line_m,)

        anim1 = animation.FuncAnimation(
            fig_a, update, frames=min(len(roll), 180),
            init_func=init, blit=True, interval=30
        )
        anim1.save("outputs/animation1_rolling_mean.gif",
                   writer="pillow", fps=25, dpi=110)
        plt.close(fig_a)
        print("  Saved: outputs/animation1_rolling_mean.gif")

        # Animation 2: Scatter — HRR vs Rest Time (Plotly HTML)
        plot_df = df[["HRR", "rest_time", "heart_rate_stop"]].copy()
        plot_df["group"] = np.where(
            plot_df["HRR"] <= self.stats["median_hrr"],
            "Low HRR", "High HRR"
        )
        plot_df["frame"] = (
            np.arange(len(plot_df)) // max(1, len(plot_df) // 25)
        )

        fig_b = px.scatter(
            plot_df,
            x="rest_time",
            y="HRR",
            color="group",
            animation_frame="frame",
            size="heart_rate_stop",
            size_max=18,
            color_discrete_map={"Low HRR": "#58a6ff", "High HRR": "#f78166"},
            title="Animated HRR vs Rest Time — Recovery Cluster Analysis<br>"
                  "<sup>BIO-01: Abdallah 25-0599</sup>",
            labels={"rest_time": "Rest Duration (seconds)",
                    "HRR": "Heart Rate Recovery (bpm)"},
            template="plotly_dark",
            opacity=0.75,
        )
        fig_b.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font_color="#e6edf3",
            title_font_size=14,
        )
        fig_b.write_html("outputs/animation2_scatter_cluster.html")
        print("  Saved: outputs/animation2_scatter_cluster.html")

    # =========================================================================
    # RUN FULL PIPELINE
    # =========================================================================
    def run(self):
        self.ingest_data()
        self.clean_data()
        self.analyze_data()
        self.visualize_static()
        self.visualize_animated()
        print("\n" + "=" * 65)
        print("  PIPELINE COMPLETE — check your outputs/ folder!")
        print("=" * 65)


if __name__ == "__main__":
    pipeline = HeartRateRecoveryPipeline(filepath="data/dataset_original.csv")
    pipeline.run()
