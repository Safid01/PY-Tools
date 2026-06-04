"""
eeg_gdf_viewer_fixed.py

Tkinter GUI to safely view .gdf EEG files (Training / Evaluation).
Shows:
 - File details (sampling rate, channels, duration)
 - Event mapping & trial counts
 - EEG Signal Visualization (stacked first N channels, short window)
 - Event counts bar chart
 - Power Spectral Density (PSD) using Welch (1-50 Hz)

This version avoids mne's interactive plotting and uses matplotlib only,
which prevents GUI backend collisions and large-memory plotting that caused crashes.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mne
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.signal import welch
import traceback
import os

# Use Agg backend for safety when embedding in Tkinter
matplotlib.use("Agg")

# Maximum amount of time (seconds) to plot in the time-series viewer
PLOT_SECONDS = 5.0

# Maximum number of channels to render in the time-series plot (stacked)
MAX_PLOT_CHANNELS = 10

# PSD parameters
PSD_FMIN = 1.0
PSD_FMAX = 50.0
PSD_NPERSEG = 1024  # larger value gives better frequency resolution; if file shorter, welch adjusts

# Event code mapping (common for BCI IV-2a). If your dataset differs, update this dict.
EVENT_MAP = {
    769: "Left Hand Imagery",
    770: "Right Hand Imagery",
    771: "Feet Imagery",
    772: "Tongue Imagery",
    1023: "Trial Start / Cue",
    1024: "Trial End",
    1072: "Misc / Calibration",
    276: "Marker 276",
    277: "Marker 277",
    32766: "Record Start/Stop",
}


class EEGViewer:
    def __init__(self, master):
        self.master = master
        master.title("EEG GDF Viewer — Fixed (Safe plotting)")
        master.geometry("1200x800")

        # Top controls
        top = ttk.Frame(master)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(top, text="Open GDF File", command=self.open_file).pack(side=tk.LEFT)
        ttk.Button(top, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=8)

        # Left area: details text
        left_frame = ttk.Frame(master)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=6, anchor="n")

        details_label = ttk.Label(left_frame, text="File Details & Event Summary", font=("Segoe UI", 11, "bold"))
        details_label.pack(anchor="nw")

        self.text = tk.Text(left_frame, width=44, height=40, wrap="word", font=("Consolas", 10))
        self.text.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        self.text.configure(state="disabled")

        text_scroll = ttk.Scrollbar(left_frame, command=self.text.yview)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text['yscrollcommand'] = text_scroll.set

        # Right area: plots
        right_frame = ttk.Frame(master)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.plot_container = ttk.Frame(right_frame)
        self.plot_container.pack(fill=tk.BOTH, expand=True)

        # Keep references to canvases to destroy them cleanly later
        self._canvases = []

    def clear_all(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.configure(state="disabled")
        self._clear_plots()

    def _clear_plots(self):
        for c in self._canvases:
            try:
                c.get_tk_widget().destroy()
            except Exception:
                pass
        self._canvases = []

    def open_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("GDF Files", "*.gdf"), ("All files", "*.*")])
        if not filepath:
            return
        try:
            self._process_file(filepath)
        except Exception as e:
            tb = traceback.format_exc()
            messagebox.showerror("Error", f"Failed to open/plot file:\n{e}\n\nTraceback:\n{tb}")

    def _process_file(self, filepath):
        self.clear_all()
        self._append_text(f"Opening: {os.path.basename(filepath)}\n")

        # Load with mne (non-graphical). Use preload=False to reduce memory; we'll request small slices.
        raw = mne.io.read_raw_gdf(filepath, preload=False, verbose=False)

        sfreq = raw.info['sfreq']
        n_channels = len(raw.ch_names)
        duration = raw.times[-1] if raw.times.size else 0.0

        # Annotations and events
        annotations = raw.annotations
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
        except Exception:
            # If events_from_annotations fails, fallback to empty
            events = np.zeros((0, 3), dtype=int)
            event_id = {}

        # Count events
        counts = {}
        for e in events[:, 2] if events.size else []:
            counts[e] = counts.get(e, 0) + 1

        # Display details
        self._append_text(f"Sampling frequency: {sfreq} Hz\n")
        self._append_text(f"Number of channels: {n_channels}\n")
        self._append_text(f"Duration: {duration:.2f} s  (~{duration/60.0:.1f} min)\n")
        self._append_text(f"Total annotations: {len(annotations) if annotations is not None else 0}\n\n")

        # Show channels (first 40 to avoid huge print)
        ch_list = raw.ch_names
        self._append_text("Channels (first 40 shown):\n")
        self._append_text(", ".join(ch_list[:40]) + ("\n" if len(ch_list) <= 40 else ", ...\n"))

        # Show event counts and mapping
        self._append_text("\nEvent counts:\n")
        if counts:
            # map event codes -> names if known
            for code in sorted(counts.keys()):
                name = EVENT_MAP.get(code, event_id.get(str(code), event_id.get(code, "Unknown")))
                # if event_id mapping was string->int, we rely on EVENT_MAP; show both
                self._append_text(f"{code} -> {name}: {counts[code]} occurrences\n")
        else:
            self._append_text("No events detected (or events parsing failed).\n")

        self._append_text("\nKnown event code mapping (defaults):\n")
        for code, name in sorted(EVENT_MAP.items()):
            self._append_text(f"{code}: {name}\n")

        # Now extract small chunk of data for plotting to avoid huge memory usage
        n_plot_samples = int(min(sfreq * PLOT_SECONDS, raw.n_times))
        if n_plot_samples <= 0:
            raise RuntimeError("Recording contains zero samples.")

        # Choose a slice: start at 0 (beginning). Could add UI later for selecting time window.
        start_sample = 0
        stop_sample = start_sample + n_plot_samples

        # pick first few channels for waveform plot
        n_plot_ch = min(n_channels, MAX_PLOT_CHANNELS)

        # Load only the small data slice into memory
        data_slice, times = raw[:n_plot_ch, start_sample:stop_sample]  # shape: (n_plot_ch, n_plot_samples)
        # data is in Volts (mne default), convert to microvolts for plotting
        data_uv = data_slice * 1e6
        times = times  # seconds

        # Compute PSD (average across channels if many)
        # We'll compute PSD on a downsampled selection if file is long; compute on first 60s or on the plotted window
        psd_seconds = min(60.0, duration)  # use up to 60 s for PSD
        psd_n_samples = int(min(psd_seconds * sfreq, raw.n_times))
        psd_start, psd_stop = 0, psd_n_samples

        # Get fewer channels for PSD (avoid computing PSD across all channels if not necessary)
        psd_pick_ch = slice(0, min(n_channels, 8))

        psd_data, _ = raw[psd_pick_ch, psd_start:psd_stop]  # shape (n_ch, n_samples)
        # compute PSD per channel and average
        fs = sfreq
        freqs = None
        psd_avg = None
        try:
            psd_list = []
            for ch in range(psd_data.shape[0]):
                f, Pxx = welch(psd_data[ch, :], fs=fs, nperseg=min(PSD_NPERSEG, psd_data.shape[1]))
                psd_list.append(Pxx)
                freqs = f
            if psd_list:
                psd_avg = np.mean(np.vstack(psd_list), axis=0)
            else:
                psd_avg = np.array([])
        except Exception as e:
            self._append_text("\nFailed to compute PSD: " + str(e) + "\n")
            freqs = np.array([])
            psd_avg = np.array([])

        # Prepare event-bar data
        event_codes = sorted(counts.keys())
        event_counts = [counts[c] for c in event_codes]

        # Build and show plots (matplotlib Figures). Do not use mne plotting functions.
        self._clear_plots()
        # 1) EEG time-series (stacked)
        fig1 = self._make_timeseries_figure(times, data_uv, raw.ch_names[:n_plot_ch])
        canvas1 = FigureCanvasTkAgg(fig1, master=self.plot_container)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=6, pady=6)
        self._canvases.append(canvas1)

        # 2) Event counts bar chart
        fig2 = self._make_event_bar_figure(event_codes, event_counts)
        canvas2 = FigureCanvasTkAgg(fig2, master=self.plot_container)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=6, pady=6)
        self._canvases.append(canvas2)

        # 3) PSD figure
        fig3 = self._make_psd_figure(freqs, psd_avg)
        canvas3 = FigureCanvasTkAgg(fig3, master=self.plot_container)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=6, pady=6)
        self._canvases.append(canvas3)

    def _append_text(self, s):
        self.text.configure(state="normal")
        self.text.insert(tk.END, s)
        self.text.configure(state="disabled")

    def _make_timeseries_figure(self, times, data_uv, ch_names):
        # stacked plot
        n_ch = data_uv.shape[0]
        offset = np.max(np.ptp(data_uv, axis=1)) * 1.4 if n_ch > 0 else 1.0
        if not np.isfinite(offset) or offset == 0:
            offset = 100.0  # fallback

        fig, ax = plt.subplots(figsize=(9, 3 + 0.25 * n_ch), constrained_layout=True)
        for i in range(n_ch):
            ax.plot(times, data_uv[i, :] + i * offset)
        # label y-ticks with channel names
        yticks = [i * offset for i in range(n_ch)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ch_names, fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"EEG signals — first {n_ch} channels — {times[-1]-times[0]:.2f} s window")
        return fig

    def _make_event_bar_figure(self, event_codes, event_counts):
        fig, ax = plt.subplots(figsize=(9, 2.5), constrained_layout=True)
        if len(event_codes) == 0:
            ax.text(0.5, 0.5, "No events to display", ha="center", va="center")
            ax.set_axis_off()
            return fig

        labels = []
        for c in event_codes:
            # prefer human-readable name if present
            labels.append(EVENT_MAP.get(c, str(c)))

        positions = np.arange(len(event_codes))
        ax.bar(positions, event_counts)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Event Mapping & Trial Counts")
        for i, v in enumerate(event_counts):
            ax.text(i, v + max(1, 0.01 * max(event_counts)), str(v), ha="center", va="bottom", fontsize=8)
        return fig

    def _make_psd_figure(self, freqs, psd_avg):
        fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
        if freqs is None or freqs.size == 0 or psd_avg is None or psd_avg.size == 0:
            ax.text(0.5, 0.5, "PSD unavailable", ha="center", va="center")
            ax.set_axis_off()
            return fig
        # limit to requested band
        mask = (freqs >= PSD_FMIN) & (freqs <= PSD_FMAX)
        if not np.any(mask):
            ax.text(0.5, 0.5, "No frequency content in requested band", ha="center", va="center")
            ax.set_axis_off()
            return fig
        ax.semilogy(freqs[mask], psd_avg[mask])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (V^2/Hz)")
        ax.set_title(f"Power Spectral Density ({PSD_FMIN}–{PSD_FMAX} Hz) — averaged over channels")
        return fig


def main():
    root = tk.Tk()
    app = EEGViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()