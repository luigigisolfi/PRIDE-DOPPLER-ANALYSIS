import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
from scipy.stats import norm
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
# --- LIBRARY IMPORTS ---
from pride_doppler.io.fdets import extract_parameters
from pride_doppler.analysis.processing import filter_data_zscore
from pride_doppler.analysis.allan import compute_oadev
from pride_doppler.analysis.geometry import compute_elevation_data
from pride_doppler.core.constants import HORIZONS_TARGETS, ANTENNA_DIAMETERS

# --- CONFIGURATION ---
st.set_page_config(page_title="PRIDE Doppler Analyzer", layout="wide", page_icon="📡")

# --- INIT SESSION STATE ---
if 'processing_queue' not in st.session_state:
    st.session_state.processing_queue = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# --- HELPER: CACHED FUNCTIONS ---
@st.cache_data(show_spinner=False)
def extract_parameters_wrapper(f_path):
    return extract_parameters(f_path)

@st.cache_data(ttl=3600)
def get_cached_elevation(station_data, target_id):
    try:
        return compute_elevation_data(station_data, target_id)
    except Exception as e:
        return None, None, 0.0

# --- BATCH PROCESSING ---
@st.cache_data(show_spinner=False)
def load_and_process_batch(queue, z_thresh):
    batch_raw = []
    batch_filtered = []
    file_tasks = []
    for m_name, f_name, root_dir in queue:
        input_dir = os.path.join(root_dir, m_name, f_name, 'input')
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) if f.startswith('Fdets') and f.endswith('r2i.txt')]
            for f in files:
                file_tasks.append((m_name, f_name, os.path.join(input_dir, f)))

    if not file_tasks: return [], []

    def worker(task):
        m_name, f_name, f_path = task
        raw = extract_parameters(f_path)
        if raw:
            raw.experiment_name = f_name
            raw.mission_name = m_name
            filt_list = filter_data_zscore([raw], threshold=z_thresh)
            filt = filt_list[0] if filt_list else None
            if filt:
                filt.experiment_name = f_name
                filt.mission_name = m_name
                return raw, filt
        return None, None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, t) for t in file_tasks]
        for future in as_completed(futures):
            r, f = future.result()
            if r and f:
                batch_raw.append(r)
                batch_filtered.append(f)
    return batch_raw, batch_filtered

# --- PLOTTING ADAPTERS ---
def get_time_series_fig(data):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(data.utc_datetime, data.signal_to_noise, '+-', color='blue', lw=0.5, ms=5)
    axs[0].set_ylabel('SNR'); axs[0].grid(True, alpha=0.3)
    axs[0].set_title(f"{data.receiving_station_name} - {data.utc_date}")

    axs[1].plot(data.utc_datetime, data.doppler_noise_hz*1e3, '+-', color='orange', lw=0.5, ms=5)
    axs[1].set_ylabel('Doppler Noise [mHz]'); axs[1].grid(True, alpha=0.3)

    axs[2].plot(data.utc_datetime, data.frequency_detection/1e6, 'o', color='black', ms=2)
    axs[2].set_ylabel('Freq Det [MHz]'); axs[2].set_xlabel('UTC Time'); axs[2].grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig

def get_comparison_fig(raw, filtered):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    retention = (len(filtered.utc_datetime) / len(raw.utc_datetime)) * 100 if len(raw.utc_datetime) > 0 else 0

    ax1.plot(raw.utc_datetime, raw.signal_to_noise, marker = 'o',  linestyle='-', color='blue', ms=2, lw = 0.5, label='Removed')
    ax1.plot(filtered.utc_datetime, filtered.signal_to_noise, 'o', color='orange', lw=0.5, ms=2, label=f'Kept ({retention:.1f}%)', alpha = 0.5)
    ax1.set_ylabel("SNR"); ax1.legend(loc='upper right'); ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{filtered.receiving_station_name} - {filtered.utc_date} (Raw vs Filtered)")

    ax2.plot(raw.utc_datetime, raw.doppler_noise_hz*1e3, marker = 'o',  linestyle='-', color='blue', ms=2, lw = 0.5, label='Removed')
    ax2.plot(filtered.utc_datetime, filtered.doppler_noise_hz*1e3, 'o', color='orange', lw=0.5, ms=2, label='Kept', alpha = 0.5)
    ax2.set_ylabel("Doppler Noise [mHz]"); ax2.grid(True, alpha=0.3)

    ax3.plot(raw.utc_datetime, raw.frequency_detection/1e6,'o', color='blue', ms=2, label='Removed')
    ax3.plot(filtered.utc_datetime, filtered.frequency_detection/1e6, 'o', color='orange', ms=2, label='Kept', alpha = 0.5)
    ax3.set_ylabel("Freq Det [MHz]"); ax3.set_xlabel("UTC Time"); ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()
    return fig

def get_plot_color(mission, exp_name):
    if mission == 'vex': return 'red'
    if mission == 'mro': return 'black'
    if mission == 'mex': return 'magenta'
    if exp_name == 'ed045a': return 'blue'
    if exp_name == 'ed045c': return 'cyan'
    if exp_name == 'ed045e': return 'orangered'
    # Random color generator
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def get_mission_summary_plot(stats_df, color_by="Experiment"):
    if stats_df.empty: return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 1. Create a color map to ensure consistency
    # This dictionary will store { "Group Name": "Color Code" }
    group_color_map = {}

    # Track which labels we have added to the legend so we don't duplicate them
    added_labels = set()

    for _, row in stats_df.iterrows():
        # Extract Data
        st_name = row['Station']
        curr_mission = row['Mission']
        experiment_name = row['Experiment']

        snr = row['Mean SNR (dB)']
        rms = row['RMS Doppler (mHz)']
        diam = ANTENNA_DIAMETERS.get(st_name, 30)

        # Identify the grouping key (e.g., "VEX" or "ed045a")
        group_name = row[color_by]

        # 2. Assign Color
        if group_name not in group_color_map:
            group_color_map[group_name] = get_plot_color(curr_mission, experiment_name)

        # Retrieve the fixed color for this group
        color = group_color_map[group_name]

        # 3. Handle Legend Labels
        # Only add the label to the plot if we haven't seen this group name before
        if group_name not in added_labels:
            label = group_name
            added_labels.add(group_name)
        else:
            label = None

        marker_style = {'fmt': 'o', 'markersize': 6, 'alpha': 0.6, 'color': color}

        # 4. Plotting
        # Plot 1: Station vs SNR
        ax1.errorbar(st_name, snr, label=label, **marker_style)

        # Plot 2: SNR vs RMS Doppler
        ax2.errorbar(snr, rms, label=label, **marker_style)

    # 5. Formatting
    ax1.set_ylabel('SNR [dB]'); ax1.set_xlabel('Station Code'); ax1.grid(True)
    # Legend is placed on top plot
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title=color_by)

    ax2.set_ylabel('RMS Doppler [mHz]'); ax2.set_xlabel('SNR [dB]'); ax2.grid(True); ax2.set_yscale('log')

    plt.tight_layout()
    return fig

# --- ALLAN DEVIATION PLOTTER ---
def plot_allan_deviation(data_list):
    fig, ax = plt.subplots(figsize=(10, 6))

    for p in data_list:
        taus, oadev, _ = compute_oadev(p, tau_min=1, tau_max=1000)
        if taus is not None:
            ax.loglog(taus, oadev, '.-', label=f"{p.receiving_station_name} ({p.experiment_name})", alpha=0.7)

    # Plot -1/2 slope reference white noise segments
    anchor_taus = [0, 10, 20, 50, 100]  # Anchor points

    # Create evenly-spaced reference levels in log space
    oadev_levels = np.logspace(np.log10(ax.get_ylim()[0]), np.log10(ax.get_ylim()[1]), 50)

    for i, tau_start in enumerate(anchor_taus):
        # Determine end tau (next anchor point, or plot limit)
        if i < len(anchor_taus) - 1:
            tau_end = anchor_taus[i + 1]
        else:
            tau_end = ax.get_xlim()[1]

        if tau_start == 0:
            tau_start = ax.get_xlim()[0]

        tau_segment = np.array([tau_start, tau_end])

        for oadev_ref in oadev_levels:
            # Each segment: OADEV(tau) = OADEV_ref * (tau/tau_start)^(-0.5)
            ref_segment = oadev_ref * (tau_segment / tau_start)**-0.5
            ax.loglog(tau_segment, ref_segment, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)


    #ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_xlabel("Averaging Time (τ) [s]")
    ax.set_ylabel("Overlapping Allan Deviation")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Force Integer Ticks on X-Axis
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')

    return fig
# --- MAIN APP UI ---
st.title("📡 PRIDE Doppler Characterization")

with st.sidebar:
    st.header("Settings")
    root_dir = st.text_input("Data Root", value="/Users/lgisolfi/Desktop/PRIDE_DATA_NEW/analysed_pride_data")
    all_missions = ['min', 'jui', 'mex', 'mro', 'vex']
    analysis_mode = st.radio("Analysis Scope", ["Single Experiment", "Full Mission", "Custom Selection (Global)"])

    if 'last_mode' not in st.session_state: st.session_state.last_mode = analysis_mode
    if st.session_state.last_mode != analysis_mode:
        st.session_state.processing_queue = []
        st.session_state.data_loaded = False
        st.session_state.last_mode = analysis_mode
        st.rerun()

    if analysis_mode == "Custom Selection (Global)":
        available_options = []
        for m in all_missions:
            m_path = os.path.join(root_dir, m)
            if os.path.exists(m_path):
                folders = sorted([f for f in os.listdir(m_path) if f.startswith(m) and os.path.isdir(os.path.join(m_path, f))])
                for f in folders: available_options.append((m, f, root_dir))
        options_map = {f"[{m.upper()}] {f}": (m, f, r) for m, f, r in available_options}
        with st.form("global_selector"):
            selected_labels = st.multiselect("Select Experiments", options=list(options_map.keys()), default=[])
            load_btn = st.form_submit_button("Load Selected Data")
        if load_btn:
            st.session_state.processing_queue = [options_map[label] for label in selected_labels]
            st.session_state.data_loaded = True
    else:
        selected_mission = st.selectbox("Select Mission", all_missions)
        mission_path = os.path.join(root_dir, selected_mission)
        available_folders = []
        if os.path.exists(mission_path):
            available_folders = sorted([
                f for f in os.listdir(mission_path)
                if f.startswith(selected_mission) and os.path.isdir(os.path.join(mission_path, f))
            ], reverse=True)
        if analysis_mode == "Full Mission":
            if st.button("Load Full Mission Data"):
                st.session_state.processing_queue = [(selected_mission, f, root_dir) for f in available_folders]
                st.session_state.data_loaded = True
        else:
            selected_folder = st.selectbox("Select Experiment", available_folders)
            if selected_folder:
                st.session_state.processing_queue = [(selected_mission, selected_folder, root_dir)]
                st.session_state.data_loaded = True

    st.divider()
    st.subheader("Processing")
    z_threshold = st.slider("Z-Score Threshold", 1.0, 10.0, 3.5)
    bad_thresh_mhz = st.number_input("Bad Station Limit (mHz)", value=5.0, step=0.5) / 1000.0
    show_bad = st.checkbox("Include 'Bad' Stations in Summary", value=False)

if st.session_state.data_loaded and st.session_state.processing_queue:
    queue_tuple = tuple(st.session_state.processing_queue)
    with st.spinner("Loading & Filtering Data..."):
        all_raw_data, all_filtered_data = load_and_process_batch(queue_tuple, z_threshold)

    if not all_filtered_data:
        st.error("No valid data found."); st.stop()

    def calculate_row(p_data):
        hid = HORIZONS_TARGETS.get(p_data.mission_name, {}).get('target', '-999')
        mean_snr = np.mean(p_data.signal_to_noise) if len(p_data.signal_to_noise) > 0 else 0
        mean_dopp = np.mean(p_data.doppler_noise_hz) if len(p_data.doppler_noise_hz) > 0 else 0
        rms_dopp = np.std(p_data.doppler_noise_hz) if len(p_data.doppler_noise_hz) > 0 else 0
        if not show_bad and abs(mean_dopp) > bad_thresh_mhz: return None
        _, _, mean_el = get_cached_elevation(p_data, hid)
        return {
            "Mission": p_data.mission_name, "Experiment": p_data.experiment_name,
            "Station": p_data.receiving_station_name, "Mean SNR (dB)": 10 * np.log10(mean_snr) if mean_snr > 0 else 0,
            "Mean Doppler (mHz)": mean_dopp * 1000, "RMS Doppler (mHz)": rms_dopp * 1000,
            "Mean Elevation": mean_el, "Status": "❌ Bad" if abs(mean_dopp) > bad_thresh_mhz else "✅ Good"
        }

    with st.spinner("Calculating Statistics..."):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(calculate_row, all_filtered_data))
    df_stats = pd.DataFrame([r for r in results if r is not None])

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary Plots", "🔍 Station Inspector", "📉 Histograms", "⏱️ Allan Deviation"])

    with tab1:
        c_plt, c_tbl = st.columns([1, 1])
        with c_plt:
            color_col = "Mission" if analysis_mode == "Custom Selection (Global)" else "Experiment"
            st.pyplot(get_mission_summary_plot(df_stats, color_by=color_col))
        with c_tbl:
            st.dataframe(df_stats.style.apply(lambda x: ['background-color: #ffcccc' if 'Bad' in x['Status'] else '' for i in x], axis=1), use_container_width=True)

    with tab2:
        st_names = sorted(list(set([d.receiving_station_name for d in all_filtered_data])))
        c1, c2 = st.columns(2)
        sel_stat = c1.selectbox("Select Station", st_names)
        candidates = [d for d in all_filtered_data if d.receiving_station_name == sel_stat]
        target_data = candidates[0] if candidates else None
        if len(candidates) > 1:
            opts = {f"{d.utc_date} | {d.experiment_name}": d for d in candidates}
            sel_opt = c2.selectbox("Select Observation", list(opts.keys()))
            target_data = opts[sel_opt]
        if target_data:
            raw_target = next((r for r in all_raw_data if r.receiving_station_name == target_data.receiving_station_name and r.utc_date == target_data.utc_date and r.mission_name == target_data.mission_name), None)
            st.divider()
            c_metrics, c_plot = st.columns([1, 3])
            with c_metrics:
                hid = HORIZONS_TARGETS.get(target_data.mission_name, {}).get('target', '-999')
                times, els, mean_el = get_cached_elevation(target_data, hid)
                st.metric("Mission / Exp", f"{target_data.mission_name.upper()} / {target_data.experiment_name}")
                st.metric("Elevation", f"{mean_el:.1f}°")
                st.metric("Base Frequency", f"{target_data.base_frequency/1e6:.2f} MHz")
                if times:
                    fig_el, ax_el = plt.subplots(figsize=(4, 2.5))
                    ax_el.plot(times, els, color='green')
                    ax_el.set_title("Elevation Profile", fontsize=9)
                    ax_el.set_ylabel("El [deg]"); ax_el.set_xlabel("Time"); ax_el.grid(True, alpha=0.3)
                    ax_el.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); ax_el.tick_params(labelsize=8)
                    fig_el.autofmt_xdate(); st.pyplot(fig_el)
            with c_plot:
                compare_mode = st.toggle("Compare Raw vs Filtered", value=True, key="inspector_toggle")
                if compare_mode and raw_target: st.pyplot(get_comparison_fig(raw_target, target_data))
                else: st.pyplot(get_time_series_fig(target_data))

    with tab3:
        valid_objs = [p for p in all_filtered_data if show_bad or abs(np.mean(p.doppler_noise_hz)) <= bad_thresh_mhz]
        if len(valid_objs) > 12: st.info(f"Showing first 12 of {len(valid_objs)} plots.")
        cols = st.columns(3)
        for i, p_data in enumerate(valid_objs[:12]):
            noise_mhz = p_data.doppler_noise_hz * 1000
            if len(noise_mhz) < 2: continue
            mu, std = norm.fit(noise_mhz)
            with cols[i % 3]:
                fig_g, ax_g = plt.subplots(figsize=(5, 3))
                ax_g.hist(noise_mhz, bins=40, density=True, alpha=0.6, color='steelblue')
                x = np.linspace(ax_g.get_xlim()[0], ax_g.get_xlim()[1], 100); ax_g.plot(x, norm.pdf(x, mu, std), 'r--', lw=1)
                ax_g.set_title(f"{target_data.mission_name.upper()}- Gaussian Fit of Doppler Noise | Station: {p_data.receiving_station_name}", fontsize=5)
                ax_g.set_xlabel("Doppler Noise [mHz]", fontsize = 6); ax_g.set_ylabel("Counts", fontsize = 6); ax_g.tick_params(labelsize=6)
                st.pyplot(fig_g); plt.close(fig_g)

    # --- TAB 4: ALLAN DEVIATION ---
    with tab4:
        st.subheader("Stability Analysis")

        # 1. Query Tool
        c_q1, c_q2 = st.columns([1, 3])
        with c_q1:
            target_tau = st.number_input("Query OADEV at Tau (s):", value=10.0, step=10.0)
            calc_btn = st.button("Compute Allan Deviation")

        if calc_btn:
            valid_objs = [p for p in all_filtered_data if show_bad or abs(np.mean(p.doppler_noise_hz)) <= bad_thresh_mhz]

            with c_q2:
                with st.spinner("Computing Allan Deviation..."):
                    # A. Plot
                    fig_allan = plot_allan_deviation(valid_objs)
                    st.pyplot(fig_allan)

            # B. Table
            st.write(f"**Values at Tau ≈ {target_tau} s**")
            adevs = []
            for p in valid_objs:
                taus, oadev, err = compute_oadev(p, tau_min=1, tau_max=1000)
                if taus is not None:
                    # Find closest Tau
                    idx = (np.abs(taus - target_tau)).argmin()
                    adevs.append({
                        "Station": p.receiving_station_name,
                        "Mission": p.mission_name,
                        "Experiment": p.experiment_name,
                        "Actual Tau (s)": f"{taus[idx]:.2f}",
                        "OADEV": f"{oadev[idx]:.2e}",
                        "Error": f"{err[idx]:.2e}"
                    })
            st.dataframe(pd.DataFrame(adevs), use_container_width=True)

else:
    st.info("👈 Select Scope and Load Data to begin.")