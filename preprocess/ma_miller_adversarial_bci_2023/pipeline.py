# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "requests==2.32.3",
#   "py7zr",
# ]
# ///

import datetime
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import py7zr
import requests

from torch_brain.data import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from torch_brain.pipeline import BrainsetPipeline
from torch_brain.data import ArrayDict, Data, Interval, IrregularTimeSeries
from torch_brain.data import RegularTimeSeries

DRYAD_BASE = "https://datadryad.org"
DATASET_DOI = "doi%3A10.5061%2Fdryad.cvdncjt7n"

TASK_MAP = {
    "iso": "isometric_wrist",
    "pg": "power_grasp",
    "key": "key_grasp",
    "co": "center_out_reaching",
    "rt": "random_target_reaching",
}

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


# ---------------------------------------------------------------------------
# Dryad API helpers (mirrors gallego_carracedo_lfp_2022/pipeline.py)
# ---------------------------------------------------------------------------


def _get_access_token() -> str:
    app_id = os.environ.get("DRYAD_APP_ID")
    app_secret = os.environ.get("DRYAD_APP_SECRET")
    if not app_id or not app_secret:
        raise RuntimeError(
            "Set DRYAD_APP_ID and DRYAD_APP_SECRET env vars.\n"
            "Get credentials at: https://datadryad.org (My account → API)"
        )
    resp = requests.post(
        f"{DRYAD_BASE}/oauth/token",
        data={
            "client_id": app_id,
            "client_secret": app_secret,
            "grant_type": "client_credentials",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _get_dryad_files():
    versions_url = f"{DRYAD_BASE}/api/v2/datasets/{DATASET_DOI}/versions"
    resp = requests.get(versions_url, timeout=30)
    resp.raise_for_status()
    versions = resp.json()["_embedded"]["stash:versions"]
    files_href = versions[-1]["_links"]["stash:files"]["href"]
    resp = requests.get(f"{DRYAD_BASE}{files_href}?per_page=100", timeout=30)
    resp.raise_for_status()
    return resp.json()["_embedded"]["stash:files"]


def _parse_archive_id(filename: str) -> str:
    """'Jango_ISO_2016.7z' -> 'jango_iso_2016'"""
    stem = Path(filename).stem
    return "_".join(p.lower() for p in stem.split("_"))


def _archive_id_to_task(archive_id: str) -> str:
    """'jango_iso_2016' -> 'isometric_wrist'"""
    task_abbrev = archive_id.split("_")[1]
    return TASK_MAP[task_abbrev]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline(BrainsetPipeline):
    brainset_id = "ma_miller_adversarial_bci_2023"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        rows = []
        for f in _get_dryad_files():
            filename = f["path"]
            if not filename.endswith(".7z"):
                continue
            download_url = DRYAD_BASE + f["_links"]["stash:download"]["href"]
            rows.append(
                {
                    "session_id": _parse_archive_id(filename),
                    "filename": filename,
                    "url": download_url,
                }
            )
        return pd.DataFrame(rows).set_index("session_id")

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        archive_dest = self.raw_dir / manifest_item.filename
        extract_dir = self.raw_dir / self._asset_id

        if extract_dir.exists() and not self.args.redownload:
            self.update_status("Already downloaded and extracted")
            return extract_dir

        if not archive_dest.exists() or self.args.redownload:
            token = _get_access_token()
            resp = requests.get(
                manifest_item.url,
                headers={"Authorization": f"Bearer {token}"},
                stream=True,
                timeout=600,
            )
            resp.raise_for_status()
            with open(archive_dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)

        self.update_status("EXTRACTING")
        extract_dir.mkdir(exist_ok=True, parents=True)
        with py7zr.SevenZipFile(archive_dest, mode="r") as z:
            z.extractall(path=extract_dir)

        return extract_dir

    def process(self, extract_dir):
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        archive_id = self._asset_id
        task = _archive_id_to_task(archive_id)

        brainset_description = BrainsetDescription(
            id="ma_miller_adversarial_bci_2023",
            origin_version="dryad/10.5061/dryad.cvdncjt7n",
            derived_version="1.0.0",
            source="https://datadryad.org/dataset/doi:10.5061/dryad.cvdncjt7n",
            description=(
                "Neural and behavioral recordings from M1 of six rhesus macaques "
                "performing isometric wrist, power grasp, key grasp, center-out "
                "reaching, or random target reaching tasks. Multi-unit threshold "
                "crossings from 96-channel Utah arrays. "
                "Ma et al. 2023, eLife 84296."
            ),
        )

        extract_path = Path(extract_dir)
        mat_files = sorted(extract_path.rglob("*.mat"))
        pkl_files = sorted(extract_path.rglob("*.pkl"))

        for fpath in mat_files:
            self._process_mat_session(fpath, task, brainset_description)

        for fpath in pkl_files:
            self._process_pkl_session(fpath, task, brainset_description)

    # ------------------------------------------------------------------

    def _process_mat_session(self, fpath, task, brainset_description):
        session_id = _parse_session_id(fpath, task)
        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            return

        self.update_status(f"Processing {fpath.name}")

        xds = _load_xds_h5py(fpath)

        subject_name = session_id.split("_")[0]
        date_str = session_id.split("_")[1]

        subject = SubjectDescription(
            id=subject_name,
            species="MACACA_MULATTA",
            sex="UNKNOWN",
        )
        session = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(date_str, "%Y%m%d"),
            task="REACHING",
        )
        device = DeviceDescription(
            id=f"{subject_name}_{date_str}",
            recording_tech="UTAH_ARRAY_THRESHOLD_CROSSINGS",
        )

        spikes, units = extract_spikes(xds)
        trials, movement_phases = extract_trials(xds)

        data_kwargs = dict(
            brainset_description=brainset_description,
            subject=subject,
            session=session,
            device=device,
            spikes=spikes,
            units=units,
        )
        if trials is not None:
            data_kwargs["trials"] = trials
        if movement_phases is not None:
            data_kwargs["movement_phases"] = movement_phases

        if xds["has_cursor"]:
            cursor = extract_cursor(xds)
            domain = cursor.domain
            data_kwargs["cursor"] = cursor
        if xds["has_force"]:
            data_kwargs["hand"] = extract_hand(xds)
            domain = data_kwargs["hand"].domain

        data_kwargs["domain"] = domain
        # if xds["has_emg"]:
        #     emg, emg_channels = extract_emg(xds)
        #     data_kwargs["emg"] = emg
        #     data_kwargs["emg_channels"] = emg_channels

        data = Data(**data_kwargs)
        with h5py.File(store_path, "w") as out:
            data.to_hdf5(out)

    def _process_pkl_session(self, fpath, task, brainset_description):
        # Skip helper scripts
        if fpath.name == "load_monkey_pop_data.py":
            return

        session_id = _parse_session_id(fpath, task)
        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            return

        self.update_status(f"Processing {fpath.name}")

        # Pop pkl stores three sequential pickled objects — loading order and variable names
        # match load_monkey_pop_data.py shipped inside Pop_PG_2021.7z on Dryad
        # (https://datadryad.org/dataset/doi:10.5061/dryad.cvdncjt7n).
        #
        # Per the original script's docstring:
        #   - spike_counts: M1 firing rates already binned (50 ms) and Gaussian-smoothed
        #     (SD=100 ms) — do NOT bin/smooth again.
        #   - EMG: outliers removed, baseline subtracted, normalized — do NOT preprocess again.
        #   - force: raw isometric force (2D).
        #   - Each element in every list is one trial. Trials run from gocue to reward
        #     (no pre-gocue hold period), so hold/reach splitting is not applicable here.
        with open(fpath, "rb") as fp:
            spike_counts = pickle.load(fp)  # list of (T_i, 96) smoothed rates
            emg_trials = pickle.load(fp)  # list of (T_i, 11)
            force_trials = pickle.load(fp)  # list of (T_i, 2)

        subject_name = session_id.split("_")[0]
        date_str = session_id.split("_")[1]

        subject = SubjectDescription(
            id=subject_name,
            species="MACACA_MULATTA",
            sex="UNKNOWN",
        )
        session = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(date_str, "%Y%m%d"),
            task="REACHING",
        )
        device = DeviceDescription(
            id=f"{subject_name}_{date_str}",
            recording_tech="UTAH_ARRAY_THRESHOLD_CROSSINGS",
        )

        # Pop spike_counts are pre-smoothed firing rates (50ms bins), not timestamps.
        # We store them as binned neural rates rather than spike timestamps.
        rates_ts, rates_arr, domain = _stitch_trials(spike_counts, bin_size=0.05)
        n_units = rates_arr.shape[1]

        neural_rates = IrregularTimeSeries(
            timestamps=rates_ts,
            rates=rates_arr.astype(np.float64),
            domain="auto",
        )
        units_meta = [
            {"id": f"elec{u+1:03d}", "unit_index": u, "electrode": u + 1}
            for u in range(n_units)
        ]
        units = ArrayDict.from_dataframe(
            pd.DataFrame(units_meta), unsigned_to_long=True
        )

        data = Data(
            brainset_description=brainset_description,
            subject=subject,
            session=session,
            device=device,
            neural_rates=neural_rates,
            units=units,
            domain=domain,
        )

        if emg_trials:
            emg_ts, emg_arr, _ = _stitch_trials(emg_trials, bin_size=0.05)
            n_ch = emg_arr.shape[1]
            EMG_NAMES = [
                "APB",
                "FPB",
                "Lum",
                "PT",
                "FDS1",
                "FDP2",
                "1DI",
                "4DI",
                "EPM",
                "ECR",
                "EDC1",
            ]
            data.emg = IrregularTimeSeries(
                timestamps=emg_ts, signals=emg_arr.astype(np.float64), domain="auto"
            )
            data.emg_channels = ArrayDict.from_dataframe(
                pd.DataFrame(
                    [
                        {
                            "id": EMG_NAMES[i] if i < len(EMG_NAMES) else f"emg_{i}",
                            "channel_index": i,
                        }
                        for i in range(n_ch)
                    ]
                ),
                unsigned_to_long=True,
            )

        if force_trials:
            force_ts, force_arr, _ = _stitch_trials(force_trials, bin_size=0.05)
            data.hand = IrregularTimeSeries(
                timestamps=force_ts, force=force_arr.astype(np.float64), domain="auto"
            )

        with h5py.File(store_path, "w") as out:
            data.to_hdf5(out)


# ---------------------------------------------------------------------------
# Session ID helpers
# ---------------------------------------------------------------------------


def _parse_session_id(fpath: Path, task: str) -> str:
    """'Chewie_20160927_001.mat' or 'Pop_20210602_pg.pkl' -> 'chewie_20160927_<task>'"""
    parts = fpath.stem.split("_")
    return f"{parts[0].lower()}_{parts[1]}_{task}"


# ---------------------------------------------------------------------------
# xds .mat loading (MATLAB v7.3 HDF5)
# ---------------------------------------------------------------------------


def _load_xds_h5py(fpath: Path) -> dict:
    """Load an xds MATLAB v7.3 .mat file into a flat Python dict.

    - Scalar flags (has_cursor etc.): uint8, shape (1, 1)
    - time_frame, trial timing: float64, shape (1, T) or (1, N_trials)
    - spikes, unit_names: object cell arrays, shape (N_electrodes, 1)
    - curs_p/v/a, force, EMG: float64, shape (2, T) or (M, T)
    - trial_result: uint16, shape (1, N_trials)  [char codes, not references]
    """
    out = {}
    with h5py.File(fpath, "r") as f:
        xds = f["xds"]

        # --- Scalar flags ---
        for key in [
            "has_cursor",
            "has_force",
            "has_EMG",
        ]:
            out[key] = bool(int(xds[key][()].flat[0])) if key in xds else False

        # --- bin_width ---
        out["bin_width"] = float(xds["bin_width"][()].flat[0])

        # --- time_frame: (1, T) float64 ---
        out["time_frame"] = xds["time_frame"][()].flatten().astype(np.float64)

        # --- spikes: (N_electrodes, 1) object refs ---
        spikes_ref = xds["spikes"]
        n_electrodes = spikes_ref.shape[0]
        out["spikes"] = [
            f[spikes_ref[i, 0]][()].flatten().astype(np.float64)
            for i in range(n_electrodes)
        ]

        # --- unit_names: (N_electrodes, 1) object refs → char strings ---
        # unit_names is stored as a (N_electrodes, 1) array of HDF5 object references.
        # Each reference points to an array of ASCII char codes (MATLAB cell-array encoding).
        # Follow each ref and decode char codes → string, e.g. [101,108,...] → "elec001".
        if "unit_names" in xds:
            names_ref = xds["unit_names"]
            out["unit_names"] = [
                "".join(chr(c) for c in f[names_ref[i, 0]][()].flatten())
                for i in range(names_ref.shape[0])
            ]
        else:
            raise ValueError("unit_names not found in xds")
        # else:
        #     out["unit_names"] = [f"elec{i+1:03d}" for i in range(n_electrodes)]

        # --- 2D continuous arrays: (D, T) float64 → transpose to (T, D) ---
        for key in [
            "curs_p",
            "curs_v",
            "curs_a",
            "force",
            "EMG",
        ]:
            out[key] = xds[key][()].T.astype(np.float64) if key in xds else None

        # EMG channel names (not always present)
        if "emg_names" in xds:
            out["emg_names"] = xds["emg_names"][()].flatten().astype(str)

        # --- Trial timing: (1, N_trials) float64 ---
        for key in [
            "trial_start_time",
            "trial_end_time",
        ]:
            out[key] = xds[key][()].flatten().astype(np.float64) if key in xds else None

        # trial_gocue_time may be (1, N_trials) for CO or (K, N_trials) for RT.
        # Take the first row as the primary go-cue; store all rows for RT.
        if "trial_gocue_time" in xds:
            gc = xds["trial_gocue_time"][()].astype(np.float64)  # (K, N_trials)
            out["trial_gocue_time"] = gc[0, :]  # first go-cue per trial
            out["trial_gocue_time_all"] = gc.T  # (N_trials, K) all go-cues
        else:
            out["trial_gocue_time"] = None
            out["trial_gocue_time_all"] = None

        # --- trial_result: (1, N_trials) uint16 char codes ---
        if "trial_result" in xds:
            vals = xds["trial_result"][()].flatten()
            out["trial_result"] = np.array([chr(int(v)) for v in vals])
        else:
            out["trial_result"] = None

        # --- trial_target_dir: (1, N_trials) float64 ---
        out["trial_target_dir"] = (
            xds["trial_target_dir"][()].flatten().astype(np.float64)
            if "trial_target_dir" in xds
            else None
        )

    # make all keys lowercase
    out = {k.lower(): v for k, v in out.items()}

    return out


# ---------------------------------------------------------------------------
# Extraction helpers for .mat sessions
# ---------------------------------------------------------------------------


def extract_spikes(xds: dict):
    """Build spike IrregularTimeSeries from the xds spike cell array.

    Each electrode's spike timestamps are concatenated and sorted.
    """
    spike_list = xds["spikes"]
    unit_names = xds["unit_names"]
    n_units = len(spike_list)

    all_ts, all_idx = [], []
    unit_meta = []

    for u, (st, name) in enumerate(zip(spike_list, unit_names)):
        if len(st) > 0:
            all_ts.append(st)
            all_idx.append(np.full(len(st), u, dtype=np.int64))
        unit_meta.append({"id": str(name), "unit_index": u, "electrode": u + 1})

    timestamps = (
        np.concatenate(all_ts).astype(np.float64)
        if all_ts
        else np.array([], dtype=np.float64)
    )
    unit_index = np.concatenate(all_idx) if all_idx else np.array([], dtype=np.int64)
    sort_idx = np.argsort(timestamps, kind="stable")

    spikes = IrregularTimeSeries(
        timestamps=timestamps[sort_idx],
        unit_index=unit_index[sort_idx],
        domain="auto",
    )
    units = ArrayDict.from_dataframe(pd.DataFrame(unit_meta), unsigned_to_long=True)
    return spikes, units


def extract_cursor(xds: dict) -> RegularTimeSeries:
    timestamps = xds["time_frame"]

    sampling_rate = (
        round(1.0 / np.median(np.diff(timestamps)), 5)
        if np.median(np.diff(timestamps)) > 0
        else 1.0
    )

    return RegularTimeSeries(
        sampling_rate=sampling_rate,
        pos=xds["curs_p"],
        vel=xds["curs_v"],
        acc=xds["curs_a"],
        domain_start=timestamps[0],
        domain="auto",
    )
    # return IrregularTimeSeries(
    #     timestamps=xds["time_frame"],
    #     pos=xds["curs_p"],
    #     vel=xds["curs_v"],
    #     acc=xds["curs_a"],
    #     domain="auto",
    # )


def extract_hand(xds: dict) -> IrregularTimeSeries:
    force = xds["force"]
    if force.ndim == 1:
        force = force[:, np.newaxis]
    return IrregularTimeSeries(
        timestamps=xds["time_frame"],
        force=force,
        domain="auto",
    )


# def extract_emg(xds: dict):
#     emg_arr = xds["emg"]
#     if emg_arr.ndim == 1:
#         emg_arr = emg_arr[:, np.newaxis]
#     n_ch = emg_arr.shape[1]
#     emg = IrregularTimeSeries(
#         timestamps=xds["time_frame"],
#         signals=emg_arr,
#         domain="auto",
#     )
#     names = xds.get("emg_names") or [f"emg_{i}" for i in range(n_ch)]
#     emg_channels = ArrayDict.from_dataframe(
#         pd.DataFrame([{"id": str(names[i]), "channel_index": i} for i in range(n_ch)]),
#         unsigned_to_long=True,
#     )
#     return emg, emg_channels


def extract_trials(xds: dict):
    # trial_result codes (from limblab/xds xds_python/readme.md):
    #   'R' = Rewarded  — successful trial, monkey received juice reward
    #   'F' = Failed    — monkey did not reach the target in time
    #   'A' = Aborted   — monkey broke fixation or moved before go cue
    # Source: https://github.com/limblab/xds/blob/master/xds_python/readme.md
    start = xds.get("trial_start_time")
    end = xds.get("trial_end_time")
    go_cue = xds.get("trial_gocue_time")
    result = xds.get("trial_result")
    target_dir = xds.get("trial_target_dir")

    if start is None or end is None:
        return None, None

    n = len(start)
    if go_cue is None:
        go_cue = np.full(n, np.nan)
    if result is None:
        result = np.array(["" for _ in range(n)])
    # target_dir is the reach target angle (radians) per trial — present in CO and RT
    # tasks but absent in isometric wrist / grasp tasks (no spatial target direction).
    # len check handles RT sessions where xds may store a ragged or mismatched array
    # (e.g. multiple go-cues can shift the effective trial count vs target count).
    if target_dir is None or len(target_dir) != n:
        target_dir = np.full(n, np.nan)

    # A trial is valid if it was rewarded and has a sensible duration.
    # Go cue is not required for validity — it's only needed for movement phases.
    is_valid = (result == "R") & ((end - start) > 0.1)

    trial_kwargs = dict(
        start=start.astype(np.float64),
        end=end.astype(np.float64),
        go_cue_time=go_cue.astype(np.float64),
        target_dir=target_dir.astype(np.float64),
        result=result,
        is_valid=is_valid,
    )
    gocue_all = xds.get("trial_gocue_time_all")
    if gocue_all is not None and gocue_all.shape[1] > 1:
        trial_kwargs["go_cue_time_all"] = gocue_all.astype(np.float64)
    trials = Interval(**trial_kwargs)

    # movement_phases only for valid trials that also have a go cue recorded,
    # since hold/reach split requires knowing when the go cue fired.
    valid_with_gocue = trials.select_by_mask(is_valid & ~np.isnan(go_cue))
    movement_phases = None
    if len(valid_with_gocue.start) > 0:
        movement_phases = Data(
            hold_period=Interval(
                start=valid_with_gocue.start, end=valid_with_gocue.go_cue_time
            ),
            reach_period=Interval(
                start=valid_with_gocue.go_cue_time, end=valid_with_gocue.end
            ),
            domain="auto",
        )
    return trials, movement_phases


# ---------------------------------------------------------------------------
# Pop pkl helpers
# ---------------------------------------------------------------------------


def _stitch_trials(trial_list, bin_size: float, gap: float = 1.0):
    """Concatenate trial arrays separated by `gap` seconds.

    Returns (timestamps, signals, domain).
    """
    all_ts, all_signals = [], []
    trial_starts, trial_ends = [], []
    t = 0.0

    for arr in trial_list:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        n_bins = arr.shape[0]
        ts = t + np.arange(n_bins) * bin_size
        trial_starts.append(t)
        trial_ends.append(t + n_bins * bin_size)
        all_ts.append(ts)
        all_signals.append(arr)
        t += n_bins * bin_size + gap

    timestamps = np.concatenate(all_ts)
    signals = np.vstack(all_signals)
    domain = Interval(
        start=np.array([trial_starts[0]], dtype=np.float64),
        end=np.array([trial_ends[-1]], dtype=np.float64),
    )
    return timestamps, signals, domain
