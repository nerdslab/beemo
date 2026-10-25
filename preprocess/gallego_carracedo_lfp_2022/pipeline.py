# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "requests==2.32.3",
# ]
# ///

import datetime
import os
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests

from torch_brain.data import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
)
from torch_brain.pipeline import BrainsetPipeline
from torch_brain.data import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)

DRYAD_BASE = "https://datadryad.org"
DATASET_DOI = "doi%3A10.5061%2Fdryad.xd2547dkt"

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


# ---------------------------------------------------------------------------
# Dryad API helpers
# ---------------------------------------------------------------------------


def _get_access_token() -> str:
    """Exchange client credentials for a Bearer access token (valid 10 hours)."""
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
    """Fetch file list from the latest version of this Dryad dataset."""
    versions_url = f"{DRYAD_BASE}/api/v2/datasets/{DATASET_DOI}/versions"
    resp = requests.get(versions_url, timeout=30)
    resp.raise_for_status()
    versions = resp.json()["_embedded"]["stash:versions"]
    files_href = versions[-1]["_links"]["stash:files"]["href"]

    resp = requests.get(f"{DRYAD_BASE}{files_href}?per_page=100", timeout=30)
    resp.raise_for_status()
    return resp.json()["_embedded"]["stash:files"]


def _parse_session_id(filename: str) -> str:
    """'Chewie_CO_20150313.mat' -> 'chewie_20150313_center_out_reaching'"""
    parts = Path(filename).stem.split("_")
    return f"{parts[0].lower()}_{parts[2]}_center_out_reaching"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline(BrainsetPipeline):
    brainset_id = "gallego_carracedo_lfp_2022"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        rows = []
        for f in _get_dryad_files():
            filename = f["path"]
            if not filename.endswith(".mat"):
                continue
            download_url = DRYAD_BASE + f["_links"]["stash:download"]["href"]
            rows.append(
                {
                    "session_id": _parse_session_id(filename),
                    "filename": filename,
                    "url": download_url,
                }
            )
        return pd.DataFrame(rows).set_index("session_id")

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        dest = self.raw_dir / manifest_item.filename
        if dest.exists() and not self.args.redownload:
            self.update_status("Already downloaded")
            return dest

        token = _get_access_token()
        resp = requests.get(
            manifest_item.url,
            headers={"Authorization": f"Bearer {token}"},
            stream=True,
            timeout=5,
        )
        with open(dest, "wb") as fh:
            fh.write(resp.content)

        return dest

    def process(self, fpath):
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        brainset_description = BrainsetDescription(
            id="gallego_carracedo_lfp_2022",
            origin_version="dryad/10.5061/dryad.xd2547dkt",
            derived_version="1.0.0",
            source="https://datadryad.org/dataset/doi:10.5061/dryad.xd2547dkt",
            description=(
                "Spikes + LFPs from Utah arrays in M1, PMd, and Area 2 of "
                "four rhesus macaques during a center-out delayed reaching task. "
                "Gallego-Carracedo et al. 2022, eLife 73155."
            ),
        )

        session_id = self._asset_id
        store_path = self.processed_dir / f"{session_id}.h5"

        date_str = session_id.split("_")[1]
        subject_name = session_id.split("_")[0].lower()
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        subject = SubjectDescription(
            id=subject_name,
            species="UNKNOWN",
            sex="UNKNOWN",
        )

        session = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(date_str, "%Y%m%d"),
            task="REACHING",
        )

        self.update_status("Loading .mat")
        f = h5py.File(fpath, "r")
        td = f["trial_data"]

        brain_areas = _get_area_keys(td)
        num_trials = td["vel"].shape[0]

        self.update_status("Building absolute time axis")
        trial_offsets, bin_size = _build_absolute_times(f, td, num_trials)

        self.update_status("Extracting spikes")
        timestamps, unit_index, counts, unit_meta = extract_spikes_and_units(
            f, td, trial_offsets, bin_size, brain_areas, num_trials
        )

        binned_spikes = IrregularTimeSeries(
            timestamps=timestamps.astype(np.float64),
            unit_index=unit_index,
            count=counts.astype(np.int64),
            domain="auto",
        )
        units = ArrayDict.from_dataframe(pd.DataFrame(unit_meta), unsigned_to_long=True)

        self.update_status("Extracting behavior")
        timestamps, hand_vel = extract_behavior(
            f, td, trial_offsets, bin_size, num_trials
        )

        sampling_rate = (
            round(1.0 / np.median(np.diff(timestamps)), 5)
            if np.median(np.diff(timestamps)) > 0
            else 1.0
        )

        hand = IrregularTimeSeries(
            timestamps=timestamps,
            vel=hand_vel,
            domain="auto",
        )

        self.update_status("Extracting LFP")
        lfp_timestamps, lfp_signals, lfp_channel_meta = extract_lfp(
            f, td, trial_offsets, bin_size, brain_areas, num_trials
        )

        lfp = IrregularTimeSeries(
            timestamps=lfp_timestamps.astype(np.float64),
            signals=lfp_signals,
            domain="auto",
        )
        lfp_channels = ArrayDict.from_dataframe(
            pd.DataFrame(lfp_channel_meta), unsigned_to_long=True
        )

        self.update_status("Extracting trials")
        trials, movement_phases = extract_trials(
            f, td, trial_offsets, bin_size, num_trials
        )

        f.close()

        data = Data(
            brainset_description=brainset_description,
            subject=subject,
            session=session,
            # neural activity
            binned_spikes=binned_spikes,
            units=units,
            # LFP
            lfp=lfp,
            lfp_channels=lfp_channels,
            # stimuli and behavior
            trials=trials,
            movement_phases=movement_phases,
            hand=hand,
            # domain
            domain=hand.domain,
        )

        self.update_status("Storing")
        with h5py.File(store_path, "w") as out:
            data.to_hdf5(out)

        self.update_status("Done")


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _deref_scalar(f, dataset, i: int) -> float:
    return float(f[dataset[i, 0]][()].flat[0])


def _get_area_keys(td) -> list:
    return [k.replace("_spikes", "") for k in td.keys() if k.endswith("_spikes")]


def _build_absolute_times(f, td, num_trials):
    """Assign each trial an absolute start time, separated by 1s gaps."""

    bin_size = _deref_scalar(f, td["bin_size"], 0)  # seconds per bin (0.01s)
    for i in range(1, num_trials):
        assert (
            _deref_scalar(f, td["bin_size"], i) == bin_size
        ), f"bin_size differs at trial {i}"
    offsets = np.empty(num_trials)
    t = 0.0
    for i in range(num_trials):
        offsets[i] = t  # record this trial's start time
        n_bins = f[td["vel"][i, 0]].shape[1]  # vel is (2, N_bins)
        t += n_bins * bin_size + 1.0  # advance by trial duration + 1s gap
    return offsets, bin_size


def extract_spikes_and_units(f, td, trial_offsets, bin_size, brain_areas, num_trials):
    """Convert per-trial binned spike counts into a single IrregularTimeSeries.

    For each brain area, builds unit metadata from the unit guide, then
    iterates over trials converting the (N_units x N_bins) count matrix into
    spike timestamps. Each non-zero bin produces one entry at the bin center
    time, with the spike count stored as a separate field.

    Returns:
        timestamps: sorted bin-start times, one per non-zero (unit, bin) pair.
        unit_index: corresponding unit indices.
        counts: number of spikes in each bin.
        units: ArrayDict of unit metadata (id, electrode, unit_number, area, type).
    """

    all_timestamps = []
    all_unit_idx = []
    all_counts = []
    unit_meta = []
    unit_offset = 0  # needed to map unit indices to area wise global indices

    for area in brain_areas:
        guide = f[td[f"{area}_unit_guide"][0, 0]][()]  # (2, N_units)
        n_units = guide.shape[1]
        for i in range(1, num_trials):
            assert np.array_equal(
                guide, f[td[f"{area}_unit_guide"][i, 0]][()]
            ), f"unit_guide differs at trial {i} for area {area}"

        # Build unit metadata
        for u in range(n_units):
            # unit_guide is (N_units, 2) in MATLAB: col 1 = electrode, col 2 = unit number.
            # h5py transposes it to (2, N_units), so guide[0] = electrode, guide[1] = unit number.
            # Ref: https://github.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/blob/master/MainAnalysis/unit_correlation.m
            unit_meta.append(
                {
                    "id": f"{area}_elec{guide[0, u]}_unit{guide[1, u]}",
                    "unit_index": unit_offset + u,
                    "electrode": int(guide[0, u]),
                    "unit_number": int(guide[1, u]),
                    "area": area,
                }
            )

        # Extract spikes
        for i in range(num_trials):
            spike_counts = f[td[f"{area}_spikes"][i, 0]][()]  # (N_units, N_bins)
            assert (
                spike_counts.shape[0] == n_units
            ), f"spike matrix unit dim {spike_counts.shape[0]} != guide n_units {n_units} at trial {i}, area {area}"
            nz_units, nz_bins = np.where(spike_counts > 0)
            ts = trial_offsets[i] + nz_bins * bin_size
            all_timestamps.append(ts)
            all_unit_idx.append((unit_offset + nz_units).astype(np.int64))
            all_counts.append(spike_counts[nz_units, nz_bins])

        unit_offset += n_units

    timestamps = np.concatenate(all_timestamps)
    unit_index = np.concatenate(all_unit_idx)
    counts = np.concatenate(all_counts)
    sort_idx = np.argsort(timestamps, kind="stable")

    return timestamps[sort_idx], unit_index[sort_idx], counts[sort_idx], unit_meta


def extract_behavior(f, td, trial_offsets, bin_size, num_trials):
    all_timestamps = []
    all_hand_vel = []

    for i in range(num_trials):
        vel = f[td["vel"][i, 0]][()]  # (2, N_bins)
        n_bins = vel.shape[1]
        t_starts = trial_offsets[i] + np.arange(n_bins) * bin_size
        all_timestamps.append(t_starts)
        all_hand_vel.append(vel.T)  # (N_bins, 2)

    return np.concatenate(all_timestamps), np.vstack(all_hand_vel).astype(np.float64)


def extract_lfp(f, td, trial_offsets, bin_size, brain_areas, num_trials):
    """Extract LFP signals across all areas into a single dense time series.

    lfp_guide is (S, 3) in MATLAB → (3, S) after h5py transpose:
        guide[0] = channel, guide[1] = freq_low, guide[2] = freq_high
    lfp data is (T, S) in MATLAB → (S, T) after h5py transpose.

    Returns:
        timestamps: (total_bins,) bin-start times, same grid as hand velocity.
        signals: (total_bins, n_total_signals) LFP values, areas concatenated.
        lfp_channel_meta: list of dicts with channel, freq_low, freq_high, area.
    """
    all_timestamps = []
    all_signals = []
    lfp_channel_meta = []
    channel_offset = 0

    for area in brain_areas:
        guide = f[td[f"{area}_lfp_guide"][0, 0]][()]  # (3, S)
        n_signals = guide.shape[1]
        for i in range(1, num_trials):
            assert np.array_equal(
                guide, f[td[f"{area}_lfp_guide"][i, 0]][()]
            ), f"lfp_guide differs at trial {i} for area {area}"

        for s in range(n_signals):
            lfp_channel_meta.append(
                {
                    "channel_index": channel_offset + s,
                    "channel": int(guide[0, s]),
                    "freq_low": float(guide[1, s]),
                    "freq_high": float(guide[2, s]),
                    "area": area,
                }
            )

        for i in range(num_trials):
            lfp = f[td[f"{area}_lfp"][i, 0]][()]  # (n_signals, n_bins)
            assert lfp.shape[0] == n_signals, (
                f"lfp signal dim {lfp.shape[0]} != guide n_signals {n_signals} "
                f"at trial {i}, area {area}"
            )
            n_bins = lfp.shape[1]
            if area == brain_areas[0]:
                t_starts = trial_offsets[i] + np.arange(n_bins) * bin_size
                all_timestamps.append(t_starts)
            assert n_bins == len(all_timestamps[i]), (
                f"n_bins mismatch at trial {i}: area {area} has {n_bins} bins "
                f"but {brain_areas[0]} has {len(all_timestamps[i])}"
            )

            all_signals.append(lfp.T)  # (n_bins, n_signals)

        channel_offset += n_signals

    timestamps = np.concatenate(all_timestamps)
    # all_signals is area-first: [area0_trial0, area0_trial1, ..., area1_trial0, ...]
    # vstack each area's trials, then hstack areas column-wise
    signals = np.hstack(
        [
            np.vstack(all_signals[i * num_trials : (i + 1) * num_trials])
            for i in range(len(brain_areas))
        ]
    ).astype(np.float64)

    return timestamps, signals, lfp_channel_meta


def extract_trials(f, td, trial_offsets, bin_size, num_trials):

    def get_idx_abs(key):
        vals = np.array([_deref_scalar(f, td[key], i) for i in range(num_trials)])
        return np.where(np.isnan(vals), np.nan, trial_offsets + (vals - 1) * bin_size)

    tgt_on = get_idx_abs("idx_tgtOnTime")
    go_cue = get_idx_abs("idx_goCueTime")
    move_on = get_idx_abs("idx_movement_on")
    end_time = get_idx_abs("idx_endTime")
    tgt_dir = np.array([_deref_scalar(f, td["tgtDir"], i) for i in range(num_trials)])

    trial_ends = np.array(
        [
            trial_offsets[i] + f[td["vel"][i, 0]].shape[1] * bin_size
            for i in range(num_trials)
        ]
    )
    end_time = np.where(np.isnan(end_time), trial_ends, end_time)

    is_valid = ~np.isnan(tgt_on) & ~np.isnan(go_cue) & ~np.isnan(move_on)

    trials = Interval(
        start=trial_offsets,
        end=trial_ends,
        tgt_on_time=tgt_on,
        go_cue_time=go_cue,
        move_on_time=move_on,
        end_time=end_time,
        target_dir=tgt_dir,
        is_valid=is_valid,
    )

    valid_trials = trials.select_by_mask(is_valid)
    movement_phases = Data(
        hold_period=Interval(
            start=valid_trials.tgt_on_time,
            end=valid_trials.go_cue_time,
        ),
        reach_period=Interval(
            start=valid_trials.move_on_time,
            end=valid_trials.end_time,
        ),
        domain="auto",
    )

    return trials, movement_phases
