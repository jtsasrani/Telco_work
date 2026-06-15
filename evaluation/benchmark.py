#!/usr/bin/env python3
"""
================================================================================
 5G Core/RAN Intelligent Diagnostic Engine — Evaluation & Benchmark Suite
================================================================================
 Target Hardware : AMD Instinct™ MI300X (192 GB HBM3, ROCm 7.0)
 Model           : Llama-3.3-70B-Instruct (4-bit) + telco_expert_master LoRA
 Framework       : Transformers 4.x + PEFT + SafeTensors + PyTorch 2.10+rocm7.0
 Purpose         : Quantitative comparison of BASE vs FINE-TUNED model on 30
                   curated telecom diagnostic queries across 7 failure categories.
 Outputs         : JSON results, console summary table, publication-quality PNGs
================================================================================
 Usage:
   python evaluation/benchmark.py              # Full 30-query benchmark
   python evaluation/benchmark.py --quick      # Fast 5-query smoke test
================================================================================
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

# ── Silence noisy loggers ────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.4.2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1 — CONSTANTS & PATHS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
LORA_ADAPTER_DIR = PROJECT_ROOT / "telco_expert_master_integrated_lora"
BASE_MODEL_ID = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

# AMD-branded color palette for publication-quality charts
AMD_BG        = "#0a0a0f"
AMD_FG_TEXT   = "#e0e0e0"
AMD_RED       = "#ED1C24"
AMD_RED_LIGHT = "#FF6B6F"
AMD_GREY      = "#666666"
AMD_GREY_LIGHT = "#999999"
AMD_GRID      = "#1a1a2e"
AMD_PANEL     = "#111122"

# ── Generation hyper-parameters (identical for both models) ──────────────────
GENERATION_CONFIG = dict(
    max_new_tokens=700,
    temperature=0.4,
    top_p=0.85,
    repetition_penalty=1.2,
    do_sample=True,
    use_cache=True,
)

# ── System prompt (canonical — used for both models) ─────────────────────────
SYSTEM_PROMPT = (
    "You are an expert autonomous Tier-3 telecom network engineering system. "
    "Analyze the input and provide a structured technical diagnosis following "
    "this format:\n"
    "### Low-Level Protocol Root Cause Trace (3GPP Metrics):\n"
    "1. **[Analysis Area]**:\n"
    "   - [Detailed protocol analysis with 3GPP spec references]\n"
    "### Recommended Corrective Actions:\n"
    "1. [Specific technical recommendation]"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2 — VALID 3GPP SPECIFICATION REGISTRY (60+ real specs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID_3GPP_SPECS: set[str] = {
    # ── 5G NR Radio (38 series) ──
    "TS 38.101", "TS 38.104", "TS 38.133", "TS 38.201", "TS 38.202",
    "TS 38.211", "TS 38.212", "TS 38.213", "TS 38.214", "TS 38.215",
    "TS 38.300", "TS 38.304", "TS 38.306", "TS 38.321", "TS 38.322",
    "TS 38.323", "TS 38.331", "TS 38.401", "TS 38.410", "TS 38.411",
    "TS 38.412", "TS 38.413", "TS 38.414", "TS 38.415", "TS 38.420",
    "TS 38.421", "TS 38.422", "TS 38.423", "TS 38.424", "TS 38.425",
    "TS 38.455", "TS 38.460", "TS 38.470", "TS 38.471", "TS 38.472",
    "TS 38.473", "TS 38.474", "TS 38.475",
    # ── 5G Core (23 series) ──
    "TS 23.040", "TS 23.122", "TS 23.228", "TS 23.401", "TS 23.501",
    "TS 23.502", "TS 23.503", "TS 23.316", "TS 23.288",
    # ── NAS / Security (24 series) ──
    "TS 24.008", "TS 24.301", "TS 24.501", "TS 24.502",
    # ── LTE Radio / Core legacy (36 series) ──
    "TS 36.211", "TS 36.212", "TS 36.213", "TS 36.214", "TS 36.300",
    "TS 36.304", "TS 36.306", "TS 36.321", "TS 36.322", "TS 36.323",
    "TS 36.331", "TS 36.401", "TS 36.410", "TS 36.413", "TS 36.423",
    # ── IMS / VoNR (26 / 24 series) ──
    "TS 24.229", "TS 24.228", "TS 26.114", "TS 26.346",
    # ── GTP / User Plane (29 series) ──
    "TS 29.244", "TS 29.274", "TS 29.281", "TS 29.500", "TS 29.501",
    "TS 29.502", "TS 29.503", "TS 29.504", "TS 29.505", "TS 29.507",
    "TS 29.508", "TS 29.509", "TS 29.510", "TS 29.511", "TS 29.512",
    "TS 29.513", "TS 29.514", "TS 29.518", "TS 29.571",
    # ── Management / Performance (28 / 32 series) ──
    "TS 28.310", "TS 28.531", "TS 28.532", "TS 28.541", "TS 28.552",
    "TS 28.554", "TS 32.421", "TS 32.422", "TS 32.425",
    # ── Technical Reports ──
    "TR 38.801", "TR 38.802", "TR 38.803", "TR 38.804", "TR 38.811",
    "TR 38.812", "TR 38.821", "TR 38.889", "TR 38.900", "TR 38.901",
    "TR 38.912", "TR 38.913", "TR 23.700", "TR 23.786",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3 — 30 CURATED TELECOM DIAGNOSTIC TEST QUERIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST_QUERIES: list[dict[str, Any]] = [
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 1: Handover Failures (inter-gNB, inter-frequency)
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "HO-001",
        "category": "Handover Failures",
        "difficulty": "easy",
        "input": (
            "Subscriber reports frequent call drops while driving on the highway "
            "between two gNB coverage areas. UE logs show RRC re-establishment "
            "attempts with cause T304 timer expiry. RSRP drops below -120 dBm "
            "in the handover region."
        ),
        "expected_specs": ["TS 38.331", "TS 38.300", "TS 38.133"],
        "expected_keywords": [
            "RRCReconfiguration", "T304", "MeasurementReport",
            "handover", "RSRP", "A3 event"
        ],
    },
    {
        "id": "HO-002",
        "category": "Handover Failures",
        "difficulty": "medium",
        "input": (
            "Inter-frequency handover from n78 (3.5 GHz) to n257 (mmWave) fails "
            "consistently. UE sends MeasurementReport with B1 event but gNB does "
            "not initiate RRCReconfiguration. Measurement gap configuration may be "
            "incorrect. FR2 SSB beam index not being reported."
        ),
        "expected_specs": ["TS 38.331", "TS 38.133", "TS 38.215"],
        "expected_keywords": [
            "RRCReconfiguration", "MeasurementReport", "B1 event",
            "measGapConfig", "SSB", "inter-frequency", "FR2"
        ],
    },
    {
        "id": "HO-003",
        "category": "Handover Failures",
        "difficulty": "hard",
        "input": (
            "Conditional handover (CHO) execution failure in dense urban macro-to-small-cell "
            "scenario. UE evaluates CHO conditions but T304 expires before Random Access on "
            "target PSCell succeeds. CG-ConfigInfo in RRCReconfigurationComplete shows "
            "SCG failure with cause 'reconfigurationFailure'. X2/Xn handover preparation "
            "succeeds but SN Status Transfer is delayed."
        ),
        "expected_specs": ["TS 38.331", "TS 38.300", "TS 38.423", "TS 38.401"],
        "expected_keywords": [
            "ConditionalHandover", "T304", "RRCReconfiguration",
            "PSCell", "CG-ConfigInfo", "SCG failure",
            "SN Status Transfer", "Xn", "RACH"
        ],
    },
    {
        "id": "HO-004",
        "category": "Handover Failures",
        "difficulty": "medium",
        "input": (
            "During inter-gNB handover, UE completes RACH procedure on target cell but "
            "fails to receive RRCReconfigurationComplete acknowledgment. Path switch at "
            "AMF fails with NGAP cause 'unknown-targetID'. Tracking Area Update rejected "
            "on the target side. Source gNB retains UE context causing resource leakage."
        ),
        "expected_specs": ["TS 38.331", "TS 38.413", "TS 23.502", "TS 24.501"],
        "expected_keywords": [
            "RRCReconfiguration", "RACH", "NGAP",
            "PathSwitchRequest", "TrackingAreaUpdate",
            "AMF", "UE context release"
        ],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 2: VoNR / IMS Call Setup Failures
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "VONR-001",
        "category": "VoNR/IMS Failures",
        "difficulty": "easy",
        "input": (
            "VoNR calls fail at SIP INVITE stage. UE registers on IMS successfully "
            "but 180 Ringing is never received. P-CSCF returns 503 Service Unavailable. "
            "QoS flow for voice (5QI=1) not established on the gNB side."
        ),
        "expected_specs": ["TS 24.229", "TS 23.228", "TS 23.501", "TS 38.331"],
        "expected_keywords": [
            "SIP INVITE", "P-CSCF", "5QI", "QoS flow",
            "IMS registration", "503", "VoNR", "PDU Session"
        ],
    },
    {
        "id": "VONR-002",
        "category": "VoNR/IMS Failures",
        "difficulty": "hard",
        "input": (
            "VoNR call setup succeeds but audio is one-way. RTP media path established "
            "via SDP negotiation shows mismatched codec (EVS vs AMR-WB). UPF downlink "
            "GTP-U tunnel delivers packets but UE reports ROHC decompression failures on "
            "DRB mapped to 5QI=1. PDCP COUNT wraparound suspected on the voice bearer."
        ),
        "expected_specs": ["TS 24.229", "TS 26.114", "TS 38.323", "TS 29.244", "TS 38.322"],
        "expected_keywords": [
            "RTP", "SDP", "EVS", "AMR-WB", "UPF", "GTP-U",
            "ROHC", "DRB", "5QI", "PDCP", "QoS flow"
        ],
    },
    {
        "id": "VONR-003",
        "category": "VoNR/IMS Failures",
        "difficulty": "medium",
        "input": (
            "Emergency VoNR call (IMS Emergency PDU Session) fails establishment. "
            "UE sends PDU Session Establishment Request with request type 'emergency' "
            "but SMF rejects with 5GSM cause #31 'request rejected, unspecified'. "
            "The AMF did not include emergency service support indication in "
            "Registration Accept. SUPI-based emergency services not configured."
        ),
        "expected_specs": ["TS 24.501", "TS 23.501", "TS 23.502", "TS 24.229"],
        "expected_keywords": [
            "PDU Session Establishment", "emergency", "SMF",
            "5GSM cause", "AMF", "Registration Accept",
            "SUPI", "IMS emergency"
        ],
    },
    {
        "id": "VONR-004",
        "category": "VoNR/IMS Failures",
        "difficulty": "medium",
        "input": (
            "EPS Fallback for voice fails. UE receives RRCRelease with redirectedCarrierInfo "
            "to LTE band n3 but IRAT redirection timer expires. UE re-selects NR instead. "
            "SRVCC handover not attempted. Network does not support N26 interface between "
            "AMF and MME."
        ),
        "expected_specs": ["TS 38.331", "TS 23.502", "TS 23.401", "TS 36.331"],
        "expected_keywords": [
            "EPS Fallback", "RRCRelease", "redirectedCarrierInfo",
            "IRAT", "SRVCC", "N26", "AMF", "MME"
        ],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 3: 5G SA/NSA Registration Issues
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "REG-001",
        "category": "5G Registration Issues",
        "difficulty": "easy",
        "input": (
            "UE fails 5G SA initial registration. NAS Registration Request sent but "
            "Authentication fails with 5GMM cause #3 'Illegal UE'. SUCI concealment "
            "using ECIES Profile A. AUSF returns authentication vector mismatch from UDM."
        ),
        "expected_specs": ["TS 24.501", "TS 23.501", "TS 33.501"],
        "expected_keywords": [
            "Registration Request", "NAS", "Authentication",
            "5GMM cause", "SUCI", "AUSF", "UDM", "ECIES"
        ],
    },
    {
        "id": "REG-002",
        "category": "5G Registration Issues",
        "difficulty": "medium",
        "input": (
            "NSA EN-DC setup fails. UE camps on LTE anchor and receives SCG addition "
            "via RRCConnectionReconfiguration with nr-Config. However, RACH on NR PSCell "
            "fails — msg3 not received at gNB. SgNB Addition Request Acknowledge sent "
            "by SN but X2-U transport bearer setup delayed. MCG split bearer not established."
        ),
        "expected_specs": ["TS 36.331", "TS 37.340", "TS 38.321", "TS 36.423"],
        "expected_keywords": [
            "EN-DC", "SCG addition", "RRCConnectionReconfiguration",
            "PSCell", "RACH", "SgNB", "X2", "MCG", "split bearer"
        ],
    },
    {
        "id": "REG-003",
        "category": "5G Registration Issues",
        "difficulty": "hard",
        "input": (
            "5G SA Registration Accept received but Service Request procedure fails "
            "when UE transitions from CM-IDLE to CM-CONNECTED. AMF sends Service Reject "
            "with 5GMM cause #7 '5GS services not allowed'. UE has valid 5G-GUTI but "
            "NGAP Initial UE Message contains mismatched AMF Set ID. AMF load balancing "
            "redirects to wrong AMF instance. NAS Security Mode Command not initiated."
        ),
        "expected_specs": ["TS 24.501", "TS 23.501", "TS 23.502", "TS 38.413"],
        "expected_keywords": [
            "Service Request", "CM-IDLE", "CM-CONNECTED",
            "Service Reject", "5GMM cause", "5G-GUTI",
            "NGAP", "AMF", "NAS Security Mode"
        ],
    },
    {
        "id": "REG-004",
        "category": "5G Registration Issues",
        "difficulty": "easy",
        "input": (
            "UE registration rejected with 5GMM cause #11 'PLMN not allowed'. "
            "Subscriber has international roaming package but visited PLMN not in "
            "the equivalent PLMN list. UE enters limited service state. Manual PLMN "
            "selection shows the target PLMN but automatic selection skips it."
        ),
        "expected_specs": ["TS 24.501", "TS 23.122", "TS 23.501"],
        "expected_keywords": [
            "Registration Reject", "5GMM cause", "PLMN",
            "roaming", "equivalent PLMN", "limited service",
            "PLMN selection"
        ],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 4: Beam Management & MIMO Problems
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "BEAM-001",
        "category": "Beam Management/MIMO",
        "difficulty": "easy",
        "input": (
            "mmWave (n257) coverage issues in indoor office. UE reports beam failure "
            "events every 30 seconds. SSB RSRP measurements show all beams below "
            "-130 dBm. Beam failure recovery via RACH on contention-based resources "
            "succeeds but new beam RSRP decays within seconds."
        ),
        "expected_specs": ["TS 38.321", "TS 38.213", "TS 38.215"],
        "expected_keywords": [
            "beam failure", "SSB", "RSRP", "beam recovery",
            "RACH", "mmWave", "contention-based", "BFR"
        ],
    },
    {
        "id": "BEAM-002",
        "category": "Beam Management/MIMO",
        "difficulty": "hard",
        "input": (
            "Massive MIMO 64T64R gNB shows degraded MU-MIMO performance. Downlink "
            "rank indicator (RI) capped at rank-2 despite 4-layer capable UEs. "
            "CSI-RS resource configuration uses only 8 ports instead of 32. PMI "
            "reporting configured as Type I Single Panel but Type II needed for "
            "high-resolution spatial multiplexing. SRS resources insufficient for "
            "uplink channel sounding."
        ),
        "expected_specs": ["TS 38.214", "TS 38.211", "TS 38.331", "TS 38.212"],
        "expected_keywords": [
            "MIMO", "MU-MIMO", "rank indicator", "CSI-RS",
            "PMI", "Type I", "Type II", "SRS", "beamforming",
            "spatial multiplexing", "codebook"
        ],
    },
    {
        "id": "BEAM-003",
        "category": "Beam Management/MIMO",
        "difficulty": "medium",
        "input": (
            "L1-RSRP based beam management shows stale beam indications. UE reports "
            "CSI-RS based L1-RSRP for P2 procedure but gNB TCI state activation via "
            "MAC CE takes too long. DCI-based beam indication (Unified TCI) not "
            "configured. UE continues transmitting on outdated spatial relation causing "
            "uplink SINR degradation."
        ),
        "expected_specs": ["TS 38.214", "TS 38.321", "TS 38.213", "TS 38.331"],
        "expected_keywords": [
            "L1-RSRP", "CSI-RS", "TCI state", "MAC CE",
            "beam indication", "DCI", "spatial relation",
            "SINR", "P2 procedure"
        ],
    },
    {
        "id": "BEAM-004",
        "category": "Beam Management/MIMO",
        "difficulty": "medium",
        "input": (
            "Cell-edge UE experiences throughput oscillation between 200 Mbps and 5 Mbps. "
            "CQI reports fluctuate between CQI 12 and CQI 2 every TTI. PDSCH BLER spikes "
            "to 30% during drops. Outer loop link adaptation (OLLA) over-compensates. "
            "gNB scheduler downgrades MCS to QPSK when 64QAM should be sustainable. "
            "Suspected interference from neighboring cell using same SSB beam direction."
        ),
        "expected_specs": ["TS 38.214", "TS 38.213", "TS 38.321", "TS 38.211"],
        "expected_keywords": [
            "CQI", "PDSCH", "BLER", "OLLA", "MCS",
            "link adaptation", "QPSK", "64QAM", "scheduler",
            "interference", "SSB"
        ],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 5: Core Network (AMF/SMF/UPF) Signaling Failures
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "CORE-001",
        "category": "Core Network Failures",
        "difficulty": "easy",
        "input": (
            "PDU Session Establishment fails. UE sends NAS PDU Session Establishment "
            "Request for SST=1 SD=000001 (eMBB). SMF rejects with 5GSM cause #27 "
            "'missing or unknown DNN'. UE requested DNN 'internet' but SMF subscription "
            "data from UDM shows only 'ims' DNN allowed."
        ),
        "expected_specs": ["TS 24.501", "TS 23.501", "TS 23.502", "TS 29.502"],
        "expected_keywords": [
            "PDU Session Establishment", "S-NSSAI", "SST",
            "DNN", "SMF", "5GSM cause", "UDM", "subscription"
        ],
    },
    {
        "id": "CORE-002",
        "category": "Core Network Failures",
        "difficulty": "hard",
        "input": (
            "UPF N4 session establishment failure. SMF sends PFCP Session Establishment "
            "Request to UPF but receives cause 'No resources available'. UPF FAR table "
            "full — maximum 500k forwarding rules reached. N9 interface between PSA UPF "
            "and I-UPF shows tunnel endpoint ID exhaustion. SSC mode 2 session continuity "
            "during UE mobility creates orphaned PDRs on anchor UPF."
        ),
        "expected_specs": ["TS 29.244", "TS 23.501", "TS 23.502", "TS 29.281"],
        "expected_keywords": [
            "PFCP", "UPF", "N4", "FAR", "PDR", "session establishment",
            "N9", "TEID", "SSC mode", "PSA", "I-UPF"
        ],
    },
    {
        "id": "CORE-003",
        "category": "Core Network Failures",
        "difficulty": "medium",
        "input": (
            "AMF overload scenario. NGAP Overload Start message sent to all gNBs with "
            "overloadAction 'reject-non-emergency-mo-dt'. New registrations for non-emergency "
            "UEs are rejected. AMF pool area rebalancing fails because secondary AMF's "
            "TNL associations not established. NRF service discovery returns stale AMF profiles."
        ),
        "expected_specs": ["TS 38.413", "TS 23.501", "TS 23.502", "TS 29.510"],
        "expected_keywords": [
            "NGAP", "Overload", "AMF", "gNB", "registration",
            "TNL association", "NRF", "service discovery",
            "AMF pool"
        ],
    },
    {
        "id": "CORE-004",
        "category": "Core Network Failures",
        "difficulty": "medium",
        "input": (
            "Network slice selection failure. UE includes requested NSSAI with "
            "S-NSSAI (SST=3, SD=000003) for V2X slice. NSSF returns empty allowed "
            "NSSAI — V2X slice not available in current Tracking Area. AMF does not "
            "perform inter-AMF rerouting to AMF serving the target slice. NSI ID "
            "to NF Set mapping missing in NRF."
        ),
        "expected_specs": ["TS 23.501", "TS 23.502", "TS 29.531", "TS 24.501"],
        "expected_keywords": [
            "NSSAI", "S-NSSAI", "SST", "NSSF", "network slice",
            "AMF rerouting", "V2X", "Tracking Area", "NRF", "NSI"
        ],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 6: QoS Flow Establishment Issues
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "QOS-001",
        "category": "QoS Flow Issues",
        "difficulty": "easy",
        "input": (
            "Video streaming QoS degradation. Application requests GBR QoS flow with "
            "5QI=2 (conversational video) but PCF returns non-GBR 5QI=9 via Npcf policy "
            "authorization. SMF creates QoS flow with wrong parameters. GFBR and MFBR "
            "not enforced at UPF. UE sees buffering and stalling on 4K content."
        ),
        "expected_specs": ["TS 23.501", "TS 23.503", "TS 29.512", "TS 29.244"],
        "expected_keywords": [
            "5QI", "QoS flow", "GBR", "non-GBR", "PCF",
            "GFBR", "MFBR", "UPF", "policy", "PDU Session"
        ],
    },
    {
        "id": "QOS-002",
        "category": "QoS Flow Issues",
        "difficulty": "hard",
        "input": (
            "Reflective QoS failure for enterprise application. UPF marks downlink packets "
            "with RQI=1 in GTP-U extension header but UE does not create uplink QoS rule. "
            "SDAP header configuration missing — reflective QoS not enabled on the DRB. "
            "QoS flow to DRB mapping inconsistent between gNB and UE. PDCP SDU discard "
            "timer expires for out-of-order packets on mapped DRB."
        ),
        "expected_specs": ["TS 38.323", "TS 37.324", "TS 23.501", "TS 29.281"],
        "expected_keywords": [
            "reflective QoS", "RQI", "SDAP", "DRB mapping",
            "QoS flow", "GTP-U", "PDCP", "SDU discard",
            "QoS rule", "UPF"
        ],
    },
    {
        "id": "QOS-003",
        "category": "QoS Flow Issues",
        "difficulty": "medium",
        "input": (
            "URLLC QoS flow with 5QI=82 (discrete automation) experiences packet "
            "delay budget violations. gNB scheduler assigns logical channel priority "
            "correctly but configured grant (Type 1) periodicity mismatched with "
            "application traffic pattern. Pre-emption of eMBB traffic not triggered "
            "despite ARP priority level 1. Semi-persistent scheduling ineffective."
        ),
        "expected_specs": ["TS 23.501", "TS 38.321", "TS 38.214", "TS 38.331"],
        "expected_keywords": [
            "URLLC", "5QI", "packet delay budget", "scheduler",
            "configured grant", "pre-emption", "ARP",
            "logical channel priority", "SPS"
        ],
    },
    {
        "id": "QOS-004",
        "category": "QoS Flow Issues",
        "difficulty": "medium",
        "input": (
            "Multiple QoS flow multiplexing failure. PDU Session has 3 QoS flows "
            "(5QI=1 voice, 5QI=5 IMS signaling, 5QI=9 default). gNB maps all flows "
            "to single DRB ignoring 5QI requirements. RLC AM configuration uses "
            "same poll-retransmit timer for all, causing voice latency spikes. "
            "MAC scheduler does not differentiate LCH priorities. QFI in SDAP "
            "header not parsed correctly."
        ),
        "expected_specs": ["TS 37.324", "TS 38.322", "TS 38.321", "TS 23.501"],
        "expected_keywords": [
            "QoS flow", "5QI", "DRB mapping", "RLC AM",
            "poll-retransmit", "MAC scheduler", "LCH priority",
            "QFI", "SDAP", "multiplexing"
        ],
    },
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 7: RAN Energy Efficiency Scenarios
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "ENERGY-001",
        "category": "RAN Energy Efficiency",
        "difficulty": "easy",
        "input": (
            "Cell site energy consumption increased 40% after 5G NR activation despite "
            "low traffic load at night. Massive MIMO 64T64R antenna operating at full "
            "power 24/7. No SSB beam reduction during off-peak hours. Cell DTX/DRX "
            "features not activated. Symbol-level shutdown not configured."
        ),
        "expected_specs": ["TS 38.213", "TS 38.300", "TS 28.310"],
        "expected_keywords": [
            "energy efficiency", "SSB", "beam reduction",
            "cell DTX", "symbol shutdown", "MIMO",
            "power consumption", "traffic load"
        ],
    },
    {
        "id": "ENERGY-002",
        "category": "RAN Energy Efficiency",
        "difficulty": "hard",
        "input": (
            "Carrier shutdown feature conflict with coverage requirements. When secondary "
            "NR carrier (n78) is deactivated for energy saving, remaining carrier (n1) "
            "cannot handle capacity. UEs experience RLF due to SINR degradation. Carrier "
            "reactivation trigger based on PRB utilization threshold but RACH load "
            "monitoring not considered. Inter-carrier load balancing MLB algorithm "
            "oscillates between activation/deactivation. AI/ML energy saving model "
            "from TS 28.310 not deployed."
        ),
        "expected_specs": ["TS 38.300", "TS 28.310", "TS 38.331", "TS 28.552"],
        "expected_keywords": [
            "carrier shutdown", "energy saving", "RLF",
            "PRB utilization", "load balancing", "MLB",
            "RACH", "AI/ML", "carrier activation"
        ],
    },
    {
        "id": "ENERGY-003",
        "category": "RAN Energy Efficiency",
        "difficulty": "medium",
        "input": (
            "Advanced sleep mode implementation causing paging failures. gNB enters "
            "deep sleep (SM4) but paging occasion processing delayed. UE misses "
            "paging in DRX cycle. MT call setup time exceeds 10 seconds. Paging "
            "DRX parameters misaligned between AMF and gNB. Wake-up signal (WUS) "
            "not configured for NR cells."
        ),
        "expected_specs": ["TS 38.304", "TS 38.213", "TS 38.321", "TS 23.502"],
        "expected_keywords": [
            "sleep mode", "paging", "DRX cycle",
            "paging occasion", "wake-up signal", "WUS",
            "deep sleep", "MT call"
        ],
    },
    {
        "id": "ENERGY-004",
        "category": "RAN Energy Efficiency",
        "difficulty": "medium",
        "input": (
            "MIMO layer adaptation for energy optimization misconfigured. gNB reduces "
            "from 4 MIMO layers to 2 during low-load but does not send "
            "RRCReconfiguration to update UE CSI reporting configuration. UE continues "
            "reporting rank-4 PMI causing PDSCH decoding failures. CQI becomes unreliable. "
            "Throughput drops 60% instead of expected graceful degradation. KPI monitoring "
            "via PM counters shows increasing PDSCH BLER."
        ),
        "expected_specs": ["TS 38.214", "TS 38.331", "TS 28.552", "TS 38.211"],
        "expected_keywords": [
            "MIMO layer", "CSI reporting", "rank", "PMI",
            "PDSCH", "CQI", "BLER", "RRCReconfiguration",
            "energy optimization", "PM counters"
        ],
    },
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4 — MODEL LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_base_model() -> tuple:
    """Load the raw base Llama-3.3-70B-Instruct model (4-bit, NO LoRA)."""
    print("  ⏳ Loading BASE model (no LoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model.eval()
    print("  ✅ Base model loaded.")
    return model, tokenizer


def load_finetuned_model() -> tuple:
    """Load the fine-tuned model with telco_expert LoRA adapter weights."""
    print("  ⏳ Loading FINE-TUNED model (with LoRA adapter)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, config)
    safetensors_path = str(LORA_ADAPTER_DIR / "adapter_model.safetensors")
    weights = load_file(safetensors_path)
    model.load_state_dict(weights, strict=False)
    model.eval()
    print("  ✅ Fine-tuned model loaded.")
    return model, tokenizer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5 — INFERENCE ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_prompt(user_input: str) -> str:
    """Construct a Llama-3.x chat-template prompt from the canonical system prompt."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_input}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


@torch.inference_mode()
def generate_response(
    model, tokenizer, user_input: str
) -> tuple[str, float]:
    """
    Run a single inference pass and return (response_text, latency_seconds).
    """
    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    eos_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id:
        eos_ids.append(eot_id)

    start = time.perf_counter()
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        eos_token_id=eos_ids,
        **GENERATION_CONFIG,
    )
    elapsed = time.perf_counter() - start

    # Decode only newly generated tokens (strip the prompt)
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response, elapsed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §6 — SCORING ENGINE (4 dimensions, 0-100 each)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Regex patterns for detecting 3GPP spec references in free text
_SPEC_PATTERN = re.compile(r"(?:TS|TR)\s*\d{2}\.\d{3}", re.IGNORECASE)

# Known-invalid spec numbers (common hallucinations) for penalty scoring
KNOWN_INVALID_SPECS: set[str] = {
    "TS 38.999", "TS 38.000", "TS 23.999", "TS 38.500", "TS 38.600",
    "TS 23.600", "TS 24.600", "TS 29.999", "TS 38.700", "TS 38.800",
    "TS 99.001", "TS 38.100", "TS 23.100", "TS 38.900",  # TS 38.900 invalid (TR exists)
}

# Impossible protocol combinations (hallucination markers)
IMPOSSIBLE_COMBOS: list[tuple[str, str]] = [
    ("PFCP", "RRC"),           # PFCP is core-only, not RAN
    ("GTP-C", "MAC CE"),       # GTP-C is control plane, MAC CE is L2 RAN
    ("S1AP", "gNB"),           # S1AP is LTE-only, gNB is NR
    ("X2AP", "AMF"),           # X2AP is eNB-eNB, AMF is 5GC
    ("SIP INVITE", "PDCCH"),   # SIP is application layer, PDCCH is L1
    ("RLC", "NRF"),            # RLC is RAN L2, NRF is core NF
]

# Made-up 3GPP terminology commonly hallucinated
FAKE_TERMINOLOGY: list[str] = [
    "5G-NR-Ultra-HARQ",
    "Super-MIMO-Beamforming-Protocol",
    "Quantum-NAS-Encryption",
    "NR-Turbo-Scheduler",
    "Advanced-OFDMA-Boost",
    "HyperBeam-Steering-Protocol",
    "gNB-AI-Self-Healing-Stack",
    "Ultra-PDCP-Aggregation-Layer",
    "NR-CloudRAN-Fusion-Engine",
    "Mega-QoS-Flow-Accelerator",
]


def _normalize_spec(raw: str) -> str:
    """Normalize a captured spec reference to 'TS XX.YYY' or 'TR XX.YYY'."""
    raw = raw.strip().upper()
    raw = re.sub(r"\s+", " ", raw)
    return raw


def score_3gpp_compliance(response: str, expected_specs: list[str]) -> dict:
    """
    Score how well the response cites valid 3GPP specifications.
    Returns dict with score (0-100) and detail breakdown.
    """
    found_raw = _SPEC_PATTERN.findall(response)
    found_normalized = [_normalize_spec(s) for s in found_raw]
    found_unique = set(found_normalized)

    valid_found = found_unique & VALID_3GPP_SPECS
    invalid_found = found_unique - VALID_3GPP_SPECS
    expected_set = set(expected_specs)
    expected_hit = valid_found & expected_set

    # Scoring:
    #   +40 pts for expected spec coverage (% of expected specs found)
    #   +30 pts for total valid spec citations (capped at 10)
    #   +30 pts baseline, minus penalties for invalid specs
    if len(expected_set) > 0:
        coverage_score = (len(expected_hit) / len(expected_set)) * 40
    else:
        coverage_score = 40.0

    valid_citation_score = min(len(valid_found), 10) / 10 * 30
    penalty = min(len(invalid_found) * 10, 30)  # -10 per invalid, capped at 30
    base_score = 30.0 - penalty

    total = max(0, min(100, coverage_score + valid_citation_score + base_score))

    return {
        "score": round(total, 1),
        "found_specs": sorted(found_unique),
        "valid_specs": sorted(valid_found),
        "invalid_specs": sorted(invalid_found),
        "expected_hit": sorted(expected_hit),
        "expected_miss": sorted(expected_set - valid_found),
    }


def score_protocol_accuracy(response: str, expected_keywords: list[str]) -> dict:
    """
    Score the presence of expected protocol-layer keywords.
    Case-insensitive matching with word-boundary awareness.
    """
    response_lower = response.lower()
    hits = []
    misses = []
    for kw in expected_keywords:
        # Flexible matching: check if keyword appears as substring (case-insensitive)
        if kw.lower() in response_lower:
            hits.append(kw)
        else:
            # Try without hyphens/underscores for robustness
            kw_clean = re.sub(r"[-_\s]", "", kw.lower())
            resp_clean = re.sub(r"[-_\s]", "", response_lower)
            if kw_clean in resp_clean:
                hits.append(kw)
            else:
                misses.append(kw)

    if len(expected_keywords) > 0:
        score = (len(hits) / len(expected_keywords)) * 100
    else:
        score = 100.0

    return {
        "score": round(score, 1),
        "hits": hits,
        "misses": misses,
        "total_expected": len(expected_keywords),
        "total_found": len(hits),
    }


def score_structural_quality(response: str) -> dict:
    """
    Evaluate diagnostic template adherence:
      - Has numbered sections?       (+25 pts)
      - Has 'Root Cause' section?    (+25 pts)
      - Has 'Recommendation' section? (+25 pts)
      - Has protocol-level detail?    (+25 pts)
    """
    checks = {}

    # 1. Numbered sections (e.g., "1.", "2.", "###", "**")
    has_numbered = bool(re.search(r"(?:^|\n)\s*\d+\.\s", response))
    has_headers = bool(re.search(r"#{1,4}\s", response))
    has_bold = bool(re.search(r"\*\*[^*]+\*\*", response))
    checks["numbered_sections"] = has_numbered or has_headers or has_bold

    # 2. Root cause discussion
    root_cause_phrases = [
        "root cause", "Root Cause", "root-cause", "failure analysis",
        "Failure Analysis", "cause trace", "Protocol Root Cause",
        "protocol analysis", "Protocol Analysis"
    ]
    checks["has_root_cause"] = any(p.lower() in response.lower() for p in root_cause_phrases)

    # 3. Recommendation section
    rec_phrases = [
        "recommend", "Recommend", "corrective action", "Corrective Action",
        "mitigation", "Mitigation", "resolution", "Resolution",
        "remediation", "Remediation", "action item", "Action Item"
    ]
    checks["has_recommendation"] = any(p.lower() in response.lower() for p in rec_phrases)

    # 4. Protocol-level detail (technical depth indicators)
    tech_indicators = [
        r"TS\s*\d{2}\.\d{3}", r"5QI", r"DRB", r"RRC", r"NAS",
        r"NGAP", r"PFCP", r"GTP", r"SIP", r"PDCP", r"RLC",
        r"MAC\s*CE", r"HARQ", r"MCS", r"CQI", r"RSRP",
        r"SINR", r"BLER", r"dBm", r"3GPP",
    ]
    tech_count = sum(1 for pat in tech_indicators if re.search(pat, response, re.IGNORECASE))
    checks["protocol_detail"] = tech_count >= 3  # At least 3 tech terms

    score = sum(25 for v in checks.values() if v)
    return {
        "score": round(score, 1),
        "checks": checks,
        "tech_term_count": tech_count,
    }


def score_hallucination(response: str) -> dict:
    """
    Detect hallucinations:
      Start at 100 (perfect), subtract penalties.
      - Invalid spec numbers:          -10 each (up to -30)
      - Impossible protocol combos:    -15 each (up to -30)
      - Fake terminology:              -20 each (up to -40)
    """
    penalties = []

    # 1. Check for known-invalid specs
    found_specs = _SPEC_PATTERN.findall(response)
    found_normalized = {_normalize_spec(s) for s in found_specs}
    invalid_specs = found_normalized & KNOWN_INVALID_SPECS
    # Also check specs that are not in our valid list and look suspicious
    unknown_specs = found_normalized - VALID_3GPP_SPECS - KNOWN_INVALID_SPECS
    spec_penalty = min(len(invalid_specs) * 10 + len(unknown_specs) * 5, 30)
    if invalid_specs or unknown_specs:
        penalties.append({
            "type": "invalid_specs",
            "details": sorted(invalid_specs | unknown_specs),
            "penalty": spec_penalty,
        })

    # 2. Impossible protocol combinations
    combo_count = 0
    combo_details = []
    for term_a, term_b in IMPOSSIBLE_COMBOS:
        if (term_a.lower() in response.lower() and term_b.lower() in response.lower()):
            # Check if they appear in the same sentence (within 200 chars)
            for match_a in re.finditer(re.escape(term_a), response, re.IGNORECASE):
                pos_a = match_a.start()
                # Look for term_b within ±200 chars
                window = response[max(0, pos_a - 200):pos_a + 200]
                if re.search(re.escape(term_b), window, re.IGNORECASE):
                    combo_count += 1
                    combo_details.append(f"{term_a}↔{term_b}")
                    break
    combo_penalty = min(combo_count * 15, 30)
    if combo_count > 0:
        penalties.append({
            "type": "impossible_combos",
            "details": combo_details,
            "penalty": combo_penalty,
        })

    # 3. Fake terminology
    fake_found = []
    for fake in FAKE_TERMINOLOGY:
        if fake.lower() in response.lower():
            fake_found.append(fake)
    fake_penalty = min(len(fake_found) * 20, 40)
    if fake_found:
        penalties.append({
            "type": "fake_terminology",
            "details": fake_found,
            "penalty": fake_penalty,
        })

    total_penalty = spec_penalty + combo_penalty + fake_penalty
    score = max(0, 100 - total_penalty)

    return {
        "score": round(score, 1),
        "penalties": penalties,
        "total_penalty": total_penalty,
        "invalid_specs_found": sorted(invalid_specs),
        "unknown_specs_found": sorted(unknown_specs),
        "fake_terms_found": fake_found,
    }


def compute_all_scores(
    response: str,
    expected_specs: list[str],
    expected_keywords: list[str],
) -> dict:
    """Run all four scoring dimensions and return a combined result."""
    compliance = score_3gpp_compliance(response, expected_specs)
    protocol = score_protocol_accuracy(response, expected_keywords)
    structure = score_structural_quality(response)
    hallucination = score_hallucination(response)

    composite = round(
        0.30 * compliance["score"]
        + 0.30 * protocol["score"]
        + 0.20 * structure["score"]
        + 0.20 * hallucination["score"],
        1,
    )

    return {
        "3gpp_compliance": compliance,
        "protocol_accuracy": protocol,
        "structural_quality": structure,
        "hallucination_detection": hallucination,
        "composite_score": composite,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §7 — VISUALIZATION ENGINE (Publication-Quality AMD Dark Theme)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _apply_amd_theme():
    """Configure matplotlib global rcParams for AMD dark branding."""
    plt.rcParams.update({
        "figure.facecolor": AMD_BG,
        "axes.facecolor": AMD_PANEL,
        "axes.edgecolor": AMD_GRID,
        "axes.labelcolor": AMD_FG_TEXT,
        "text.color": AMD_FG_TEXT,
        "xtick.color": AMD_FG_TEXT,
        "ytick.color": AMD_FG_TEXT,
        "grid.color": AMD_GRID,
        "grid.alpha": 0.4,
        "legend.facecolor": AMD_PANEL,
        "legend.edgecolor": AMD_GRID,
        "legend.labelcolor": AMD_FG_TEXT,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def generate_radar_chart(
    base_scores: dict[str, float],
    ft_scores: dict[str, float],
    save_path: str,
):
    """
    Create a radar/spider chart comparing Base vs Fine-Tuned across
    the 4 scoring dimensions.
    """
    _apply_amd_theme()

    categories = ["3GPP\nCompliance", "Protocol\nAccuracy",
                   "Structural\nQuality", "Hallucination\nDetection"]
    n_cats = len(categories)

    base_vals = [
        base_scores["3gpp_compliance"],
        base_scores["protocol_accuracy"],
        base_scores["structural_quality"],
        base_scores["hallucination_detection"],
    ]
    ft_vals = [
        ft_scores["3gpp_compliance"],
        ft_scores["protocol_accuracy"],
        ft_scores["structural_quality"],
        ft_scores["hallucination_detection"],
    ]

    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    base_vals += base_vals[:1]
    ft_vals += ft_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(AMD_BG)
    ax.set_facecolor(AMD_PANEL)

    # Draw axis reference circles
    for level in [20, 40, 60, 80, 100]:
        circle = np.full(len(angles), level)
        ax.plot(angles, circle, color=AMD_GRID, linewidth=0.5, alpha=0.6)

    # Plot data
    ax.plot(angles, base_vals, "o-", color=AMD_GREY_LIGHT, linewidth=2,
            markersize=7, label="Base Model", alpha=0.8)
    ax.fill(angles, base_vals, color=AMD_GREY, alpha=0.15)

    ax.plot(angles, ft_vals, "o-", color=AMD_RED, linewidth=2.5,
            markersize=8, label="Fine-Tuned (LoRA)", alpha=0.95)
    ax.fill(angles, ft_vals, color=AMD_RED, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9, color=AMD_FG_TEXT)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=11,
              framealpha=0.8)
    ax.set_title("Evaluation Radar: Base vs Fine-Tuned\n(Avg. Across All Queries)",
                 fontsize=15, fontweight="bold", pad=30, color=AMD_FG_TEXT)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=AMD_BG, edgecolor="none")
    plt.close(fig)
    print(f"  📊 Radar chart saved → {save_path}")


def generate_category_bar_chart(
    base_by_cat: dict[str, float],
    ft_by_cat: dict[str, float],
    save_path: str,
):
    """Bar chart: composite score per category, base vs fine-tuned."""
    _apply_amd_theme()

    categories = list(ft_by_cat.keys())
    n = len(categories)
    x = np.arange(n)
    width = 0.35

    base_vals = [base_by_cat.get(c, 0) for c in categories]
    ft_vals = [ft_by_cat.get(c, 0) for c in categories]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(AMD_BG)

    bars_base = ax.bar(x - width / 2, base_vals, width, label="Base Model",
                       color=AMD_GREY, edgecolor=AMD_GREY_LIGHT, linewidth=0.5,
                       alpha=0.8)
    bars_ft = ax.bar(x + width / 2, ft_vals, width, label="Fine-Tuned (LoRA)",
                     color=AMD_RED, edgecolor=AMD_RED_LIGHT, linewidth=0.5,
                     alpha=0.9)

    # Value labels on bars
    for bar in bars_base:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}",
                ha="center", va="bottom", fontsize=9, color=AMD_GREY_LIGHT,
                fontweight="bold")
    for bar in bars_ft:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}",
                ha="center", va="bottom", fontsize=9, color=AMD_RED_LIGHT,
                fontweight="bold")

    ax.set_xlabel("Diagnostic Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Composite Score (0-100)", fontsize=12, fontweight="bold")
    ax.set_title("Composite Score by Category: Base vs Fine-Tuned",
                 fontsize=15, fontweight="bold", color=AMD_FG_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=AMD_BG, edgecolor="none")
    plt.close(fig)
    print(f"  📊 Category bar chart saved → {save_path}")


def generate_overall_comparison_chart(
    base_composite: float,
    ft_composite: float,
    base_dims: dict[str, float],
    ft_dims: dict[str, float],
    save_path: str,
):
    """Overall composite + per-dimension comparison as grouped horizontal bars."""
    _apply_amd_theme()

    labels = [
        "Overall Composite",
        "3GPP Compliance",
        "Protocol Accuracy",
        "Structural Quality",
        "Hallucination Detection",
    ]
    base_vals = [
        base_composite,
        base_dims["3gpp_compliance"],
        base_dims["protocol_accuracy"],
        base_dims["structural_quality"],
        base_dims["hallucination_detection"],
    ]
    ft_vals = [
        ft_composite,
        ft_dims["3gpp_compliance"],
        ft_dims["protocol_accuracy"],
        ft_dims["structural_quality"],
        ft_dims["hallucination_detection"],
    ]

    y = np.arange(len(labels))
    height = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(AMD_BG)

    ax.barh(y + height / 2, base_vals, height, label="Base Model",
            color=AMD_GREY, edgecolor=AMD_GREY_LIGHT, linewidth=0.5, alpha=0.8)
    ax.barh(y - height / 2, ft_vals, height, label="Fine-Tuned (LoRA)",
            color=AMD_RED, edgecolor=AMD_RED_LIGHT, linewidth=0.5, alpha=0.9)

    # Value labels
    for i, (bv, fv) in enumerate(zip(base_vals, ft_vals)):
        ax.text(bv + 1, i + height / 2, f"{bv:.1f}", va="center",
                fontsize=9, color=AMD_GREY_LIGHT, fontweight="bold")
        ax.text(fv + 1, i - height / 2, f"{fv:.1f}", va="center",
                fontsize=9, color=AMD_RED_LIGHT, fontweight="bold")

    # Highlight the composite row
    ax.axhspan(y[0] - 0.5, y[0] + 0.5, alpha=0.08, color=AMD_RED)

    ax.set_xlabel("Score (0-100)", fontsize=12, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.set_title("Overall Model Comparison: Base vs Fine-Tuned",
                 fontsize=15, fontweight="bold", color=AMD_FG_TEXT)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=AMD_BG, edgecolor="none")
    plt.close(fig)
    print(f"  📊 Overall comparison chart saved → {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §8 — CONSOLE SUMMARY TABLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def print_summary_table(results: list[dict]):
    """Pretty-print a comparison summary table to stdout."""
    sep = "═" * 130
    thin_sep = "─" * 130

    print(f"\n{sep}")
    print("  5G DIAGNOSTIC ENGINE — BENCHMARK SUMMARY TABLE")
    print(f"{sep}\n")

    header = (
        f"{'ID':<10} {'Category':<26} {'Diff':<7} "
        f"│ {'Base Comp':>10} │ {'FT Comp':>10} │ {'Δ':>7} "
        f"│ {'3GPP(B/F)':>12} │ {'Proto(B/F)':>12} │ {'Struct(B/F)':>12} │ {'Halluc(B/F)':>12}"
    )
    print(header)
    print(thin_sep)

    for r in results:
        qid = r["query_id"]
        cat = r["category"][:25]
        diff = r["difficulty"]
        bc = r["base_scores"]["composite_score"]
        fc = r["finetuned_scores"]["composite_score"]
        delta = fc - bc
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

        b3 = r["base_scores"]["3gpp_compliance"]["score"]
        f3 = r["finetuned_scores"]["3gpp_compliance"]["score"]
        bp = r["base_scores"]["protocol_accuracy"]["score"]
        fp = r["finetuned_scores"]["protocol_accuracy"]["score"]
        bs = r["base_scores"]["structural_quality"]["score"]
        fs = r["finetuned_scores"]["structural_quality"]["score"]
        bh = r["base_scores"]["hallucination_detection"]["score"]
        fh = r["finetuned_scores"]["hallucination_detection"]["score"]

        print(
            f"{qid:<10} {cat:<26} {diff:<7} "
            f"│ {bc:>10.1f} │ {fc:>10.1f} │ {delta_str:>7} "
            f"│ {b3:>5.1f}/{f3:<5.1f} │ {bp:>5.1f}/{fp:<5.1f} "
            f"│ {bs:>5.1f}/{fs:<5.1f} │ {bh:>5.1f}/{fh:<5.1f}"
        )

    print(thin_sep)

    # Aggregate averages
    base_composites = [r["base_scores"]["composite_score"] for r in results]
    ft_composites = [r["finetuned_scores"]["composite_score"] for r in results]
    avg_base = sum(base_composites) / len(base_composites)
    avg_ft = sum(ft_composites) / len(ft_composites)
    avg_delta = avg_ft - avg_base

    print(
        f"{'AVERAGE':<10} {'(all categories)':<26} {'---':<7} "
        f"│ {avg_base:>10.1f} │ {avg_ft:>10.1f} │ "
        f"{'+'if avg_delta>=0 else ''}{avg_delta:>6.1f} │"
    )
    print(f"{sep}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §9 — MAIN BENCHMARK ORCHESTRATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_benchmark(quick: bool = False):
    """
    Main entry point.  Loads both models, runs queries, scores, and
    generates all outputs.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    queries = TEST_QUERIES[:5] if quick else TEST_QUERIES
    mode_label = "QUICK (5 queries)" if quick else "FULL (30 queries)"

    print("=" * 80)
    print("  5G Core/RAN Diagnostic Engine — Benchmark Suite")
    print(f"  Mode         : {mode_label}")
    print(f"  Hardware      : AMD Instinct MI300X (192 GB HBM3)")
    print(f"  Base Model    : {BASE_MODEL_ID}")
    print(f"  LoRA Adapter  : {LORA_ADAPTER_DIR}")
    print(f"  Timestamp     : {timestamp}")
    print("=" * 80)

    # ── Load Models ───────────────────────────────────────────────────────────
    print("\n▶ Phase 1: Loading Models")
    print("─" * 40)
    base_model, base_tokenizer = load_base_model()

    print("\n  🔄 Unloading base model from GPU to free VRAM...")
    # We'll run base inference first, then swap
    # Actually, to keep memory manageable on 192GB, we run base queries first,
    # then release and load fine-tuned model.

    # ── Run Base Model Inference ──────────────────────────────────────────────
    print("\n▶ Phase 2: Base Model Inference")
    print("─" * 40)
    base_responses: dict[str, dict] = {}
    for i, q in enumerate(queries, 1):
        print(f"  [{i:>2}/{len(queries)}] {q['id']} ({q['category']}) ...", end=" ", flush=True)
        response, latency = generate_response(base_model, base_tokenizer, q["input"])
        base_responses[q["id"]] = {
            "response": response,
            "latency_s": round(latency, 2),
        }
        print(f"✓ ({latency:.1f}s, {len(response)} chars)")

    # ── Release base model VRAM ───────────────────────────────────────────────
    print("\n  🧹 Releasing base model VRAM...")
    del base_model, base_tokenizer
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()

    # ── Load & Run Fine-Tuned Model ───────────────────────────────────────────
    print("\n▶ Phase 3: Fine-Tuned Model Inference")
    print("─" * 40)
    ft_model, ft_tokenizer = load_finetuned_model()

    ft_responses: dict[str, dict] = {}
    for i, q in enumerate(queries, 1):
        print(f"  [{i:>2}/{len(queries)}] {q['id']} ({q['category']}) ...", end=" ", flush=True)
        response, latency = generate_response(ft_model, ft_tokenizer, q["input"])
        ft_responses[q["id"]] = {
            "response": response,
            "latency_s": round(latency, 2),
        }
        print(f"✓ ({latency:.1f}s, {len(response)} chars)")

    # Release fine-tuned model
    del ft_model, ft_tokenizer
    torch.cuda.empty_cache()

    # ── Score All Responses ───────────────────────────────────────────────────
    print("\n▶ Phase 4: Scoring Responses")
    print("─" * 40)
    results: list[dict] = []
    for q in queries:
        qid = q["id"]
        print(f"  Scoring {qid}...", end=" ", flush=True)

        base_scores = compute_all_scores(
            base_responses[qid]["response"],
            q["expected_specs"],
            q["expected_keywords"],
        )
        ft_scores = compute_all_scores(
            ft_responses[qid]["response"],
            q["expected_specs"],
            q["expected_keywords"],
        )

        results.append({
            "query_id": qid,
            "category": q["category"],
            "difficulty": q["difficulty"],
            "input": q["input"],
            "expected_specs": q["expected_specs"],
            "expected_keywords": q["expected_keywords"],
            "base_response": base_responses[qid]["response"],
            "base_latency_s": base_responses[qid]["latency_s"],
            "base_scores": base_scores,
            "finetuned_response": ft_responses[qid]["response"],
            "finetuned_latency_s": ft_responses[qid]["latency_s"],
            "finetuned_scores": ft_scores,
        })
        delta = ft_scores["composite_score"] - base_scores["composite_score"]
        print(f"Base={base_scores['composite_score']:.1f}  "
              f"FT={ft_scores['composite_score']:.1f}  "
              f"Δ={'+' if delta>=0 else ''}{delta:.1f}")

    # ── Print Console Summary ─────────────────────────────────────────────────
    print_summary_table(results)

    # ── Compute Aggregated Averages ───────────────────────────────────────────
    dim_keys = ["3gpp_compliance", "protocol_accuracy",
                "structural_quality", "hallucination_detection"]

    def avg_dim(results_list, model_key, dim):
        return sum(r[model_key][dim]["score"] for r in results_list) / len(results_list)

    base_avg_dims = {d: round(avg_dim(results, "base_scores", d), 1) for d in dim_keys}
    ft_avg_dims = {d: round(avg_dim(results, "finetuned_scores", d), 1) for d in dim_keys}

    base_avg_composite = round(
        sum(r["base_scores"]["composite_score"] for r in results) / len(results), 1
    )
    ft_avg_composite = round(
        sum(r["finetuned_scores"]["composite_score"] for r in results) / len(results), 1
    )

    # Per-category composite averages
    categories_seen = sorted(set(q["category"] for q in queries))
    base_by_cat = {}
    ft_by_cat = {}
    for cat in categories_seen:
        cat_results = [r for r in results if r["category"] == cat]
        base_by_cat[cat] = round(
            sum(r["base_scores"]["composite_score"] for r in cat_results) / len(cat_results), 1
        )
        ft_by_cat[cat] = round(
            sum(r["finetuned_scores"]["composite_score"] for r in cat_results) / len(cat_results), 1
        )

    # ── Save JSON Results ─────────────────────────────────────────────────────
    print("\n▶ Phase 5: Saving Outputs")
    print("─" * 40)

    json_path = RESULTS_DIR / f"benchmark_results_{timestamp}.json"
    output_payload = {
        "metadata": {
            "timestamp": timestamp,
            "mode": "quick" if quick else "full",
            "num_queries": len(queries),
            "base_model": BASE_MODEL_ID,
            "lora_adapter": str(LORA_ADAPTER_DIR),
            "generation_config": GENERATION_CONFIG,
            "system_prompt": SYSTEM_PROMPT,
        },
        "summary": {
            "base_avg_composite": base_avg_composite,
            "finetuned_avg_composite": ft_avg_composite,
            "improvement": round(ft_avg_composite - base_avg_composite, 1),
            "base_avg_dimensions": base_avg_dims,
            "finetuned_avg_dimensions": ft_avg_dims,
            "base_by_category": base_by_cat,
            "finetuned_by_category": ft_by_cat,
        },
        "results": results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"  💾 JSON results saved → {json_path}")

    # Also save a "latest" symlink / copy for convenience
    latest_path = RESULTS_DIR / "benchmark_results_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"  💾 Latest copy saved  → {latest_path}")

    # ── Generate Charts ───────────────────────────────────────────────────────
    print("\n▶ Phase 6: Generating Visualizations")
    print("─" * 40)

    radar_path = str(RESULTS_DIR / f"radar_chart_{timestamp}.png")
    generate_radar_chart(base_avg_dims, ft_avg_dims, radar_path)

    bar_path = str(RESULTS_DIR / f"category_bar_chart_{timestamp}.png")
    generate_category_bar_chart(base_by_cat, ft_by_cat, bar_path)

    overall_path = str(RESULTS_DIR / f"overall_comparison_{timestamp}.png")
    generate_overall_comparison_chart(
        base_avg_composite, ft_avg_composite,
        base_avg_dims, ft_avg_dims, overall_path
    )

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"  Base Model Avg Composite  : {base_avg_composite:.1f} / 100")
    print(f"  Fine-Tuned Avg Composite  : {ft_avg_composite:.1f} / 100")
    improvement = ft_avg_composite - base_avg_composite
    print(f"  Improvement               : {'+'if improvement>=0 else ''}{improvement:.1f} pts")
    print(f"  Queries Evaluated         : {len(queries)}")
    print(f"  Results Directory         : {RESULTS_DIR}")
    print("=" * 80)

    return output_payload


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §10 — CLI ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "5G Core/RAN Diagnostic Engine — Evaluation & Benchmark Suite. "
            "Compares BASE vs FINE-TUNED Llama-3.3-70B on telecom diagnostic queries."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluation/benchmark.py            # Full 30-query run\n"
            "  python evaluation/benchmark.py --quick    # Fast 5-query smoke test\n"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Run only the first 5 queries for fast smoke testing.",
    )

    args = parser.parse_args()
    run_benchmark(quick=args.quick)
