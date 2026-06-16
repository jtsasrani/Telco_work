"""
3GPP Specification RAG (Retrieval-Augmented Generation) Module - SOTA Hybrid
===========================================================================
Provides context-aware 3GPP spec retrieval to augment the fine-tuned model's
responses with actual specification text, enabling precise citations.

Implements Hybrid Search (FAISS dense vector + BM25 sparse lexical search)
fused via Reciprocal Rank Fusion (RRF) and re-ranked using a Cross-Encoder.
"""

import os
import json
import re
import pickle
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional

# ============================================================================
# 3GPP SPECIFICATION KNOWLEDGE BASE (Fallback core dataset)
# ============================================================================
SPEC_KNOWLEDGE_BASE = [
    # TS 38.331 - NR RRC Protocol
    {
        "spec_id": "TS 38.331",
        "title": "NR; Radio Resource Control (RRC); Protocol specification",
        "section": "5.3.5 RRC connection reconfiguration",
        "content": (
            "The purpose of the RRC Connection Reconfiguration procedure is to modify an RRC connection. "
            "The procedure includes: establishment/modification/release of radio bearers, handover, "
            "measurement configuration changes, and other configuration changes. "
            "The network initiates the procedure by sending the RRCReconfiguration message which may contain "
            "reconfigurationWithSync (for handover), radioBearerConfig, measConfig, and other IEs. "
            "If reconfigurationWithSync is included, the UE shall perform synchronization to the target cell, "
            "apply the dedicated configuration, and send RRCReconfigurationComplete to the target cell. "
            "Timer T304 is started upon receiving reconfigurationWithSync. If T304 expires before successful "
            "completion, the UE declares a handover failure and initiates the RRC re-establishment procedure."
        ),
        "keywords": ["RRCReconfiguration", "handover", "T304", "reconfigurationWithSync", "re-establishment"]
    },
    {
        "spec_id": "TS 38.331",
        "title": "NR; Radio Resource Control (RRC); Protocol specification",
        "section": "5.3.7 RRC connection re-establishment",
        "content": (
            "The RRC connection re-establishment procedure is used to re-establish the RRC connection. "
            "A UE in RRC_CONNECTED may initiate the procedure when: radio link failure is detected (T310 expires "
            "and T311 is started), handover failure occurs (T304 expires), RRC integrity check failure, "
            "or RRC reconfiguration failure. The UE sends RRCReestablishmentRequest to the selected cell "
            "including the reason for re-establishment (reconfigurationFailure, handoverFailure, otherFailure). "
            "If the network cannot find the UE context, it responds with RRCReject, and the UE transitions "
            "to RRC_IDLE state, requiring full connection setup again."
        ),
        "keywords": ["RRCReestablishment", "radio link failure", "T310", "T311", "handover failure"]
    },
    {
        "spec_id": "TS 38.331",
        "title": "NR; Radio Resource Control (RRC); Protocol specification",
        "section": "5.5.2 Measurement configuration",
        "content": (
            "The network configures the UE measurement procedures via measConfig in RRCReconfiguration. "
            "Measurement objects define what to measure (frequencies, cells). Report configs define when to report "
            "(event-triggered: A1-A6, B1-B2 for inter-RAT, or periodical). Measurement IDs link objects to configs. "
            "Key events: A1 (serving better than threshold), A2 (serving worse than threshold), "
            "A3 (neighbour better than serving + offset), A5 (serving worse AND neighbour better). "
            "The UE evaluates measurement events using layer 3 filtering (filterCoefficient) and "
            "time-to-trigger (timeToTrigger) to avoid ping-pong handovers."
        ),
        "keywords": ["measConfig", "A3 event", "A5 event", "timeToTrigger", "ping-pong handover", "measurement"]
    },
    # TS 38.300 - NR Overall Description
    {
        "spec_id": "TS 38.300",
        "title": "NR; NR and NG-RAN Overall Description",
        "section": "9.2 Mobility in RRC_CONNECTED",
        "content": (
            "For intra-NR mobility in RRC_CONNECTED state, the handover procedure involves: "
            "1) Measurement configuration and reporting by UE, 2) Handover decision by source gNB, "
            "3) Handover preparation between source and target gNB via Xn interface, "
            "4) Handover execution: source gNB sends RRCReconfiguration with reconfigurationWithSync, "
            "5) UE detaches from source cell and synchronizes to target cell. "
            "Conditional Handover (CHO) allows the network to configure multiple candidate cells. "
            "The UE executes handover to the first candidate cell whose execution condition is met, "
            "reducing handover latency and failure rate."
        ),
        "keywords": ["handover", "Xn interface", "conditional handover", "CHO", "mobility"]
    },
    # TS 23.501 - 5G System Architecture
    {
        "spec_id": "TS 23.501",
        "title": "System architecture for the 5G System (5GS)",
        "section": "5.2.2 AMF (Access and Mobility Management Function)",
        "content": (
            "The AMF provides: Registration management, Connection management, Reachability management, "
            "Mobility management (handover between N3IWF and 3GPP access, inter-system handover), "
            "Access authentication and authorization, Security context management, Location services, "
            "NAS signaling with UE, NAS ciphering and integrity protection, "
            "N1/N2 interface termination. The AMF selection during registration is based on: "
            "S-NSSAI, GUAMI, 5G-TMSI, and network slicing support. "
            "Multiple AMF instances form an AMF set for load balancing and redundancy."
        ),
        "keywords": ["AMF", "registration", "NAS", "mobility management", "access management"]
    },
    {
        "spec_id": "TS 23.501",
        "title": "System architecture for the 5G System (5GS)",
        "section": "5.2.3 SMF (Session Management Function)",
        "content": (
            "The SMF provides: Session management (establishment, modification, release of PDU sessions), "
            "UE IP address allocation and management, DHCPv4/v6 functions, "
            "Selection and control of UPF, Traffic steering configuration at UPF, "
            "QoS control including QoS Flow to DRB mapping, Charging data collection, "
            "Downlink data notification. The SMF interacts with the AMF via N11 interface, "
            "with UPF via N4 interface (PFCP protocol), and with PCF via N7 interface for policy control."
        ),
        "keywords": ["SMF", "PDU session", "UPF", "QoS", "N4", "PFCP", "session management"]
    },
    {
        "spec_id": "TS 23.501",
        "title": "System architecture for the 5G System (5GS)",
        "section": "5.2.4 UPF (User Plane Function)",
        "content": (
            "The UPF provides: Packet routing and forwarding, Traffic usage reporting, "
            "Uplink classifier to support multi-homed PDU sessions, Branching point for multi-homed PDU sessions, "
            "QoS handling (packet marking, policing, gating), Downlink packet buffering, "
            "Uplink and downlink rate enforcement. UPF selection by SMF considers: "
            "DNN, S-NSSAI, UE location, UPF capabilities. The N3 interface connects RAN to UPF, "
            "N9 interface connects UPFs, N6 interface connects UPF to data network."
        ),
        "keywords": ["UPF", "packet routing", "N3", "N9", "N6", "data network", "QoS"]
    },
    # TS 23.502 - 5G Procedures
    {
        "spec_id": "TS 23.502",
        "title": "Procedures for the 5G System (5GS)",
        "section": "4.2.2 Registration procedures",
        "content": (
            "Registration procedure types: Initial Registration (first attach or after deregistration), "
            "Mobility Registration Update (TAI change, inter-system change), "
            "Periodic Registration Update (mobile reachable timer expiry). "
            "Steps: 1) UE sends Registration Request via NAS, 2) AMF performs authentication via AUSF, "
            "3) NAS Security Mode Command procedure, 4) AMF retrieves UE context from old AMF if needed, "
            "5) AMF performs Policy Association with PCF, 6) Registration Accept sent to UE. "
            "Causes for registration failure include: PLMN not allowed, illegal UE, "
            "5GS services not allowed, congestion."
        ),
        "keywords": ["registration", "NAS", "AUSF", "authentication", "TAI", "Registration Accept"]
    },
    {
        "spec_id": "TS 23.502",
        "title": "Procedures for the 5G System (5GS)",
        "section": "4.3.2 PDU Session Establishment",
        "content": (
            "PDU Session Establishment procedure: 1) UE sends PDU Session Establishment Request to AMF "
            "(includes PDU session ID, requested S-NSSAI, DNN, PDU session type, SSC mode), "
            "2) AMF selects SMF based on S-NSSAI, DNN, and local policy, "
            "3) SMF creates SM context and selects UPF, 4) SMF establishes N4 session with UPF via PFCP, "
            "5) SMF sends N1N2MessageTransfer to AMF with NAS and N2 info, "
            "6) AMF sends N2 PDU Session Request to RAN, 7) RAN sets up DRB and responds, "
            "8) UPF forwarding rules activated. QoS flows are established with 5QI values "
            "defining delay, error rate, and priority characteristics."
        ),
        "keywords": ["PDU session", "S-NSSAI", "DNN", "UPF", "PFCP", "QoS flow", "5QI", "DRB"]
    },
    # TS 24.501 - NAS Protocol
    {
        "spec_id": "TS 24.501",
        "title": "Non-Access-Stratum (NAS) protocol for 5G System (5GS)",
        "section": "5.5.1 5GS Registration Procedures",
        "content": (
            "The UE initiates registration by sending REGISTRATION REQUEST message containing: "
            "5GS registration type, 5GS mobile identity (SUCI for initial, 5G-GUTI for subsequent), "
            "requested NSSAI, UE security capability, UE network capability. "
            "NAS security establishment includes: Authentication (5G-AKA or EAP-AKA'), "
            "NAS Security Mode Command (activates NAS ciphering and integrity), "
            "followed by REGISTRATION ACCEPT with assigned 5G-GUTI, TAI list, allowed NSSAI. "
            "Registration reject causes (5GMM cause values): #3 Illegal UE, #6 Illegal ME, "
            "#7 5GS services not allowed, #11 PLMN not allowed, #73 Serving network not authorized."
        ),
        "keywords": ["NAS", "SUCI", "5G-GUTI", "NSSAI", "5G-AKA", "registration", "5GMM cause"]
    },
    # TS 38.321 - MAC Protocol
    {
        "spec_id": "TS 38.321",
        "title": "NR; Medium Access Control (MAC) protocol specification",
        "section": "5.1 Random Access procedure",
        "content": (
            "The Random Access procedure is used for: initial access, handover target cell access, "
            "transition from RRC_INACTIVE, beam failure recovery, scheduling request if no PUCCH resources, "
            "timing advance alignment. Two types: Contention-Based (4-step RACH: Msg1 preamble, "
            "Msg2 RAR, Msg3 RRC/NAS, Msg4 contention resolution) and Contention-Free (2-step: "
            "dedicated preamble, RAR). CFRA is used during handover for faster access. "
            "Key parameters: ra-PreambleIndex, ra-ResponseWindow, preambleReceivedTargetPower. "
            "2-step RACH (msgA/msgB) reduces latency for URLLC applications."
        ),
        "keywords": ["RACH", "random access", "preamble", "contention", "beam failure recovery", "CFRA"]
    },
    # TS 38.211/212/213/214 - Physical Layer
    {
        "spec_id": "TS 38.214",
        "title": "NR; Physical layer procedures for data",
        "section": "5.1 Downlink link adaptation",
        "content": (
            "The gNB determines the modulation and coding scheme (MCS) for PDSCH based on "
            "CSI (Channel State Information) reports from the UE. CSI includes: CQI (Channel Quality Indicator), "
            "PMI (Precoding Matrix Indicator), RI (Rank Indicator), and LI (Layer Indicator). "
            "The UE reports CSI via PUCCH or PUSCH. CQI maps to a target BLER of 10% for the selected MCS. "
            "HARQ (Hybrid ARQ) provides error correction with soft combining. "
            "Up to 16 HARQ processes can operate in parallel for the DL."
        ),
        "keywords": ["PDSCH", "MCS", "CSI", "CQI", "HARQ", "link adaptation", "BLER"]
    },
    {
        "spec_id": "TS 38.213",
        "title": "NR; Physical layer procedures for control",
        "section": "8.1 PUCCH Resource allocation",
        "content": (
            "PUCCH carries Uplink Control Information (UCI) including HARQ-ACK, CSI, and Scheduling Request. "
            "PUCCH formats: Format 0 (1-2 bits, short), Format 1 (1-2 bits, long), "
            "Format 2 (>2 bits, short), Format 3 (>2 bits, long, no frequency hopping), "
            "Format 4 (>2 bits, long, with frequency hopping and multiplexing). "
            "PDCCH (Physical Downlink Control Channel) carries DCI (Downlink Control Information) "
            "for scheduling grants. CORESET (Control Resource Set) defines time-frequency resources for PDCCH. "
            "Search spaces can be common (broadcast) or UE-specific."
        ),
        "keywords": ["PUCCH", "PDCCH", "UCI", "DCI", "CORESET", "scheduling", "HARQ-ACK"]
    },
    # TS 38.401 - NG-RAN Architecture
    {
        "spec_id": "TS 38.401",
        "title": "NG-RAN; Architecture description",
        "section": "6.1 gNB split architecture",
        "content": (
            "The gNB can be split into: gNB-CU (Central Unit) handling RRC and PDCP, "
            "and gNB-DU (Distributed Unit) handling RLC, MAC, and PHY layers. "
            "The F1 interface connects CU and DU. gNB-CU can be further split into: "
            "gNB-CU-CP (Control Plane, handles RRC/PDCP-C) and gNB-CU-UP (User Plane, handles PDCP-U/SDAP). "
            "E1 interface connects CU-CP and CU-UP. This split enables: centralized RRM, "
            "multi-vendor interoperability, flexible deployment (DU at cell site, CU at edge/core), "
            "and easier scaling of capacity vs coverage."
        ),
        "keywords": ["gNB-CU", "gNB-DU", "F1 interface", "E1 interface", "CU-CP", "CU-UP", "split architecture"]
    },
    # TS 38.423 - Xn Application Protocol
    {
        "spec_id": "TS 38.423",
        "title": "NG-RAN; Xn Application Protocol (XnAP)",
        "section": "8.4 Mobility procedures",
        "content": (
            "Xn-based handover procedure: 1) Source gNB sends HANDOVER REQUEST to target gNB "
            "(includes UE context, PDU sessions, target cell ID, UE security capabilities), "
            "2) Target gNB performs admission control and allocates resources, "
            "3) Target gNB responds with HANDOVER REQUEST ACKNOWLEDGE (includes RRC container "
            "with target cell configuration), 4) Source gNB sends RRCReconfiguration to UE, "
            "5) SN STATUS TRANSFER for PDCP SN synchronization, "
            "6) UE connects to target, sends RRCReconfigurationComplete, "
            "7) PATH SWITCH REQUEST to AMF via NG interface for CN path update."
        ),
        "keywords": ["Xn handover", "HANDOVER REQUEST", "admission control", "PATH SWITCH", "SN STATUS"]
    },
    # IMS / VoNR
    {
        "spec_id": "TS 24.229",
        "title": "IP multimedia call control protocol based on SIP and SDP",
        "section": "5.1 Registration procedures",
        "content": (
            "IMS registration for VoNR: UE sends SIP REGISTER to P-CSCF (discovered via PCO in PDU session), "
            "P-CSCF forwards to I-CSCF, I-CSCF queries HSS/UDM for S-CSCF assignment, "
            "S-CSCF performs authentication (IMS AKA), sends 200 OK. "
            "VoNR call setup: UE sends SIP INVITE with SDP offer (codec, QoS), "
            "P-CSCF interacts with PCF for dedicated QoS flow (5QI=1 for conversational voice), "
            "Network establishes dedicated bearer with GBR. "
            "Common failure: SIP 503 Service Unavailable indicates IMS core overload or P-CSCF unreachable."
        ),
        "keywords": ["VoNR", "SIP", "IMS", "P-CSCF", "S-CSCF", "SDP", "503", "QoS flow"]
    },
]

# ============================================================================
# PURE-PYTHON BM25 RETRIEVER CLASS
# ============================================================================
class BM25Retriever:
    """
    Pure-Python BM25 retriever with zero external dependencies.
    Optimized for fast keyword matching over 3GPP specification chunks.
    """
    def __init__(self, corpus: List[Dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        
        self.doc_lens = []
        self.doc_term_freqs = []
        self.df = Counter()
        
        for doc in corpus:
            # Tokenize content, section title, spec ID, and keywords combined
            keywords_str = " ".join(doc.get("keywords", []))
            text = f"{doc['spec_id']} {doc.get('section', '')} {doc['content']} {keywords_str}"
            tokens = self._tokenize(text)
            self.doc_lens.append(len(tokens))
            self.doc_term_freqs.append(Counter(tokens))
            
            for token in set(tokens):
                self.df[token] += 1
                
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)
        
        # Calculate IDF scores
        self.idf = {}
        for term, freq in self.df.items():
            # Standard BM25 IDF formula with smoothing
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def _tokenize(self, text: str) -> List[str]:
        """Convert string to standard normalized tokens."""
        return re.findall(r'[a-zA-Z0-9]{2,}', text.lower())

    def retrieve(self, query: str, top_k: int = 15) -> List[Tuple[float, int]]:
        """
        Calculates BM25 lexical scores for all documents.
        Returns list of (score, doc_index) sorted descending.
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_lens[idx]
            term_freqs = self.doc_term_freqs[idx]
            
            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = term_freqs[token]
                idf = self.idf[token]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (num / den)
                
            scores.append((score, idx))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]


# ============================================================================
# MAIN SPEC RETRIEVER CLASS
# ============================================================================
class SpecRetriever:
    """
    State-of-the-Art (SOTA) 3GPP specification retriever.
    - Implements Hybrid Search (FAISS Dense Vector + BM25 Lexical).
    - Fuses rankings using Reciprocal Rank Fusion (RRF).
    - Refines candidates via Cross-Encoder Reranking if available.
    - Gracefully falls back to TF-IDF/BM25 on CPU nodes.
    """

    def __init__(self, use_embeddings: bool = None):
        """
        Args:
            use_embeddings: True to force FAISS vector search.
                            False to force TF-IDF mode.
                            None (default) to auto-detect.
        """
        self.knowledge_base = SPEC_KNOWLEDGE_BASE
        self.use_embeddings = use_embeddings
        self.index = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.embed_model = None
        self.bm25 = None
        self.reranker = None
        self.has_reranker = False

        # Directory configurations
        self.index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index")
        self.faiss_path = os.path.join(self.index_dir, "index.faiss")
        self.metadata_path = os.path.join(self.index_dir, "metadata.pkl")

        if self.use_embeddings is None:
            self._autodetect_mode()

        self._load_or_build_index()

    def _autodetect_mode(self):
        """Checks if precompiled index files and libraries are present."""
        has_index_files = os.path.exists(self.faiss_path) and os.path.exists(self.metadata_path)
        if not has_index_files:
            self.use_embeddings = False
            return

        try:
            import sentence_transformers
            import faiss
            self.use_embeddings = True
        except ImportError:
            self.use_embeddings = False

    def _load_or_build_index(self):
        """Loads index from disk or builds TF-IDF index from core specifications."""
        if self.use_embeddings:
            success = self._load_embedding_index()
            if not success:
                print("[WARNING] Failed to load FAISS index. Falling back to TF-IDF...")
                self.use_embeddings = False
                self._build_tfidf_index()
        else:
            self._build_tfidf_index()
            
        # Build BM25 index over the loaded specifications corpus (runs in all modes!)
        self.bm25 = BM25Retriever(self.knowledge_base)

    def _load_embedding_index(self) -> bool:
        """Loads FAISS index and chunk metadata from disk."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            print(f"[LOADING] FAISS vector index from: {self.faiss_path}")
            self.index = faiss.read_index(self.faiss_path)

            print(f"[LOADING] RAG chunk metadata from: {self.metadata_path}")
            with open(self.metadata_path, 'rb') as f:
                self.knowledge_base = pickle.load(f)

            print("[LOADING] Instantiating lightweight SentenceTransformer model ('all-MiniLM-L6-v2')...")
            # Set cache dir to local directory
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.path.dirname(self.index_dir), "cache")
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load Cross-Encoder reranker if installed
            try:
                from sentence_transformers import CrossEncoder
                print("[LOADING] Instantiating Cross-Encoder reranker ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.has_reranker = True
                print("[SUCCESS] Cross-Encoder reranker loaded successfully.")
            except Exception as e:
                print(f"[WARNING] Reranker loading skipped or failed ({e}). Proceeding without reranking.")
                self.reranker = None
                self.has_reranker = False
                
            dim = self.index.d
            print(f"[SUCCESS] Real RAG active: {len(self.knowledge_base)} chunks loaded ({dim}-dim vectors).")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load embedding index: {e}")
            return False

    def _build_tfidf_index(self):
        """Build a TF-IDF based search index (fallback strategy)."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            documents = []
            for entry in self.knowledge_base:
                doc = f"{entry['spec_id']} {entry['title']} {entry['section']} {entry['content']} {' '.join(entry.get('keywords', []))}"
                documents.append(doc)

            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
            print(f"[SUCCESS] Fallback RAG active: Built TF-IDF index for {len(documents)} static chunks.")

        except ImportError:
            print("[WARNING] scikit-learn not available. Using keyword-based fallback retrieval.")
            self.vectorizer = None

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the most relevant 3GPP specifications using Hybrid Search.
        Fuses dense vector/TF-IDF rankings with BM25 lexical rankings.
        """
        # 1. Fetch dense / TF-IDF candidates
        if self.use_embeddings and self.embed_model is not None and self.index is not None:
            dense_indices = self._retrieve_dense_raw(query, top_k=15)
        elif self.vectorizer is not None:
            dense_indices = self._retrieve_tfidf_raw(query, top_k=15)
        else:
            dense_indices = []

        # 2. Fetch BM25 lexical candidates
        bm25_results = self.bm25.retrieve(query, top_k=15)
        sparse_indices = [idx for _, idx in bm25_results]

        # 3. Reciprocal Rank Fusion (RRF)
        # Combine dense and sparse lists
        dense_rank = {idx: rank for rank, idx in enumerate(dense_indices)}
        sparse_rank = {idx: rank for rank, idx in enumerate(sparse_indices)}
        
        all_candidates = set(dense_indices) | set(sparse_indices)
        
        # Constant k is 60 by standard RRF design
        k = 60
        rrf_scores = []
        for idx in all_candidates:
            # Map index to rank (use 999 if not present in one of the search modes)
            d_rank = dense_rank.get(idx, 999)
            s_rank = sparse_rank.get(idx, 999)
            
            rrf_score = (1.0 / (k + d_rank + 1)) + (1.0 / (k + s_rank + 1))
            rrf_scores.append((rrf_score, idx))
            
        rrf_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Take the top 15 fused candidates for potential reranking
        top_candidates = rrf_scores[:15]
        
        # 4. Optional: Cross-Encoder Reranking
        final_indices_with_scores = []
        if self.has_reranker and self.reranker is not None and len(top_candidates) > 0:
            try:
                candidate_indices = [idx for _, idx in top_candidates]
                # Prepare query-chunk pairs
                pairs = [[query, self.knowledge_base[idx]["content"]] for idx in candidate_indices]
                rerank_scores = self.reranker.predict(pairs)
                
                # Sort candidate indices by their rerank scores
                reranked = sorted(zip(rerank_scores, candidate_indices), key=lambda x: x[0], reverse=True)
                # Normalize scores to 0-1 scale for RAG interface
                for r_score, idx in reranked[:top_k]:
                    # Map raw logit (typically -5 to 5) to a pseudo-prob using sigmoid
                    prob = 1.0 / (1.0 + math.exp(-r_score))
                    final_indices_with_scores.append((prob, idx))
            except Exception as e:
                # Fallback to pure RRF if reranker fails at runtime
                final_indices_with_scores = [(score, idx) for score, idx in top_candidates[:top_k]]
        else:
            # Fallback to standard RRF results
            final_indices_with_scores = [(score, idx) for score, idx in top_candidates[:top_k]]

        # 5. Build final response dictionaries
        results = []
        for score, idx in final_indices_with_scores:
            entry = self.knowledge_base[idx]
            results.append({
                "spec_id": entry["spec_id"],
                "title": entry["title"],
                "section": entry["section"],
                "content": entry["content"],
                "keywords": entry.get("keywords", []),
                "relevance_score": float(score)
            })
            
        return results

    def _retrieve_dense_raw(self, query: str, top_k: int) -> List[int]:
        """Runs vector search and returns document index list."""
        import numpy as np
        import faiss

        query_embedding = self.embed_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        k = min(top_k, len(self.knowledge_base))
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        return [idx for idx in indices[0] if idx >= 0]

    def _retrieve_tfidf_raw(self, query: str, top_k: int) -> List[int]:
        """Runs TF-IDF cosine similarity search and returns index list."""
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        k = min(top_k, len(self.knowledge_base))
        top_indices = similarities.argsort()[::-1][:k]
        return list(top_indices)

    def format_context(self, results: List[Dict], max_chars: int = 2000) -> str:
        """Format retrieved results into a context string."""
        context_parts = []
        total_chars = 0

        for r in results:
            chunk = (
                f"[{r['spec_id']} - {r['section']}]\n"
                f"{r['content']}\n"
            )
            if total_chars + len(chunk) > max_chars:
                break
            context_parts.append(chunk)
            total_chars += len(chunk)

        if context_parts:
            return (
                "\n--- RETRIEVED 3GPP SPECIFICATION CONTEXT ---\n"
                + "\n".join(context_parts)
                + "\n--- END SPECIFICATION CONTEXT ---\n"
            )
        return ""


def build_augmented_system_prompt(
    base_prompt: str,
    user_query: str,
    retriever: Optional['SpecRetriever'] = None,
    top_k: int = 3
) -> str:
    """Build a RAG-augmented system prompt by retrieving relevant 3GPP specs."""
    if retriever is None:
        return base_prompt

    results = retriever.retrieve(user_query, top_k=top_k)
    context = retriever.format_context(results)

    if context:
        return (
            f"{base_prompt}\n\n"
            f"You have access to the following relevant 3GPP specification excerpts. "
            f"Use them to provide accurate, specification-compliant analysis:\n"
            f"{context}"
        )
    return base_prompt


if __name__ == "__main__":
    print("======================================================================")
    print("3GPP Specification RAG Module - SOTA Hybrid Self-Test")
    print("======================================================================")

    # Initialize retriever
    retriever = SpecRetriever()

    test_queries = [
        "My phone drops data service when moving between cell tower sectors",
        "VoNR call setup failing with SIP 503 errors on 5G SA network",
    ]

    for query in test_queries:
        print(f"\nQUERY: {query}")
        results = retriever.retrieve(query, top_k=2)
        for i, r in enumerate(results):
            print(f"  [{i+1}] {r['spec_id']} - {r['section']} (Fused Score: {r['relevance_score']:.4f})")
            print(f"      Content: {r['content'][:120]}...")
