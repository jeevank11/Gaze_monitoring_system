"""Shared Streamlit theme.

Both dashboard pages call :func:`inject_theme` immediately after
``st.set_page_config`` so the sidebar, metric tiles, and typography stay
identical across the multipage app. Extracted from the original inline
stylesheet in ``dashboard/app.py``.
"""

from __future__ import annotations

import streamlit as st

_THEME_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"], .stApp, .stMarkdown, .stMetric,
  .stButton>button, .stSelectbox, .stTextInput, .stCheckbox,
  section[data-testid="stSidebar"],
  section[data-testid="stSidebar"] *,
  [data-testid="stSidebarNav"],
  [data-testid="stSidebarNav"] * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  }
  code, pre, .stCode {
    font-family: 'JetBrains Mono', 'Consolas', 'Menlo', monospace !important;
  }

  .block-container { padding-top: 1.75rem; padding-bottom: 3rem; }

  .ga-hero {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 22px; margin-bottom: 18px;
    border-radius: 14px;
    background: linear-gradient(135deg, #0f1c3f 0%, #1e3a8a 55%, #0ea5e9 100%);
    color: white;
    box-shadow: 0 6px 24px rgba(15, 28, 63, 0.18);
  }
  .ga-hero .ga-badge {
    width: 42px; height: 42px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(255,255,255,0.14);
    font-size: 22px;
  }
  .ga-hero h1 {
    margin: 0; font-size: 1.55rem; font-weight: 700; letter-spacing: -0.01em;
  }
  .ga-hero .ga-sub {
    margin: 2px 0 0; opacity: 0.82; font-size: 0.9rem;
  }

  h2, h3 { font-weight: 600 !important; letter-spacing: -0.01em; color: #0f172a; }

  [data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    transition: box-shadow 0.15s ease, transform 0.15s ease;
  }
  [data-testid="stMetric"]:hover {
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    transform: translateY(-1px);
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b !important;
    font-weight: 500 !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    line-height: 1.1;
  }

  .stButton>button {
    border-radius: 8px;
    font-weight: 500;
    transition: transform 0.05s ease;
  }
  .stButton>button:active { transform: translateY(1px); }

  section[data-testid="stSidebar"] { background: #f8fafc; }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #0f172a; font-weight: 600; }

  /* Multipage nav links (sidebar page list) */
  [data-testid="stSidebarNav"] a,
  [data-testid="stSidebarNav"] span,
  [data-testid="stSidebarNav"] p {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: -0.005em;
  }

  hr { margin-top: 1.25rem !important; margin-bottom: 1.25rem !important; }
</style>
"""


def inject_theme() -> None:
    """Inject the shared stylesheet. Call once per page after set_page_config."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
