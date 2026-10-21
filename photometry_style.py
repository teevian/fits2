"""Scoped light theme: the host application's global dark rules must not leak into this dialog."""
STYLE = '''<span class="photometry-dialog-style"></span><style>
div[role="dialog"]:has(.photometry-dialog-style) {
 background:#fff !important;color:#17202a !important;color-scheme:light !important;
 max-width:1040px !important;
 --text-color:#17202a;--background-color:#fff;--secondary-background-color:#f4f6f8;
}
div[role="dialog"]:has(.photometry-dialog-style) p,
div[role="dialog"]:has(.photometry-dialog-style) label,
div[role="dialog"]:has(.photometry-dialog-style) h1,
div[role="dialog"]:has(.photometry-dialog-style) h2,
div[role="dialog"]:has(.photometry-dialog-style) h3,
div[role="dialog"]:has(.photometry-dialog-style) [data-testid="stMarkdownContainer"],
div[role="dialog"]:has(.photometry-dialog-style) [data-testid="stWidgetLabel"] {color:#17202a !important;opacity:1 !important;}
div[role="dialog"]:has(.photometry-dialog-style) [data-testid="stCaptionContainer"],
div[role="dialog"]:has(.photometry-dialog-style) [data-testid="stCaptionContainer"] p {color:#455468 !important;opacity:1 !important;font-size:12px;}
div[role="dialog"]:has(.photometry-dialog-style) [data-testid="stExpander"] details,
div[role="dialog"]:has(.photometry-dialog-style) [data-testid="stExpanderDetails"],
div[role="dialog"]:has(.photometry-dialog-style) [data-testid="stVerticalBlockBorderWrapper"] {background:#fff !important;color:#17202a !important;}
div[role="dialog"]:has(.photometry-dialog-style) summary,
div[role="dialog"]:has(.photometry-dialog-style) summary span {background:#f3f6fa !important;color:#17202a !important;}
div[role="dialog"]:has(.photometry-dialog-style) button,
div[role="dialog"]:has(.photometry-dialog-style) button:hover,
div[role="dialog"]:has(.photometry-dialog-style) button:focus-visible {background:#edf2f7 !important;color:#17202a !important;border-color:#a6b4c4 !important;min-height:30px;padding:4px 10px;}
div[role="dialog"]:has(.photometry-dialog-style) button[kind="primary"] {background:#245b88 !important;color:white !important;}
div[role="dialog"]:has(.photometry-dialog-style) button[kind="primary"] p {color:white !important;}
div[role="dialog"]:has(.photometry-dialog-style) button svg {fill:#17202a !important;color:#17202a !important;}
div[role="dialog"]:has(.photometry-dialog-style) button[aria-label="Close"] {padding:4px !important;width:32px !important;height:32px !important;}
div[role="dialog"]:has(.photometry-dialog-style) button[aria-label="Close"] svg {width:20px !important;height:20px !important;flex-shrink:0;}
div[role="dialog"]:has(.photometry-dialog-style) input,
div[role="dialog"]:has(.photometry-dialog-style) textarea {background:#f4f6f8 !important;color:#17202a !important;-webkit-text-fill-color:#17202a !important;caret-color:#17202a;min-height:30px;font-size:13px;padding:4px 8px;}
div[role="dialog"]:has(.photometry-dialog-style) input::placeholder {color:#66758a !important;-webkit-text-fill-color:#66758a !important;opacity:1;}
div[role="dialog"]:has(.photometry-dialog-style) [data-baseweb="input"],
div[role="dialog"]:has(.photometry-dialog-style) [data-baseweb="input"]>div,
div[role="dialog"]:has(.photometry-dialog-style) [data-baseweb="select"]>div,
div[role="dialog"]:has(.photometry-dialog-style) [data-baseweb="select"] span,
div[role="dialog"]:has(.photometry-dialog-style) [data-baseweb="select"] div {background:#f4f6f8 !important;color:#17202a !important;min-height:0;}
div[role="dialog"]:has(.photometry-dialog-style) [data-baseweb="select"] svg {fill:#17202a !important;}
body:has(.photometry-dialog-style) [data-baseweb="popover"] [role="listbox"],
body:has(.photometry-dialog-style) [data-baseweb="popover"] [role="option"],
body:has(.photometry-dialog-style) [data-baseweb="popover"] li {background:#fff !important;color:#17202a !important;}
div[role="dialog"]:has(.photometry-dialog-style) [data-testid="stAlert"] {background:#edf3fa !important;color:#17202a !important;}
div[role="dialog"]:has(.photometry-dialog-style) a {color:#175d97 !important;}
div[role="dialog"]:has(.photometry-dialog-style) [data-testid="stNumberInput"] button {display:none !important;}
div[role="dialog"]:has(.photometry-dialog-style) [data-testid="stVerticalBlock"] {gap:.4rem !important;}
.photometry-table-wrap {overflow-x:auto;background:white;color:#17202a;}
.photometry-results {background:white;color:#17202a;font-size:12px;white-space:nowrap;}
.photometry-results th,.photometry-results td {padding:6px;border-bottom:1px solid #dce3eb;}
</style>'''
