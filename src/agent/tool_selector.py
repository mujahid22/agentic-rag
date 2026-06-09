import re
from .tools import ALL_TOOLS
from src.utils.query_logger import log_tool_selection

_PUNCT_RE = re.compile(r"[^\w\s]")

# Regex patterns that force-include a specific tool regardless of keyword score
_FORCED_TOOLS: list[tuple[re.Pattern, str]] = [
    # Age expressions like "3 years old", "18 months old", "2-year-old"
    (re.compile(r'\b\d+[\s-]*(year|month|yr)s?[\s-]*old\b', re.I), "tool_family_travel"),
    # Explicit infant/toddler/baby/newborn words not caught by token split
    (re.compile(r'\b(toddler|newborn|infant|baby|babies)\b', re.I), "tool_family_travel"),
    # "travel with my son/daughter/kid"
    (re.compile(r'\b(son|daughter|kid|kids|toddler)\b', re.I), "tool_family_travel"),
]

# Keyword sets used to score relevance of each tool against the user query
_TOOL_KEYWORDS: dict[str, set[str]] = {
    "tool_routes": {
        "route", "routes", "fly", "flies", "flight", "destination", "destinations",
        "from", "to", "hub", "hubs", "connect", "connection", "connecting", "layover",
        "stopover", "transit", "via", "through", "network", "serve", "served",
        "nonstop", "non-stop", "direct", "indirect", "duration", "how long",
        "which airport", "best way", "travel from", "travel to", "go to",
        "tokyo", "london", "dubai", "dallas", "new york", "los angeles", "miami",
        "sydney", "singapore", "seoul", "paris", "frankfurt", "amsterdam",
        "dfw", "jfk", "lax", "lhr", "dxb", "nrt", "hnd", "icn", "sin", "syd",
        "mia", "ord", "atl", "sea", "den", "bos",
    },
    "tool_booking_reservations": {
        "book", "booking", "reservation", "reserve", "pnr", "ticket", "purchase",
        "buy", "payment", "pay", "itinerary", "confirm", "name", "modify",
        "change", "group", "multi", "city", "hold", "channel", "website", "app",
        "e-ticket", "eticket",
    },
    "tool_cancellation_refunds": {
        "cancel", "cancellation", "refund", "void", "credit", "no-show", "noshow",
        "unused", "forfeiture", "waiver", "reimburse", "money", "back", "fee",
        "penalty", "reimbursement",
    },
    "tool_baggage": {
        "baggage", "bag", "luggage", "suitcase", "carry", "on", "checked",
        "excess", "oversize", "overweight", "allowance", "special", "sports",
        "musical", "instrument", "equipment", "tag", "weight", "kg", "kilo",
    },
    "tool_checkin_boarding": {
        "check", "checkin", "check-in", "boarding", "seat", "selection", "pass",
        "gate", "departure", "document", "passport", "id", "kiosk", "online",
        "mobile", "deadline", "counter",
    },
    "tool_fare_ticketing": {
        "fare", "class", "economy", "business", "first", "upgrade", "waitlist",
        "standby", "validity", "flexible", "saver", "flex", "open", "jaw",
        "stopover", "rules", "restriction",
    },
    "tool_frequent_flyer": {
        "miles", "points", "frequent", "flyer", "ffp", "meridian", "tier",
        "gold", "silver", "platinum", "earn", "redeem", "redemption", "award",
        "partner", "status", "benefit", "accrual", "account",
    },
    "tool_disruptions_compensation": {
        "delay", "delayed", "disruption", "irregular", "irops", "compensation",
        "denied", "boarding", "weather", "rebook", "voucher", "hotel",
        "overnight", "stranded", "missed", "connection", "cancelled by airline",
    },
    "tool_special_services": {
        "wheelchair", "disability", "assistance", "special", "needs", "unaccompanied",
        "minor", "umnr", "medical", "oxygen", "stretcher", "clearance", "meal",
        "dietary", "kosher", "halal", "vegan", "gluten", "allergy", "allergies",
    },
    "tool_ancillary_services": {
        "upgrade", "lounge", "pass", "inflight", "entertainment", "wifi",
        "wi-fi", "priority", "fast", "track", "legroom", "addon", "add-on",
        "purchase", "extra",
    },
    "tool_lounge_airport": {
        "lounge", "airport", "transfer", "parking", "shuttle", "valet",
        "arrival", "transit", "access", "location",
    },
    "tool_partnerships": {
        "codeshare", "alliance", "partner", "interline", "code", "share",
        "reciprocal", "through", "check-in", "other airline", "operated by",
    },
    "tool_travel_requirements": {
        "visa", "passport", "document", "entry", "requirement", "health",
        "vaccination", "transit", "immigration", "customs", "destination",
        "country", "allowed", "permitted",
    },
    "tool_family_travel": {
        "infant", "baby", "child", "children", "minor", "family", "bassinet",
        "lap", "fares", "seating", "together", "age", "kids", "toddler",
        "newborn", "son", "daughter", "old", "years", "months",
    },
    "tool_corporate_travel": {
        "corporate", "company", "account", "invoice", "negotiated",
        "manager", "tmc", "expense", "reporting", "contract",
    },
    "tool_onboard_conduct": {
        "safety", "seatbelt", "belt", "emergency", "conduct", "behavior",
        "smoking", "alcohol", "disruptive", "onboard", "inflight", "rules",
        "regulations",
    },
}

_DEFAULT_TOOLS = ["tool_booking_reservations", "tool_fare_ticketing", "tool_cancellation_refunds"]


def select_tools(query: str, top_k: int = 2) -> list[str]:
    query_tokens = set(_PUNCT_RE.sub("", query.lower()).split())

    scores: dict[str, int] = {
        name: len(query_tokens & keywords)
        for name, keywords in _TOOL_KEYWORDS.items()
    }

    # Regex-based forced inclusions — boost score to guarantee selection
    for pattern, tool_name in _FORCED_TOOLS:
        if pattern.search(query):
            scores[tool_name] = max(scores.get(tool_name, 0), 99)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    selected = [name for name, score in ranked if score > 0][:top_k]

    if len(selected) < 2:
        selected = [name for name, _ in ranked[:top_k]]

    log_tool_selection(selected)
    return selected
