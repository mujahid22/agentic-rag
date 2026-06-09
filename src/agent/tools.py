from langchain_core.tools import tool
from src.retrieval.hybrid_retriever import retrieve
from src.retrieval.types import RetrievalResult


def _format_results(results: list[RetrievalResult], route: str, reranked: bool) -> str:
    if not results:
        return "No relevant information found in Meridian Airlines documentation."
    rerank_label = "reranked" if reranked else "direct (high-confidence)"
    header = f"[Retrieval: {route} · {rerank_label}]"
    parts = [header]
    for r in results:
        citation = f"[Source: {r.source_doc} | Section: {r.section} | Effective: {r.effective_date}]"
        parts.append(f"{citation}\n{r.text}")
    return "\n\n---\n\n".join(parts)


@tool
def tool_booking_reservations(query: str) -> str:
    """
    Use for questions about: flight booking, reservations, PNR codes, e-tickets, payment
    methods, booking channels, booking modifications, name corrections, group bookings,
    booking windows, fare hold options, multi-city bookings, and itinerary rules.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Booking & Reservations"})
    return _format_results(results, route, reranked)


@tool
def tool_cancellation_refunds(query: str) -> str:
    """
    Use for questions about: cancelling a flight, refund eligibility, cancellation fees,
    no-show policy, unused tickets, partial cancellations, refund processing times,
    and fare credit vouchers.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Cancellations & Refunds"})
    return _format_results(results, route, reranked)


@tool
def tool_baggage(query: str) -> str:
    """
    Use for questions about: carry-on baggage, checked baggage, baggage allowances,
    oversize or overweight fees, excess baggage, special items, sports equipment,
    musical instruments, and baggage tag numbers.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Baggage Policy"})
    return _format_results(results, route, reranked)


@tool
def tool_checkin_boarding(query: str) -> str:
    """
    Use for questions about: online check-in, airport check-in, boarding procedures,
    seat selection, boarding passes, gate information, travel documents, passport
    requirements, and check-in deadlines.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Check-in & Boarding"})
    return _format_results(results, route, reranked)


@tool
def tool_fare_ticketing(query: str) -> str:
    """
    Use for questions about: fare classes (Economy Saver, Economy Flex, Business, First),
    fare rules, ticket validity, upgrades, waitlist, standby, open-jaw tickets,
    stopovers, and fare differences.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Fare Classes & Ticketing"})
    return _format_results(results, route, reranked)


@tool
def tool_frequent_flyer(query: str) -> str:
    """
    Use for questions about: Meridian Miles program, earning miles, redeeming miles,
    tier status (Silver, Gold, Platinum), tier benefits, partner miles, award tickets,
    FFP account numbers (MA-XXXXXXXX), and miles expiry.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Frequent Flyer Program"})
    return _format_results(results, route, reranked)


@tool
def tool_disruptions_compensation(query: str) -> str:
    """
    Use for questions about: flight delays, flight cancellations by the airline,
    irregular operations, denied boarding, weather disruptions, compensation
    entitlements, rebooking rights, meal vouchers, and hotel accommodation.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Disruptions & Compensation"})
    return _format_results(results, route, reranked)


@tool
def tool_special_services(query: str) -> str:
    """
    Use for questions about: wheelchair assistance, passengers with disabilities,
    unaccompanied minors (UMNR), medical clearance, oxygen on board, special meals,
    dietary requirements (kosher, halal, vegan, gluten-free), and allergy requests.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Special Services"})
    return _format_results(results, route, reranked)


@tool
def tool_ancillary_services(query: str) -> str:
    """
    Use for questions about: paid seat upgrades, lounge day passes, in-flight
    entertainment, Wi-Fi, priority boarding, fast-track security, extra legroom seats,
    and other purchased add-ons.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Ancillary Services"})
    return _format_results(results, route, reranked)


@tool
def tool_lounge_airport(query: str) -> str:
    """
    Use for questions about: Meridian lounge locations, lounge access rules,
    airport transfers, airport parking, valet parking, shuttle services,
    and arrival/transit lounge access.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Airport Services"})
    return _format_results(results, route, reranked)


@tool
def tool_partnerships(query: str) -> str:
    """
    Use for questions about: codeshare flights, alliance partners, interline agreements,
    partner airline policies, through check-in on partner flights, and reciprocal
    benefits with other carriers.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Codeshare & Alliances"})
    return _format_results(results, route, reranked)


@tool
def tool_travel_requirements(query: str) -> str:
    """
    Use for questions about: visa requirements, passport validity, transit visas,
    travel health requirements, vaccinations, entry requirements by destination,
    and immigration documentation.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Travel Requirements"})
    return _format_results(results, route, reranked)


@tool
def tool_family_travel(query: str) -> str:
    """
    Use for questions about: travelling with infants, lap infant policy, bassinets,
    child fares, children travelling alone, family seating arrangements, and
    age requirements for minors.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Family & Child Travel"})
    return _format_results(results, route, reranked)


@tool
def tool_corporate_travel(query: str) -> str:
    """
    Use for questions about: corporate accounts, negotiated fares, business travel
    management, company invoicing, travel management company (TMC) access,
    and corporate tier benefits.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Corporate Travel"})
    return _format_results(results, route, reranked)


@tool
def tool_onboard_conduct(query: str) -> str:
    """
    Use for questions about: onboard safety rules, seatbelt policy, smoking policy,
    alcohol service, disruptive passenger procedures, in-flight medical emergencies,
    and passenger conduct standards.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Onboard Experience & Safety"})
    return _format_results(results, route, reranked)


@tool
def tool_routes(query: str) -> str:
    """
    Use for questions about: flight routes, destinations served, hub airports,
    route from one city to another, connecting flights, layover cities,
    flight numbers, flight duration, transit options, best way to fly between
    two cities, whether Meridian flies to a specific city or country,
    minimum connection times, and route recommendations.
    """
    results, route, reranked = retrieve(query, metadata_filter={"section": "Route Network"})
    return _format_results(results, route, reranked)


ALL_TOOLS: dict[str, object] = {
    "tool_routes": tool_routes,
    "tool_booking_reservations": tool_booking_reservations,
    "tool_cancellation_refunds": tool_cancellation_refunds,
    "tool_baggage": tool_baggage,
    "tool_checkin_boarding": tool_checkin_boarding,
    "tool_fare_ticketing": tool_fare_ticketing,
    "tool_frequent_flyer": tool_frequent_flyer,
    "tool_disruptions_compensation": tool_disruptions_compensation,
    "tool_special_services": tool_special_services,
    "tool_ancillary_services": tool_ancillary_services,
    "tool_lounge_airport": tool_lounge_airport,
    "tool_partnerships": tool_partnerships,
    "tool_travel_requirements": tool_travel_requirements,
    "tool_family_travel": tool_family_travel,
    "tool_corporate_travel": tool_corporate_travel,
    "tool_onboard_conduct": tool_onboard_conduct,
}
