"""
Mock tool implementations for Project B.

order_tracker  — looks up orders from mock_data/orders.json
account_lookup — looks up customers from mock_data/customers.json

In Week 3 (LangGraph), these become real agent tool nodes.
For now they are called directly inside the pipeline when the router
selects order_tracker, account_lookup, or multi_tool.
"""
import json
import os
import re

MOCK_DIR = os.path.join(os.path.dirname(__file__), "..", "mock_data")


def _load_orders() -> list:
    with open(os.path.join(MOCK_DIR, "orders.json")) as f:
        return json.load(f)


def _load_customers() -> list:
    with open(os.path.join(MOCK_DIR, "customers.json")) as f:
        return json.load(f)


def extract_order_id(text: str) -> str | None:
    """Pull the first ORD-XXXXXX pattern out of a query string."""
    match = re.search(r"ORD-\d+", text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def lookup_order(order_id: str) -> dict | None:
    for order in _load_orders():
        if order["order_id"].upper() == order_id.upper():
            return order
    return None


def lookup_customer(customer_id: str) -> dict | None:
    for customer in _load_customers():
        if customer["id"] == customer_id:
            return customer
    return None


def lookup_customer_by_order(order_id: str) -> dict | None:
    order = lookup_order(order_id)
    if not order:
        return None
    return lookup_customer(order["customer_id"])


def _fmt_order(order: dict) -> str:
    return (
        f"[Order Data]\n"
        f"Order ID     : {order['order_id']}\n"
        f"Product      : {order['product']}\n"
        f"Price        : ₹{order['price']:,}\n"
        f"Order date   : {order['date']}\n"
        f"Delivery date: {order.get('delivery_date') or 'Not yet delivered'}\n"
        f"Status       : {order['status']}\n"
        f"Promotional  : {'Yes' if order.get('promotional') else 'No'}"
    )


def _fmt_customer(customer: dict) -> str:
    return (
        f"[Customer Data]\n"
        f"Name         : {customer['name']}\n"
        f"Tier         : {customer['tier'].title()}\n"
        f"Member since : {customer['since']}\n"
        f"Annual spend : ₹{customer['annual_spend']:,}"
    )


def run_order_tracker(query: str) -> str:
    """
    Extract order ID from query, fetch order + customer records.
    Returns formatted context string to prepend to RAG context.
    """
    order_id = extract_order_id(query)
    if not order_id:
        return "[Order Tracker] No order ID found in query — responding from policy only."

    order = lookup_order(order_id)
    if not order:
        return f"[Order Tracker] Order {order_id} not found in system."

    customer = lookup_customer_by_order(order_id)
    parts = [_fmt_order(order)]
    if customer:
        parts.append(_fmt_customer(customer))
    return "\n\n".join(parts)


def run_account_lookup(query: str) -> str:
    """
    Try to resolve customer via order ID in query.
    Returns formatted customer context or a no-data notice.
    """
    order_id = extract_order_id(query)
    if order_id:
        customer = lookup_customer_by_order(order_id)
        if customer:
            return _fmt_customer(customer)

    return "[Account Lookup] No customer ID in query — responding from policy only."


def run_multi_tool(query: str) -> str:
    """Run both order_tracker and account_lookup; deduplicate customer block."""
    order_id = extract_order_id(query)
    if not order_id:
        return "[Multi-Tool] No order ID found — responding from policy only."

    order = lookup_order(order_id)
    if not order:
        return f"[Multi-Tool] Order {order_id} not found in system."

    customer = lookup_customer(order["customer_id"])
    parts = [_fmt_order(order)]
    if customer:
        parts.append(_fmt_customer(customer))
    return "\n\n".join(parts)
