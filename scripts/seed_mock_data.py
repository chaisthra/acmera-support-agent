"""
Seed orders and customers tables from mock_data JSON files.

Safe to re-run — uses INSERT ... ON CONFLICT DO UPDATE so existing rows
are updated rather than duplicated.

Run: python scripts/seed_mock_data.py
"""
import os
import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()

MOCK_DIR = os.path.join(os.path.dirname(__file__), "..", "mock_data")


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        user=os.getenv("PG_USER", "workshop"),
        password=os.getenv("PG_PASSWORD"),
        dbname=os.getenv("PG_DATABASE", "acmera_kb"),
    )


def seed():
    customers = json.load(open(os.path.join(MOCK_DIR, "customers.json")))
    orders    = json.load(open(os.path.join(MOCK_DIR, "orders.json")))

    conn = get_connection()
    cur  = conn.cursor()

    # customers first (orders FK references customers)
    for c in customers:
        cur.execute("""
            INSERT INTO customers (id, name, email, tier, since, annual_spend)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                name         = EXCLUDED.name,
                email        = EXCLUDED.email,
                tier         = EXCLUDED.tier,
                since        = EXCLUDED.since,
                annual_spend = EXCLUDED.annual_spend;
        """, (c["id"], c["name"], c.get("email"), c["tier"],
              c.get("since"), c.get("annual_spend", 0)))

    for o in orders:
        cur.execute("""
            INSERT INTO orders
                (order_id, customer_id, product, price, date, delivery_date, status, promotional)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (order_id) DO UPDATE SET
                customer_id   = EXCLUDED.customer_id,
                product       = EXCLUDED.product,
                price         = EXCLUDED.price,
                date          = EXCLUDED.date,
                delivery_date = EXCLUDED.delivery_date,
                status        = EXCLUDED.status,
                promotional   = EXCLUDED.promotional;
        """, (o["order_id"], o["customer_id"], o.get("product"),
              o.get("price"), o.get("date"), o.get("delivery_date"),
              o.get("status"), o.get("promotional", False)))

    conn.commit()
    cur.close()
    conn.close()
    print(f"Seeded {len(customers)} customers, {len(orders)} orders.")


if __name__ == "__main__":
    seed()
