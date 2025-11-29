CREATE VIEW IF NOT EXISTS demo_orders AS SELECT * FROM Orders;
CREATE VIEW IF NOT EXISTS demo_order_details AS SELECT * FROM "Order Details";
CREATE VIEW IF NOT EXISTS demo_products AS SELECT * FROM Products;
CREATE VIEW IF NOT EXISTS demo_customers AS SELECT * FROM Customers;


PRAGMA table_info("Order Details");

select * from demo_order_items;