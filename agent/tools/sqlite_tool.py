import sqlite3
import logging

logging.basicConfig(
    filename="logs/agentlog.log",
    format='%(asctime)s %(levelname)s %(message)s',
    filemode='a',
    level=logging.INFO)

logger = logging.getLogger()

class SQLiteClient:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to {self.db_name}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")

    def disconnect(self):
        if self.conn:
            self.conn.close()
            logger.info(f"Disconnected from {self.db_name}")

    def fetch_all(self, table_name):
        try:
            self.cursor.execute(f"SELECT * FROM {table_name}")
            rows = self.cursor
            return rows
        except sqlite3.Error as e:
            logger.error(f"Error fetching data: {e}")
            return []

    def extract_schema(self, table_name = None) -> str:
        if isinstance(table_name, str):
            tables = [table_name]
        elif isinstance(table_name, list):
            tables = table_name
        else:
            tables = self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
            
        schema_text = ""

        for t in tables:
            cols = self.cursor.execute(f"PRAGMA table_info('{t}');").fetchall()
            schema_text += f"\nTable Name: **{t}**\n"
            for col in cols:
                cid, name, ctype, notnull, dflt, pk = col
                schema_text += f"  - {name} {ctype}\n"
        return schema_text 

    def execute_query(self, query, params=(), return_with_columns_names=False):
        """
        Executes a custom SQL query.
        """
        try:
            self.cursor.execute(query, params, )
            self.conn.commit()
            rows = self.cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error executing query: {e}")
            return [], [], e
        
        column_names = [description[0] for description in self.cursor.description]
        if return_with_columns_names:
            results_with_names = [
                {col_name: row[i] for i, col_name in enumerate(column_names)}
                for row in rows
            ]
            return results_with_names, column_names
        return rows, column_names, None
    
if __name__ == "__main__":
    db = SQLiteClient(r"data\database\northwind.db")
    try:
        db.connect()
        table_names = "Categories"
        table_names = ["Orders", "Order Details", "Products"]
        print( db.extract_schema(table_names) )
        print( db.execute_query("SELECT CategoryID, CategoryName, Description FROM Categories LIMIT 2", return_with_columns_names=True) )
    except Exception as e:
        print(e)
    finally:
        db.disconnect()