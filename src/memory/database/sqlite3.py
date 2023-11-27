import sqlite3

#POSSIBLE IMPROVEMENTS
# It would be a good practice to use with statements when committing or rolling back transactions



class Sqlite3Connection:
    def __init__(self, database_path: str, table_name:str):

        try:
            self.database_path = database_path
            self.conn = sqlite3.connect(self.database_path)
            self.cursor = self.conn.cursor()
            self.init_sqlite3(table_name)

        except Exception as e:
            print("Error connecting to database:", e)


    def init_sqlite3(self, table_name: str) -> None:
        try:
            with self.conn:
                self.cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id INTEGER PRIMARY KEY,
                        record TEXT, 
                        summary TEXT)
                """)
            print("Database initialized w/ id, record, summary columns")
        except sqlite3.Error as e:
            print(f"An error occurred: <<<< {e}")


    def insert_record(self, id: int, record: str, table_name: str, summary:str = "None" ) -> None:
        try:
            with self.conn:
                #print(id,  summary)
                self.cursor.execute(f"""
                    INSERT INTO {table_name} (id, record, summary) 
                    VALUES(?,?,?)
                """, (id, record, str(summary)))
            print("Record inserted successfully")
        except sqlite3.Error as e:
            print(f"An error occurred: <<< {e}")


    def get_latest_record(self, table_name: str, limit:int = 1) -> str:
        """
        Retrieves the latest 'limit' number of summaries from a specified table.
        
        Parameters:
        - table_name (str): Name of the table from which to retrieve the records.
        - limit (int): The maximum number of record summaries to retrieve.
        
        Returns:
        - str: A string concatenation of the retrieved chat history summaries.
            Returns None if no records are found or if an error occurs."""

        try:
            self.cursor.execute(f"""
                SELECT record, summary FROM {table_name}
                ORDER BY id DESC
                LIMIT ?
                """, (limit,))
            results = self.cursor.fetchall()  # Fetch the latest {limit} record/s

            if results:
                #record, summary = results
                summary_concat = "last record summaries :\n"
                for i , row in reversed(list(enumerate(results))):
                    summary_concat += f"-{(i+1)*'*' }:  {row[1]}\n"
                return str(summary_concat)

            else:
                print("No records found")
                return None

        except sqlite3.Error as e:
            print(f"An error occurred: 2223{e}")
            return None

    def close_connection(self):
        self.cursor.close()
        self.conn.close()

