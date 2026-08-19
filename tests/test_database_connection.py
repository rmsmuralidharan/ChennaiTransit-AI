from database.connection import DatabaseConnection


database_connection = DatabaseConnection()

engine = database_connection.get_engine()

print("Database engine created successfully.")

with engine.connect() as connection:
    result = connection.exec_driver_sql("SELECT 1")
    print("PostgreSQL connection successful.")
    print(f"Test result: {result.scalar()}")