from database.connection import DatabaseConnection


def test_database_connection():

    database_connection = DatabaseConnection()

    engine = database_connection.get_engine()

    assert engine is not None

    with engine.connect() as connection:

        result = connection.exec_driver_sql("SELECT 1")

        assert result.scalar() == 1