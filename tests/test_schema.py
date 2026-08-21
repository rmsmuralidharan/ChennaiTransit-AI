from database.schema import DatabaseSchemaManager


def test_create_schemas():

    db_schema_manager = DatabaseSchemaManager()

    db_schema_manager.create_schemas()

    assert db_schema_manager is not None