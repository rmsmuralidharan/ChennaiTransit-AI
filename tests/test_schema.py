from database.schema import DatabaseSchemaManager

db_schema_manager = DatabaseSchemaManager()

db_schema_manager.create_schemas()

print('Schema created successfully')