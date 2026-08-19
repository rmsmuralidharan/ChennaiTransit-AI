from ChennaiTransit_AI.utils.configuration import ConfigurationManager


configuration_manager = ConfigurationManager()


# Test database configuration
database_config = configuration_manager.get_database_config()

print("DATABASE CONFIGURATION")
print(f"Host: {database_config['host']}")
print(f"Port: {database_config['port']}")
print(f"Database: {database_config['name']}")
print(f"Username: {database_config['username']}")


# Test data configuration
data_config = configuration_manager.get_data_config()

print("\nDATA CONFIGURATION")
print(f"Raw: {data_config['raw']}")
print(f"Transformed: {data_config['transformed']}")
print(f"Final: {data_config['final']}")


# Test ETL configuration
etl_config = configuration_manager.get_etl_config()

print("\nETL CONFIGURATION")
print(etl_config)