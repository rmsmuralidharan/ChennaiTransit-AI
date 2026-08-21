from ChennaiTransit_AI.utils.configuration import ConfigurationManager


def test_database_config():

    configuration_manager = ConfigurationManager()

    # Test database configuration
    database_config = configuration_manager.get_database_config()

    assert database_config is not None
    assert "host" in database_config
    assert "port" in database_config
    assert "name" in database_config
    assert "username" in database_config
    assert "password" in database_config

    # Test data configuration
    data_config = configuration_manager.get_data_config()

    assert data_config is not None
    assert "raw" in data_config
    assert "transformed" in data_config
    assert "final" in data_config

    # Test ETL configuration
    etl_config = configuration_manager.get_etl_config()

    assert etl_config is not None