import configparser

# Initialize the parser
config = configparser.ConfigParser()

# Function to read values from config.ini
def read_config(section, key):
    config.read('config.ini')
    return config.get(section, key, fallback=None)  # Returns None if key doesn't exist

# Function to write/update values in config.ini
def write_config(section, key, value):
    config.read('config.ini')
    
    if section not in config:
        config.add_section(section)  # Add section if it doesn't exist
    
    config.set(section, key, value)
    
    with open('config.ini', "w") as configfile:
        config.write(configfile)