import os
import dotenv

dotenv.load_dotenv()

# Used in DataFilePlayGround_2.py
# Path to raw data
RAW_DATA_PATH = str(os.getenv("RAW_DATA_PATH"))
# Path to processed numpy matrices
PROCESSED_NUMPY = str(os.getenv("PROCESSED_NUMPY"))