import os
import dotenv

dotenv.load_dotenv()
RAW_DATA_PATH = str(os.getenv("RAW_DATA_PATH"))
PROCESSED_NUMPY = str(os.getenv("PROCESSED_NUMPY"))