import urllib.parse
from sqlalchemy import create_engine

USER = "root"
PASS = "@Qwertyasdfgh7"
DB = "Aeronova_2_0_Project"   # corrected (no dots allowed in MySQL)
HOST = "127.0.0.1"            # exact same as Workbench

PASS = urllib.parse.quote_plus(PASS)

engine = create_engine(
    f"mysql+mysqlconnector://{USER}:{PASS}@{HOST}:3306/{DB}",
    pool_pre_ping=True
)

print("Connected successfully!")
