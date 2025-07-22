import os
from supabase import create_client
from dotenv import load_dotenv
import mimetypes

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "images")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_image_to_supabase(file_path: str, file_name: str = None) -> str:
    if file_name is None:
        file_name = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        file_data = f.read()


    content_type, _ = mimetypes.guess_type(file_name)
    if not content_type:
        content_type = "application/octet-stream"
    supabase.storage.from_(SUPABASE_BUCKET).upload(file_name, file_data, {"content-type": content_type, "x-upsert": "true"})


    public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(file_name)
    return public_url
