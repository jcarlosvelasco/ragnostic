import os

from langfuse.callback import CallbackHandler
from langfuse.client import Langfuse

langfuse = Langfuse(
    host=os.getenv("LANGFUSE_HOST", "http://langfuse-server:3000"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
)

langfuse_handler = CallbackHandler(
    host=os.getenv("LANGFUSE_HOST", "http://langfuse-server:3000"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
)
