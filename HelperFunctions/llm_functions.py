# Code to be used when LLM Engineering

import os
from pathlib import Path
from dotenv import load_dotenv

def load_secrets(COLAB=False):
  """Load the enviornment variables to be used in notebooks/py file
     when working in local/COLAB
  """
  if COLAB is True:
        from google.colab import userdata

        for key in [
            "HF_TOKEN",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
        ]:
            value = userdata.get(key)
            if value:
                os.environ[key] = value
    else:
        load_dotenv(Path.cwd().parent / ".env")
