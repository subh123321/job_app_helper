"""Compatibility entrypoint for the v2 job application agent.

Run this file if you prefer the shorter historical name:

    python job_application_v2.py
"""

import runpy


if __name__ == "__main__":
    runpy.run_module("job_application_agent_v2", run_name="__main__")
