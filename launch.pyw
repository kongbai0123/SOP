"""Windows source entry; the installed entry uses the same safe dispatcher."""
from sop_app.launcher import start


if __name__ == "__main__":
    raise SystemExit(start())
