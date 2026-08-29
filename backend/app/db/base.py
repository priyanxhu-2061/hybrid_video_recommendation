from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import models here so Alembic autogenerate sees them.
from app.models import interaction, user, video  # noqa: E402,F401
