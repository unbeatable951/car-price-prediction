"""
app/extensions.py
===================
Shared Flask extension instances, defined in their own module (rather
than inside app.py) so other modules — routes.py in particular — can
import and use them (e.g. `@limiter.limit(...)`) without creating a
circular import with app.py, which itself imports routes.py.

This is the standard pattern Flask extensions like Flask-SQLAlchemy
and Flask-Limiter are documented to use: create the extension object
here, unbound; call `extension.init_app(app)` once inside app.py's
create_app().
"""

from __future__ import annotations

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# No default_limits here — routes opt in individually (see
# app/routes.py's @limiter.limit(...) on POST /predict) rather than
# every endpoint inheriting a blanket limit.
limiter = Limiter(key_func=get_remote_address, default_limits=[])