"""
Login, roles and the Settings > Accounts page for the dashboard.

Kept separate from the detection code on purpose: nothing here touches the
cameras, YOLO or the alert handlers. If the dashboard is down, LEDs and
voice alerts still work.

- Every page, stream and API needs a logged-in user (config
  dashboard.auth: false turns this off, e.g. for a quick local test).
- First start with no accounts: /setup creates the first admin.
- admin_required pages (Settings) are refused for viewers.
- POST forms carry a per-session CSRF token.
"""

from __future__ import annotations

import functools
import secrets
from pathlib import Path

from flask import (abort, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)

from storage.user_store import ROLES

# Reachable without logging in.
_PUBLIC = {"login", "setup", "static"}


def _secret_key(config: dict) -> str:
    """Session signing key: from config, else generated once and kept next
    to the database so logins survive a restart."""
    key = (config.get("dashboard") or {}).get("secret_key")
    if key:
        return key
    path = Path(config["storage"]["db_path"]).resolve().parent / ".dashboard_secret"
    try:
        return path.read_text().strip()
    except OSError:
        key = secrets.token_hex(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return key


def admin_required(view):
    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        if g.get("role") != "admin":
            abort(403)
        return view(*args, **kwargs)
    return wrapper


def init_auth(app, config: dict, user_store) -> None:
    enabled = bool((config.get("dashboard") or {}).get("auth", True))
    app.secret_key = _secret_key(config)
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    def _csrf_token() -> str:
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return session["csrf"]

    @app.context_processor
    def _inject():
        return {"current_user": g.get("user"), "current_role": g.get("role"),
                "auth_enabled": enabled, "csrf_token": _csrf_token}

    @app.before_request
    def _require_login():
        if not enabled:
            g.user, g.role = None, "admin"
            return None
        user = session.get("user")
        role = user_store.role_of(user) if user else None  # deleted user -> logged out
        g.user, g.role = (user, role) if role else (None, None)
        if request.method == "POST":
            token = session.get("csrf")
            sent = request.form.get("csrf") or request.headers.get("X-CSRF-Token")
            if not token or sent != token:
                abort(400, "Form expired, please reload the page")
        if request.endpoint in _PUBLIC or g.user:
            return None
        if user_store.count() == 0:
            return redirect(url_for("setup"))
        if request.path.startswith(("/api/", "/video_feed", "/snapshots/", "/settings/zones/")):
            return jsonify({"error": "login required"}), 401
        return redirect(url_for("login", next=request.path))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if not enabled:
            return redirect(url_for("index"))
        if user_store.count() == 0:
            return redirect(url_for("setup"))
        error = None
        if request.method == "POST":
            username = request.form.get("username", "")
            role = user_store.verify(username, request.form.get("password", ""))
            if role:
                session.clear()
                session["user"] = username.strip()
                nxt = request.args.get("next") or ""
                # Only same-site paths, never an outside URL.
                return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//")
                                else url_for("index"))
            error = "Wrong username or password"
        return render_template("login.html", error=error, setup=False)

    @app.route("/setup", methods=["GET", "POST"])
    def setup():
        if not enabled or user_store.count() > 0:
            return redirect(url_for("login"))
        error = None
        if request.method == "POST":
            try:
                if request.form.get("password") != request.form.get("password2"):
                    raise ValueError("Passwords do not match")
                user_store.add(request.form.get("username", ""),
                               request.form.get("password", ""), "admin")
                session.clear()
                session["user"] = request.form["username"].strip()
                return redirect(url_for("index"))
            except ValueError as e:
                error = str(e)
        return render_template("login.html", error=error, setup=True)

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # ── Settings (admin only). Accounts for now; zones/cameras come later.
    @app.route("/settings", methods=["GET"])
    @admin_required
    def settings():
        return render_template("settings.html", users=user_store.list_users(), roles=ROLES)

    @app.route("/settings/users", methods=["POST"])
    @admin_required
    def settings_users():
        action = request.form.get("action")
        name = request.form.get("username", "")
        try:
            if action == "add":
                user_store.add(name, request.form.get("password", ""),
                               request.form.get("role", "viewer"))
                flash(f"Added {name.strip()}", "ok")
            elif action == "delete":
                if name == g.user:
                    raise ValueError("You cannot delete your own account")
                user_store.delete(name)
                flash(f"Deleted {name}", "ok")
            elif action == "password":
                user_store.set_password(name, request.form.get("password", ""))
                flash(f"Password changed for {name}", "ok")
            else:
                abort(400)
        except ValueError as e:
            flash(str(e), "error")
        return redirect(url_for("settings"))
