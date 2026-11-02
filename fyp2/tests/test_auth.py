"""Dashboard login, roles and account management (dashboard/auth.py)."""

import re

import pytest

from core.shared_state import SharedState
from dashboard.app import create_app
from storage.event_store import EventStore
from storage.user_store import UserStore


@pytest.fixture
def env(tmp_path):
    shots = tmp_path / "shots"
    shots.mkdir()
    config = {"decision_engine": {"debug_overlay": False},
              "alerts": {"screenshot_dir": str(shots)},
              "storage": {"db_path": str(tmp_path / "e.db")},
              "dashboard": {}}
    users = UserStore(config)
    app = create_app(config, {"entrance": SharedState(), "room": None},
                     EventStore(config), user_store=users)
    return app.test_client(), users


def _csrf(client, path):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf" value="([0-9a-f]+)"', html).group(1)


def _login(client, user, pw):
    return client.post("/login", data={"username": user, "password": pw,
                                       "csrf": _csrf(client, "/login")})


class TestUserStore:
    def test_verify_and_roles(self, env):
        _, users = env
        users.add("boss", "secret1", "admin")
        assert users.verify("boss", "secret1") == "admin"
        assert users.verify("boss", "wrong") is None
        assert users.verify("nobody", "secret1") is None

    def test_password_is_hashed(self, env):
        _, users = env
        users.add("boss", "secret1", "admin")
        row = users._conn.execute("SELECT password_hash FROM users").fetchone()[0]
        assert "secret1" not in row

    def test_validation(self, env):
        _, users = env
        with pytest.raises(ValueError):
            users.add("a b", "secret1", "viewer")
        with pytest.raises(ValueError):
            users.add("ok", "123", "viewer")
        with pytest.raises(ValueError):
            users.add("ok", "secret1", "superadmin")
        users.add("ok", "secret1", "viewer")
        with pytest.raises(ValueError):
            users.add("ok", "secret1", "viewer")

    def test_cannot_delete_last_admin(self, env):
        _, users = env
        users.add("boss", "secret1", "admin")
        with pytest.raises(ValueError):
            users.delete("boss")
        users.add("boss2", "secret1", "admin")
        users.delete("boss")
        assert [u["username"] for u in users.list_users()] == ["boss2"]


class TestLoginFlow:
    def test_no_accounts_goes_to_setup(self, env):
        client, _ = env
        r = client.get("/")
        assert r.status_code == 302 and r.headers["Location"].endswith("/setup")

    def test_setup_creates_admin_and_logs_in(self, env):
        client, users = env
        r = client.post("/setup", data={"username": "boss", "password": "secret1",
                                        "password2": "secret1",
                                        "csrf": _csrf(client, "/setup")})
        assert r.status_code == 302
        assert users.role_of("boss") == "admin"
        assert client.get("/").status_code == 200
        # setup is closed once an account exists
        assert client.get("/setup").status_code == 302

    def test_pages_and_api_need_login(self, env):
        client, users = env
        users.add("boss", "secret1", "admin")
        assert client.get("/").headers["Location"].startswith("/login")
        assert client.get("/api/status").status_code == 401
        assert client.get("/video_feed/entrance").status_code == 401

    def test_wrong_password(self, env):
        client, users = env
        users.add("boss", "secret1", "admin")
        r = _login(client, "boss", "nope")
        assert b"Wrong username or password" in r.data

    def test_post_without_csrf_rejected(self, env):
        client, users = env
        users.add("boss", "secret1", "admin")
        r = client.post("/login", data={"username": "boss", "password": "secret1"})
        assert r.status_code == 400

    def test_logout(self, env):
        client, users = env
        users.add("boss", "secret1", "admin")
        _login(client, "boss", "secret1")
        client.post("/logout", data={"csrf": _csrf(client, "/history")})
        assert client.get("/api/status").status_code == 401

    def test_open_redirect_blocked(self, env):
        client, users = env
        users.add("boss", "secret1", "admin")
        r = client.post("/login?next=//evil.com", data={
            "username": "boss", "password": "secret1", "csrf": _csrf(client, "/login")})
        assert r.headers["Location"] == "/"


class TestRoles:
    def test_viewer_cannot_open_settings(self, env):
        client, users = env
        users.add("boss", "secret1", "admin")
        users.add("guard", "secret1", "viewer")
        _login(client, "guard", "secret1")
        assert client.get("/").status_code == 200
        assert client.get("/history").status_code == 200
        assert client.get("/settings").status_code == 403
        assert b'href="/settings"' not in client.get("/").data

    def test_admin_adds_and_deletes_user(self, env):
        client, users = env
        users.add("boss", "secret1", "admin")
        _login(client, "boss", "secret1")
        token = _csrf(client, "/settings")
        client.post("/settings/users", data={"action": "add", "username": "guard",
                                             "password": "secret1", "role": "viewer",
                                             "csrf": token})
        assert users.role_of("guard") == "viewer"
        client.post("/settings/users", data={"action": "delete", "username": "guard",
                                             "csrf": token})
        assert users.role_of("guard") is None

    def test_admin_cannot_delete_self(self, env):
        client, users = env
        users.add("boss", "secret1", "admin")
        users.add("boss2", "secret1", "admin")
        _login(client, "boss", "secret1")
        client.post("/settings/users", data={"action": "delete", "username": "boss",
                                             "csrf": _csrf(client, "/settings")})
        assert users.role_of("boss") == "admin"

    def test_deleted_user_is_logged_out(self, env):
        client, users = env
        users.add("boss", "secret1", "admin")
        users.add("guard", "secret1", "viewer")
        _login(client, "guard", "secret1")
        users.delete("guard")
        assert client.get("/api/status").status_code == 401
