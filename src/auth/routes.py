from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from src.auth.models import UserModel, HistoryModel
import re

auth_bp = Blueprint("auth", __name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def get_db():
    """Lấy db instance từ app context (được gắn vào app.extensions)"""
    from flask import current_app
    return current_app.extensions["mongo_db"]


def current_user():
    """Trả về dict user nếu đã đăng nhập, ngược lại None"""
    user_id = session.get("user_id")
    if not user_id:
        return None
    db = get_db()
    return UserModel(db).find_by_id(user_id)


def login_required(f):
    """Decorator bảo vệ route cần đăng nhập"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            flash("Bạn cần đăng nhập để tiếp tục.", "warning")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


# ── Register ─────────────────────────────────────────────────────────────────

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home_page"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        # Validate
        errors = []
        if not username or len(username) < 3:
            errors.append("Tên người dùng phải có ít nhất 3 ký tự.")
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            errors.append("Email không hợp lệ.")
        if len(password) < 6:
            errors.append("Mật khẩu phải có ít nhất 6 ký tự.")
        if password != confirm:
            errors.append("Mật khẩu xác nhận không khớp.")

        db = get_db()
        user_model = UserModel(db)

        if not errors:
            if user_model.email_exists(email):
                errors.append("Email này đã được sử dụng.")
            if user_model.username_exists(username):
                errors.append("Tên người dùng đã tồn tại.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("register.html", username=username, email=email)

        user = user_model.create(username, email, password)
        session["user_id"]   = str(user["_id"])
        session["username"]  = user["username"]
        flash("Đăng ký thành công! Chào mừng bạn.", "success")
        return redirect(url_for("home_page"))

    return render_template("register.html")


# ── Login ─────────────────────────────────────────────────────────────────────

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home_page"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        db = get_db()
        user_model = UserModel(db)
        user = user_model.find_by_email(email)

        if not user or not user_model.verify_password(user, password):
            flash("Email hoặc mật khẩu không đúng.", "error")
            return render_template("login.html", email=email)

        session["user_id"]  = str(user["_id"])
        session["username"] = user["username"]
        flash(f"Chào mừng trở lại, {user['username']}!", "success")
        return redirect(url_for("home_page"))

    return render_template("login.html")


# ── Logout ────────────────────────────────────────────────────────────────────

@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("Bạn đã đăng xuất.", "info")
    return redirect(url_for("auth.login"))


# ── History ───────────────────────────────────────────────────────────────────

@auth_bp.route("/history")
@login_required
def history():
    db = get_db()
    history_model = HistoryModel(db)
    user_id = session["user_id"]

    records = history_model.get_by_user(user_id, limit=50)
    stats   = history_model.stats(user_id)

    # Format ngày giờ cho template
    for r in records:
        r["date_str"] = r["created_at"].strftime("%d/%m/%Y %H:%M")

    return render_template(
        "history.html",
        records=records,
        stats=stats,
        username=session.get("username"),
    )


@auth_bp.route("/history/delete/<record_id>", methods=["POST"])
@login_required
def delete_history(record_id):
    db = get_db()
    success = HistoryModel(db).delete(record_id, session["user_id"])
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"success": success})
    flash("Đã xoá bản ghi." if success else "Không tìm thấy bản ghi.", "info")
    return redirect(url_for("auth.history"))