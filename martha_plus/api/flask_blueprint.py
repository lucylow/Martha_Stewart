"""Read-only operations API and authenticated kill-switch controls.

There is deliberately no HTTP order-submission endpoint. Strategies can propose
orders only through the risk-gated service layer.
"""
from __future__ import annotations
import hmac
from flask import Blueprint, jsonify, request
from ..services.engine import TradingResearchEngine


def create_blueprint(engine: TradingResearchEngine, api_token: str,
                     name: str = "martha_plus", url_prefix: str = "/martha-plus") -> Blueprint:
    if not api_token or len(api_token) < 24:
        raise ValueError("API token must be configured and at least 24 characters")
    blueprint = Blueprint(name, __name__, url_prefix=url_prefix)

    def authorized() -> bool:
        header = request.headers.get("Authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix):
            return False
        return hmac.compare_digest(header[len(prefix):], api_token)

    @blueprint.get("/health")
    def health():
        # Do not disclose account state, local paths, strategy parameters, or tokens.
        return jsonify({"status": "ok", "component": "martha_plus", "paper_only": True}), 200

    @blueprint.get("/status")
    def status():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        return jsonify(engine.status()), 200

    @blueprint.get("/orders")
    def orders():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        rows = []
        for order in engine.order_manager.all_orders[-100:]:
            rows.append({"order_id": order.order_id,
                         "client_order_id": order.request.client_order_id,
                         "symbol": order.request.symbol, "side": order.request.side.value,
                         "quantity": order.request.quantity, "status": order.status.value,
                         "filled_quantity": order.filled_quantity, "reason": order.reason,
                         "updated_at": order.updated_at.isoformat()})
        return jsonify({"orders": rows}), 200

    @blueprint.post("/kill-switch")
    def activate_kill_switch():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        payload = request.get_json(silent=True) or {}
        reason = str(payload.get("reason", "API emergency halt"))[:500]
        engine.kill_switch.activate(reason)
        return jsonify({"active": True, "message": "new orders halted"}), 200

    @blueprint.delete("/kill-switch")
    def deactivate_kill_switch():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        engine.kill_switch.deactivate()
        return jsonify({"active": False, "message": "halt removed; risk checks still apply"}), 200

    return blueprint
