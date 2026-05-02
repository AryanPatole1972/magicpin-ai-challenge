"""
Vera — magicpin Merchant AI Assistant
Bot server implementing all 5 required endpoints.
Team: Aryan Patole | Model: claude-3-5-sonnet-20241022 (free tier)
"""

import os
import time
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from composer import VeraComposer

# ─────────────────────────────────────────
# App setup
# ─────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vera-bot")

app = FastAPI(title="Vera — magicpin Merchant AI Bot", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html", context={"request": request})

from fastapi.responses import RedirectResponse
from judge_simulator import app as judge_app
app.mount("/judge", judge_app)

@app.get("/judge", include_in_schema=False)
async def redirect_to_judge():
    return RedirectResponse(url="/judge/")

START_TIME = time.time()

# ─────────────────────────────────────────
# In-memory state stores
# ─────────────────────────────────────────
# (scope, context_id) -> {version: int, payload: dict}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id -> list of turns  {from, body, ts, action}
conversations: dict[str, list[dict]] = {}

# suppression_key -> True  (sent already in this session)
suppression_set: set[str] = set()

# composer singleton
composer = VeraComposer()


# ─────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────
class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ─────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────
VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _get_ctx(scope: str, cid: str) -> Optional[dict]:
    entry = contexts.get((scope, cid))
    return entry["payload"] if entry else None

def _context_counts() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts

def _is_auto_reply(message: str) -> bool:
    """Detect WhatsApp Business canned auto-replies."""
    lower = message.lower()
    auto_signals = [
        "thank you for contacting",
        "we will get back to you",
        "automated message",
        "aapki jaankari ke liye bahut-bahut shukriya",
        "main aapki yeh sabhi baatein",
        "hamari team tak pahuncha",
        "automated assistant",
        "this is an automated",
        "i am an automated",
        "main ek automated",
    ]
    return any(sig in lower for sig in auto_signals)

def _is_negative_intent(message: str) -> bool:
    """Detect clear stop / not-interested signals."""
    lower = message.lower()
    negatives = [
        "not interested", "nahi chahiye", "band karo", "stop", "unsubscribe",
        "do not contact", "mat bhejo", "hatao", "remove me",
        "no thanks", "no thank you",
    ]
    return any(n in lower for n in negatives)

def _is_action_intent(message: str) -> bool:
    """Detect explicit 'yes do it' intent."""
    lower = message.lower()
    positives = [
        "yes", "haan", "go ahead", "kar do", "chalega", "ok", "proceed",
        "let's do it", "karo", "please do", "send me", "bhejo",
        "i want to join", "judrna hai", "join karna",
    ]
    return any(p in lower for p in positives)


# ─────────────────────────────────────────
# Endpoint 1: POST /v1/context
# ─────────────────────────────────────────
@app.post("/v1/context")
async def push_context(body: ContextBody):
    if body.scope not in VALID_SCOPES:
        return {"accepted": False, "reason": "invalid_scope",
                "details": f"scope must be one of {VALID_SCOPES}"}

    key = (body.scope, body.context_id)
    existing = contexts.get(key)

    if existing and existing["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version",
                "current_version": existing["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload}
    logger.info("Stored %s/%s v%d", body.scope, body.context_id, body.version)

    ack_id = f"ack_{body.context_id}_v{body.version}_{uuid.uuid4().hex[:6]}"
    return {"accepted": True, "ack_id": ack_id, "stored_at": _now_iso()}


# ─────────────────────────────────────────
# Endpoint 2: POST /v1/tick
# ─────────────────────────────────────────
@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        trg = _get_ctx("trigger", trg_id)
        if not trg:
            continue

        # Suppression check
        suppression_key = trg.get("suppression_key", "")
        if suppression_key in suppression_set:
            continue

        # Expiry check
        expires_at = trg.get("expires_at")
        if expires_at:
            try:
                exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) > exp:
                    continue
            except Exception:
                pass

        merchant_id = trg.get("merchant_id")
        if not merchant_id:
            continue

        merchant = _get_ctx("merchant", merchant_id)
        if not merchant:
            continue

        category_slug = merchant.get("category_slug")
        category = _get_ctx("category", category_slug) if category_slug else None

        customer_id = trg.get("customer_id")
        customer = _get_ctx("customer", customer_id) if customer_id else None

        # Compose message
        try:
            composed = composer.compose(
                category=category or {},
                merchant=merchant,
                trigger=trg,
                customer=customer,
            )
        except Exception as e:
            logger.error("Compose error for %s/%s: %s", merchant_id, trg_id, e)
            continue

        if not composed or not composed.get("body"):
            continue

        conv_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:8]}"

        # Mark suppression
        if suppression_key:
            suppression_set.add(suppression_key)

        # Store in conversations
        conversations[conv_id] = [{
            "from": "vera",
            "body": composed["body"],
            "ts": body.now,
        }]

        action_item = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind', 'generic')}_v1",
            "template_params": [
                merchant.get("identity", {}).get("name", ""),
                trg.get("kind", ""),
                composed["body"][:50],
            ],
            "body": composed["body"],
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": composed.get("rationale", ""),
        }
        actions.append(action_item)

        # Cap at 20 actions per tick (per spec)
        if len(actions) >= 20:
            break

    logger.info("Tick at %s → %d actions", body.now, len(actions))
    return {"actions": actions}


# ─────────────────────────────────────────
# Endpoint 3: POST /v1/reply
# ─────────────────────────────────────────
@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    conv_history = conversations.setdefault(conv_id, [])

    # Append incoming
    conv_history.append({
        "from": body.from_role,
        "body": body.message,
        "ts": body.received_at,
        "turn_number": body.turn_number,
    })

    # ── Auto-reply detection ──
    if _is_auto_reply(body.message):
        auto_count = sum(
            1 for t in conv_history
            if t["from"] in ("merchant", "customer") and _is_auto_reply(t.get("body", ""))
        )
        if auto_count >= 2:
            # Graceful exit after 2 auto-replies
            conv_history.append({"from": "vera", "body": "__end__", "ts": _now_iso()})
            return {
                "action": "end",
                "rationale": "Detected repeated WhatsApp Business auto-reply; gracefully exiting to avoid turn burn.",
            }
        else:
            # Try once more with a direct question
            merchant_id = body.merchant_id
            merchant = _get_ctx("merchant", merchant_id) if merchant_id else None
            name = (merchant or {}).get("identity", {}).get("name", "there")
            owner = (merchant or {}).get("identity", {}).get("owner_first_name", "")
            salutation = f"Dr. {owner}" if (merchant or {}).get("category_slug") == "dentists" else (owner or name)
            probe = (
                f"Samajh gayi. Team tak pahunchane se pehle, {salutation} — "
                f"kya aap khud dekhna chahengi ki exact kya kaam baaki hai? 2 minute ka kaam hai. Chalega?"
            )
            conv_history.append({"from": "vera", "body": probe, "ts": _now_iso()})
            return {
                "action": "send",
                "body": probe,
                "cta": "binary_yes_no",
                "rationale": "First auto-reply detected; probing once for real engagement before exiting.",
            }

    # ── Negative intent → graceful exit ──
    if _is_negative_intent(body.message):
        conv_history.append({"from": "vera", "body": "__end__", "ts": _now_iso()})
        return {
            "action": "end",
            "rationale": "Merchant signaled not-interested; exiting gracefully.",
        }

    # ── Positive / action intent → execute or continue ──
    merchant_id = body.merchant_id
    customer_id = body.customer_id
    merchant = _get_ctx("merchant", merchant_id) if merchant_id else {}
    customer = _get_ctx("customer", customer_id) if customer_id else None

    # Find the last trigger used in this conversation to maintain context
    # Look for any trigger matching this merchant
    best_trigger = None
    for (scope, cid), entry in contexts.items():
        if scope == "trigger":
            trg_payload = entry["payload"]
            if trg_payload.get("merchant_id") == merchant_id:
                best_trigger = trg_payload
                break

    category_slug = (merchant or {}).get("category_slug")
    category = _get_ctx("category", category_slug) if category_slug else {}

    try:
        reply_composed = composer.compose_reply(
            category=category or {},
            merchant=merchant or {},
            trigger=best_trigger or {},
            customer=customer,
            conversation_history=conv_history,
            merchant_message=body.message,
        )
    except Exception as e:
        logger.error("Reply compose error: %s", e)
        # Fallback
        reply_composed = {
            "action": "send",
            "body": "Got it — let me pull that up for you right away. Give me a moment.",
            "cta": "open_ended",
            "rationale": "Fallback on compose error; keeping conversation alive.",
        }

    if reply_composed.get("action") in ("wait", "end"):
        return reply_composed

    body_text = reply_composed.get("body", "")
    if not body_text:
        return {
            "action": "end",
            "rationale": "Nothing useful to add; ending conversation.",
        }

    # Anti-repetition: check last vera message
    vera_msgs = [t["body"] for t in conv_history if t["from"] == "vera"]
    if vera_msgs and body_text.strip() == vera_msgs[-1].strip():
        body_text += " — koi aur sawaal ho toh batayein."

    conv_history.append({"from": "vera", "body": body_text, "ts": _now_iso()})

    return {
        "action": "send",
        "body": body_text,
        "cta": reply_composed.get("cta", "open_ended"),
        "rationale": reply_composed.get("rationale", "Continued engagement based on merchant reply."),
    }


# ─────────────────────────────────────────
# Endpoint 4: GET /v1/healthz
# ─────────────────────────────────────────
@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": _context_counts(),
    }


# ─────────────────────────────────────────
# Endpoint 5: GET /v1/metadata
# ─────────────────────────────────────────
@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Aryan Patole",
        "team_members": ["Aryan Patole"],
        "model": "claude-3-5-sonnet-20241022",
        "approach": (
            "4-context composer with trigger-kind routing: each trigger kind maps to a "
            "specialized prompt variant. Auto-reply detection via lexical signals. "
            "Intent routing (positive/negative) in reply handler. "
            "Suppression via in-memory keyset. Language matching via merchant identity.languages."
        ),
        "contact_email": "aryanpatole@example.com",
        "version": "1.0.0",
        "submitted_at": "2026-04-29T12:00:00Z",
    }


# ─────────────────────────────────────────
# Optional teardown
# ─────────────────────────────────────────
@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    suppression_set.clear()
    logger.info("State wiped on teardown.")
    return {"status": "wiped"}


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7000))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False)
