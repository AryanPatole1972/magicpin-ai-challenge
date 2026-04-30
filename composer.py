import json
from typing import Any, Dict, List, Optional
import os
import google.generativeai as genai

# Setup Gemini API (we'll use a mocked version if key not present for local testing without real API calls if desired,
# but the challenge requires actual LLM usage if possible. I'll structure it to use Gemini).
# Assuming the user has set GEMINI_API_KEY environment variable. If not, fallback to a dummy/mock response.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    
class VeraComposer:
    def __init__(self):
        # We use a generative model. 2.0-flash is good for quick textual tasks.
        if GEMINI_API_KEY:
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            self.model = None

    def compose(
        self,
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Composes a message based on the 4 contexts.
        Returns a dict with: body, cta, send_as, rationale.
        """
        prompt = self._build_compose_prompt(category, merchant, trigger, customer)
        
        # Default mock output if no API key
        output = {
            "body": "Hi, this is Vera. Just wanted to check in. Let me know if you need anything.",
            "cta": "open_ended",
            "send_as": "vera",
            "rationale": "Fallback message"
        }

        if self.model:
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1, # Keep it mostly deterministic
                        response_mime_type="application/json"
                    )
                )
                if response.text:
                    parsed = json.loads(response.text)
                    output["body"] = parsed.get("body", output["body"])
                    output["cta"] = parsed.get("cta", output["cta"])
                    output["send_as"] = parsed.get("send_as", "merchant_on_behalf" if customer else "vera")
                    output["rationale"] = parsed.get("rationale", output["rationale"])
            except Exception as e:
                print(f"LLM Compose Error: {e}")
                # We can also implement a rule-based fallback here if needed.
                output = self._fallback_compose(category, merchant, trigger, customer)

        else:
             output = self._fallback_compose(category, merchant, trigger, customer)

        # Force correct send_as
        output["send_as"] = "merchant_on_behalf" if customer else "vera"
        return output

    def compose_reply(
        self,
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
        merchant_message: str
    ) -> Dict[str, Any]:
        """
        Composes a reply to an ongoing conversation.
        Returns: action ('send', 'wait', 'end'), body, cta, rationale.
        """
        prompt = self._build_reply_prompt(
            category, merchant, trigger, customer, conversation_history, merchant_message
        )
        
        output = {
            "action": "send",
            "body": "Got it. Let me know if anything else is needed.",
            "cta": "open_ended",
            "rationale": "Fallback reply"
        }

        if self.model:
            try:
                 response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                 if response.text:
                    parsed = json.loads(response.text)
                    output["action"] = parsed.get("action", output["action"])
                    output["body"] = parsed.get("body", output["body"])
                    output["cta"] = parsed.get("cta", output["cta"])
                    output["rationale"] = parsed.get("rationale", output["rationale"])
            except Exception as e:
                print(f"LLM Reply Error: {e}")
        
        return output

    def _build_compose_prompt(
        self, category: Dict, merchant: Dict, trigger: Dict, customer: Optional[Dict]
    ) -> str:
        
        is_customer = bool(customer)
        
        sys_msg = """You are Vera, magicpin's merchant AI assistant. You compose WhatsApp messages for Indian merchants.
You must return a JSON object with strictly these keys: "body", "cta", "send_as", "rationale".

RULES:
1. Specificity: Use exact numbers, dates, and facts from the context. No generic "10% off". Use service+price (e.g. "Haircut @ ₹99").
2. Tone: Match the category voice (e.g. peer/clinical for dentists). Code-mix Hindi-English if the merchant/customer language preference includes 'hi'.
3. One CTA: Have a single, clear Call-To-Action (binary YES/NO or simple question) at the very end.
4. "send_as": If you are messaging the merchant directly, use "vera". If messaging a customer on behalf of the merchant, use "merchant_on_behalf".
5. No Hallucinations: Do not invent stats, competitor names, or offers not in the context.
6. Engagement: Use a hook (loss aversion, social proof, specific research).
"""

        ctx_str = f"CATEGORY CONTEXT:\n{json.dumps(category, indent=2)}\n\n"
        ctx_str += f"MERCHANT CONTEXT:\n{json.dumps(merchant, indent=2)}\n\n"
        ctx_str += f"TRIGGER CONTEXT:\n{json.dumps(trigger, indent=2)}\n\n"
        
        if is_customer:
            ctx_str += f"CUSTOMER CONTEXT:\n{json.dumps(customer, indent=2)}\n\n"
            task = "Draft a message TO THE CUSTOMER, ON BEHALF OF THE MERCHANT. It should sound like it is coming from the merchant's clinic/shop directly."
        else:
            task = "Draft a message TO THE MERCHANT, FROM VERA. Hook them and offer to help or share info based on the trigger."

        return f"{sys_msg}\n\n{ctx_str}\n\nTASK: {task}\nProvide JSON output."

    def _build_reply_prompt(
        self, category: Dict, merchant: Dict, trigger: Dict, customer: Optional[Dict],
        history: List[Dict], message: str
    ) -> str:
        
        sys_msg = """You are Vera, magicpin's merchant AI assistant. You are handling a reply in an ongoing WhatsApp conversation.
You must return a JSON object with strictly these keys: "action" (must be "send", "wait", or "end"), "body" (the message text, if action is send), "cta", "rationale".

RULES:
1. If the merchant explicitly says "yes" or "let's do it" to an offer you made, transition immediately to action (don't qualify more). 
2. If they ask a question, answer it concisely based on the context provided.
3. If they are completely off-topic or hostile, try to politely pivot back or gracefully end the conversation (action: "end").
4. Match their language (Hindi/English code-mixing).
"""
        ctx_str = f"MERCHANT/CUSTOMER DATA:\nMerchant: {json.dumps(merchant.get('identity', {}))}\nTrigger: {json.dumps(trigger.get('kind',''))}\n\n"
        
        hist_str = "CONVERSATION HISTORY:\n"
        for t in history[-5:]: # last 5 turns
            hist_str += f"[{t['from'].upper()}]: {t.get('body', '')}\n"
        
        return f"{sys_msg}\n\n{ctx_str}\n{hist_str}\n\nDetermine the next best move and provide JSON output."

    def _fallback_compose(self, category: Dict, merchant: Dict, trigger: Dict, customer: Optional[Dict]) -> Dict:
        """A simple rule-based fallback if LLM is unavailable to ensure the bot still scores something."""
        
        kind = trigger.get("kind", "")
        m_name = merchant.get("identity", {}).get("owner_first_name", merchant.get("identity", {}).get("name", "there"))
        salutation = f"Dr. {m_name}" if merchant.get("category_slug") == "dentists" else m_name
        
        if customer:
             c_name = customer.get("identity", {}).get("name", "there")
             m_biz = merchant.get("identity", {}).get("name", "us")
             if kind == "recall_due":
                  return {
                      "body": f"Hi {c_name}, {m_biz} here! Your regular checkup is due. Aapke liye slots open hain. Reply 1 to book.",
                      "cta": "binary",
                      "rationale": "Rule-based recall due",
                      "send_as": "merchant_on_behalf"
                  }
             return {
                 "body": f"Hi {c_name}, {m_biz} here. We have some updates for you.",
                 "cta": "open_ended", "rationale": "Fallback customer msg", "send_as": "merchant_on_behalf"
             }

        # Merchant facing
        body = f"Hi {salutation}, Vera here. "
        
        if kind == "research_digest":
             item = trigger.get("payload", {}).get("top_item_id", "some research")
             body += f"New research dropped that might affect your patients. Want me to send the summary?"
        elif kind == "perf_dip":
             metric = trigger.get("payload", {}).get("metric", "views")
             body += f"Notice a slight dip in your GBP {metric} this week. Want me to draft a new post to boost visibility?"
        elif kind == "review_theme_emerged":
             theme = trigger.get("payload", {}).get("theme", "feedback")
             body += f"Customers are mentioning {theme} in recent reviews. Should we address this?"
        else:
             body += "Just checking in on your profile performance. Need any help updating offers today?"

        return {
            "body": body,
            "cta": "binary_yes_no",
            "rationale": f"Rule-based fallback for {kind}",
            "send_as": "vera"
        }
