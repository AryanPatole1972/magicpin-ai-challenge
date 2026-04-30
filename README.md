# Vera — magicpin Merchant AI Assistant

## Overview
This is a submission for the magicpin Vera AI Challenge. The bot is designed to act as an intelligent merchant assistant, handling both merchant-facing nudges and customer-facing (on-behalf-of-merchant) communication.

## Approach
The bot is built on a **4-Context Routing Framework** using FastAPI and Google's Gemini 2.0 Flash Exp model.

1. **Stateful In-Memory Context**: The `/v1/context` endpoint stores the Category, Merchant, Customer, and Trigger contexts in-memory, updating them if a higher version is received.
2. **Intelligent Composition**: At each `/v1/tick`, active triggers are evaluated. The composer (`composer.py`) builds a comprehensive prompt merging the 4 contexts. It instructs the LLM to format its response strictly as JSON, enforcing tone, specificity (anchoring on data), and a single binary CTA.
3. **Graceful Reply Handling**: The `/v1/reply` endpoint manages the conversation history. It includes specific lexical checks for WhatsApp Business auto-replies (exiting gracefully after 2 to avoid turn burn) and negative intents. If an action intent is detected, it handles the context appropriately.

## Tradeoffs Made
- **In-Memory State**: For the sake of the evaluation timeline, state is kept in-memory. In production, this would be backed by Redis (for fast, ephemeral session state) and a permanent DB.
- **Synchronous LLM Calls**: The bot waits for the LLM during the `/v1/tick` and `/v1/reply` loops. Given the 30-second timeout, this is manageable for a small number of actions per tick but would require async task queues (e.g. Celery) for large-scale production.
- **Rule-Based Fallbacks**: If the LLM call fails or times out, the bot falls back to structured, safe rule-based messages to ensure continuous engagement without hallucination.

## What additional context would have helped
- **Actual LLM performance data**: Knowing exactly how the judge's simulated merchants behave (e.g. do they always reply with short texts or sometimes long paragraphs?) would help fine-tune the reply LLM prompt.
- **Specific Rate Limits for third-party LLMs**: Since we are under a 30s timeout, knowing the P99 latency of the judge harness would help decide when to short-circuit and just return `wait`.

## Running the Bot
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"
python bot.py
```
