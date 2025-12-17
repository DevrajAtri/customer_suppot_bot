AI E-Commerce Support Agent
A graph-based conversational agent built with Python, LangChain, and Google Gemini. It uses a state-machine architecture to handle customer support tasks and implements a Human-in-the-Loop (HITL) system for sensitive actions like refunds.

Features
Graph Architecture: Implements a modular flow (Router → Planner → Executor → Synthesizer) rather than a linear chain.

Human-in-the-Loop (HITL): Sensitive tools (e.g., refunds) pause execution and trigger a local browser pop-up for manual agent approval before committing to the database.

Structured Outputs: Uses Pydantic to ensure precise tool argument extraction.

SQLite Backend: Uses WAL mode for concurrency; schema includes users, orders, invoices, and tickets.

Dependency Injection: Database connections are injected into tools to ensure transactional integrity.

Architecture
The agent operates as a directed graph:

Router: Classifies user intent and routes to the Tool Planner, Retrieval, or Chit-Chat.

Tool Planner: Maps intents to specific tools and extracts arguments (e.g., order_id).

Tool Executor: Executes Python functions and manages DB transactions.

Synthesizer: Generates the natural language response based on tool outputs.

Available Tools
get_order_status: Checks shipping status and delivery dates.

process_refund: HITL Enabled. Validates eligibility and requires human approval via pop-up.

check_account_status: Retrieves subscription plans and account standing.

upgrade_plan: Logic for handling subscription upgrades.

handoff_to_human: detects complex requests and logs a support ticket.

Installation
Prerequisites: Python 3.10+, Google Cloud API Key.

Clone the repository:

Bash

git clone https://github.com/yourusername/ecommerce-support-bot.git
cd ecommerce-support-bot
Set up virtual environment:

Bash

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install dependencies:

Bash

pip install -r requirements.txt
Configuration: Create a .env file in the root directory:

Ini, TOML

GOOGLE_API_KEY=your_actual_api_key_here
Database Setup
Run the data entry script to generate a SQLite database (data/demo.db) seeded with test users and orders.

Bash

python data_entry.py
Usage
Start the agent in console mode:

Bash
python main.py
Example Flow
User: "Where is order #1001?" Bot: "Order #1001 was shipped on 2025-12-15." User: "It arrived damaged. I want a refund." Bot: "Processing request... Waiting for agent approval." (Pop-up opens for manual approval) Bot: "Your refund for order #1001 has been processed."

python main.py
Example Flow
User: "Where is order #1001?" Bot: "Order #1001 was shipped on 2025-12-15." User: "It arrived damaged. I want a refund." Bot: "Processing request... Waiting for agent approval." (Pop-up opens for manual approval) Bot: "Your refund for order #1001 has been processed."
